"""Agent Loop 检查点：每轮存一次完整运行状态，支持进程崩溃/重启后从存档点
接着跑、API 持续故障后手动重试，以及事后按轮次排查"模型当时看到了什么"。

只服务走 task_registry 的普通分析（有 task_id 可以接回）。Best-of-N 和同步
接口 `POST /api/analyze/{ticker}` 不存检查点。

**两个存档点**（`phase`）：
- `turn_start`：每轮开头、调用 LLM 之前。被质量检查打回的那几次 `continue`
  会在下一轮开头自然存档，不用单独处理。
- `llm_done`：LLM 已经返回、开始执行工具之前——最贵的一步刚做完，崩在工具
  执行期间时恢复只重跑工具，不再付一次 LLM 的钱。

8 个工具全是只读查询、没有副作用，所以崩在一轮中间时整轮（或整段工具调用）
重做是安全的。

**存储布局**（一个任务一个目录，全部是 JSON，写入一律"先写临时文件再
`os.replace`"，崩在写一半时不会留下损坏文件）：

    .cache/checkpoints/{task_id}/
        meta.json                       任务级信息：状态、ticker、session_id、配置、失败原因
        turn-03-1-turn_start.json       逐轮快照，不覆盖，文件名字典序 = 时间顺序
        turn-03-2-llm_done.json

逐轮快照不覆盖，是为了排查时能看到每一轮的完整状态，而不是只剩最后一份。
任务**成功**时整个目录删除（trace 日志已经足够还原一次成功运行，快照含完整
对话历史，单个任务可达几 MB，全留着会无限增长）；**失败**时保留，按
`FAILED_RETENTION` 过期清理；用户**取消**时删除。

**任务状态**（`meta.status`）：
- `running`：正在跑。进程重启后看到的 `running` 就是"跑到一半被打断"，启动时自动恢复。
- `failed_retryable`：LLM/网络这类外部故障，保留存档，等用户手动重试。
- `failed`：代码 bug、请求本身有问题这类重试也没用的失败，只保留快照供排查。

落盘失败绝不拖垮正在跑的分析（跟 trace_log 同一个原则）：检查点是容错能力，
不是核心链路，写失败只记日志。
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import anthropic

from app.models.agent import ReasoningNote, TranscriptEntry
from app.services.agent.llm_client import LLMResponse, ToolCall
from app.services.polygon_client import CACHE_DIR

if TYPE_CHECKING:
    from anthropic.types import Message

logger = logging.getLogger(__name__)

CHECKPOINT_DIR = CACHE_DIR / "checkpoints"

# 状态结构变了（LoopState加减字段、phase语义变化）就加1——版本对不上的存档
# 不恢复，因为没法保证旧状态在新代码里的含义还一样
SCHEMA_VERSION = 1

# 自动恢复次数上限：防止某个任务每次都把进程搞崩，每次重启又恢复它、又崩一遍。
# 只限制"启动时自动恢复"，用户手动重试不计入（那是人主动决定再花一次钱）
MAX_AUTO_RESUMES = 2

# 失败任务的快照保留多久供排查，过期后在启动扫描时清理
FAILED_RETENTION_SECONDS = 7 * 24 * 3600

# 请求本身有问题（参数不合法），原样重试必然再失败，不算可重试
_NON_RETRYABLE_STATUS_CODES = frozenset({400, 422})

_TASK_ID_PATTERN = re.compile(r"^[0-9a-f]{32}$")
_SNAPSHOT_PATTERN = re.compile(r"^turn-\d{2}-[12]-(turn_start|llm_done)\.json$")

STATUS_RUNNING = "running"
STATUS_FAILED_RETRYABLE = "failed_retryable"
STATUS_FAILED = "failed"


@dataclass
class LoopState:
    """`run_agent_loop` 里所有跨轮次存活的可变状态——恢复时原样还原，循环
    就能像没被打断过一样接着跑。"""

    messages: list[dict[str, Any]]
    transcript: list[TranscriptEntry] = field(default_factory=list)
    reasoning_notes: list[ReasoningNote] = field(default_factory=list)
    raw_tool_outputs: list[str] = field(default_factory=list)
    verify_number_outcomes: list[tuple[str, str, bool | None]] = field(default_factory=list)
    # verify_number核实过的数值（简报声称的值和数据源的实际值都算），带默认值，旧检查点照常恢复
    verified_values: list[float] = field(default_factory=list)
    tool_call_names: dict[str, str] = field(default_factory=dict)
    nudged_for_gate_batch: bool = False
    nudged_for_ignored_format: bool = False
    nudged_for_reflexion: bool = False
    reflexion_triggered: bool = False
    structure_gate_triggered: bool = False
    tool_coverage_gate_triggered: bool = False
    verification_mismatch_triggered: bool = False
    traceability_gate_triggered: bool = False
    sentiment_consistency_gate_triggered: bool = False
    # 新增字段一律给默认值：旧快照里没有这个键时用默认值还原，不需要升SCHEMA_VERSION
    data_gap_disclosure_gate_triggered: bool = False
    verification_coverage_gate_triggered: bool = False
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cache_creation_input_tokens: int = 0
    total_cache_read_input_tokens: int = 0
    last_model: str = ""


@dataclass
class ResumePoint:
    phase: str  # "turn_start" | "llm_done"
    turn: int
    state: LoopState
    # 只在 phase == "llm_done" 时有值：那一轮 LLM 已经返回的内容，恢复时
    # 跳过 LLM 调用直接用它往下走（assistant 消息本身已经在 state.messages 里）
    pending_response: LLMResponse | None = None


@dataclass
class Snapshot:
    resume_point: ResumePoint
    event_count: int  # 存档那一刻已经推送过多少个 SSE 事件，恢复时把事件历史截断到这里
    saved_at: float


class RetryableTaskError(Exception):
    """runner 把可重试的外部故障包成这个异常抛给 task_registry，后者据此在
    error 事件里带上 `retryable: true`，前端显示"从断点重试"按钮。"""

    retryable = True


def is_retryable_error(exc: BaseException) -> bool:
    """LLM 供应商/网络层的故障（超时、断连、限流、余额不足、5xx）重试有意义；
    请求本身不合法（400/422）和其它所有异常（多半是代码 bug）重试没用。"""
    if isinstance(exc, anthropic.APIStatusError):
        return exc.status_code not in _NON_RETRYABLE_STATUS_CODES
    return isinstance(exc, anthropic.APIConnectionError)  # APITimeoutError 是它的子类


# ---------- 序列化 ----------


def _jsonable_block(block: Any) -> Any:
    # assistant 消息的 content 是 SDK 对象（TextBlock/ToolUseBlock/ThinkingBlock），
    # 不能直接存 JSON。exclude_none 去掉 citations=None 这类空字段；
    # ThinkingBlock 的 signature 会保留——回传时供应商要校验它。已经用真实
    # DeepSeek 请求验证过：转成 dict 回传后对话能正常继续
    if hasattr(block, "model_dump"):
        return block.model_dump(exclude_none=True)
    return block


def _jsonable_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for message in messages:
        content = message.get("content")
        if isinstance(content, list):
            content = [_jsonable_block(block) for block in content]
        result.append({**message, "content": content})
    return result


def _state_to_dict(state: LoopState) -> dict[str, Any]:
    data = asdict(state)
    data["messages"] = _jsonable_messages(state.messages)
    data["transcript"] = [entry.model_dump() for entry in state.transcript]
    data["reasoning_notes"] = [note.model_dump() for note in state.reasoning_notes]
    data["verify_number_outcomes"] = [list(outcome) for outcome in state.verify_number_outcomes]
    return data


def _state_from_dict(data: dict[str, Any]) -> LoopState:
    return LoopState(
        **{
            **data,
            "transcript": [TranscriptEntry(**entry) for entry in data["transcript"]],
            "reasoning_notes": [ReasoningNote(**note) for note in data["reasoning_notes"]],
            "verify_number_outcomes": [tuple(outcome) for outcome in data["verify_number_outcomes"]],
        }
    )


def _response_to_dict(response: LLMResponse) -> dict[str, Any]:
    return {
        "stop_reason": response.stop_reason,
        "text": response.text,
        "tool_calls": [asdict(call) for call in response.tool_calls],
        "model": response.model,
    }


def _response_from_dict(data: dict[str, Any]) -> LLMResponse:
    return LLMResponse(
        stop_reason=data["stop_reason"],
        text=data["text"],
        tool_calls=[ToolCall(**call) for call in data["tool_calls"]],
        # 恢复出来的响应不需要原始SDK对象：assistant消息本身已经在state.messages
        # 里了，raw只被append_assistant_turn用，恢复路径不会再调用它
        raw=cast("Message", None),
        model=data.get("model", ""),
    )


# ---------- 文件读写 ----------


def _task_dir(task_id: str) -> Path | None:
    if not _TASK_ID_PATTERN.match(task_id):
        return None
    return CHECKPOINT_DIR / task_id


def _atomic_write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False))
    os.replace(tmp, path)


def create_task(
    task_id: str,
    *,
    kind: str,
    ticker: str,
    session_id: str | None,
    started_at: float,
    config: dict[str, Any],
) -> None:
    task_dir = _task_dir(task_id)
    if task_dir is None:
        return
    try:
        _atomic_write_json(
            task_dir / "meta.json",
            {
                "schema_version": SCHEMA_VERSION,
                "task_id": task_id,
                "kind": kind,
                "ticker": ticker,
                "session_id": session_id,
                "started_at": started_at,
                "status": STATUS_RUNNING,
                "auto_resume_count": 0,
                "config": config,
                "error": None,
            },
        )
    except Exception:  # noqa: BLE001 - 检查点是容错能力，不是核心链路，写失败不拖垮分析
        logger.exception("创建检查点失败 task_id=%s", task_id)


def read_meta(task_id: str) -> dict[str, Any] | None:
    task_dir = _task_dir(task_id)
    if task_dir is None:
        return None
    try:
        return json.loads((task_dir / "meta.json").read_text())
    except (OSError, json.JSONDecodeError):
        return None


def update_meta(task_id: str, **fields: Any) -> None:
    task_dir = _task_dir(task_id)
    meta = read_meta(task_id)
    if task_dir is None or meta is None:
        return
    try:
        _atomic_write_json(task_dir / "meta.json", {**meta, **fields})
    except Exception:  # noqa: BLE001 - 同 create_task
        logger.exception("更新检查点元信息失败 task_id=%s", task_id)


def save_snapshot(
    task_id: str,
    *,
    phase: str,
    turn: int,
    state: LoopState,
    pending_response: LLMResponse | None,
    event_count: int,
) -> None:
    task_dir = _task_dir(task_id)
    if task_dir is None:
        return
    order = 1 if phase == "turn_start" else 2
    try:
        _atomic_write_json(
            task_dir / f"turn-{turn:02d}-{order}-{phase}.json",
            {
                "schema_version": SCHEMA_VERSION,
                "phase": phase,
                "turn": turn,
                "event_count": event_count,
                "saved_at": time.time(),
                "state": _state_to_dict(state),
                "pending_response": _response_to_dict(pending_response) if pending_response else None,
            },
        )
    except Exception:  # noqa: BLE001 - 同 create_task
        logger.exception("保存检查点快照失败 task_id=%s turn=%s phase=%s", task_id, turn, phase)


def list_snapshots(task_id: str) -> list[Path]:
    task_dir = _task_dir(task_id)
    if task_dir is None or not task_dir.exists():
        return []
    return sorted(p for p in task_dir.iterdir() if _SNAPSHOT_PATTERN.match(p.name))


def load_latest_snapshot(task_id: str) -> Snapshot | None:
    """读最新的一份快照。版本不对或文件损坏时返回 None（调用方据此判定
    "无法恢复"），不抛异常。"""
    snapshots = list_snapshots(task_id)
    if not snapshots:
        return None
    try:
        data = json.loads(snapshots[-1].read_text())
        if data.get("schema_version") != SCHEMA_VERSION:
            return None
        pending = data.get("pending_response")
        return Snapshot(
            resume_point=ResumePoint(
                phase=data["phase"],
                turn=data["turn"],
                state=_state_from_dict(data["state"]),
                pending_response=_response_from_dict(pending) if pending else None,
            ),
            event_count=data["event_count"],
            saved_at=data["saved_at"],
        )
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError):
        logger.exception("读取检查点快照失败 task_id=%s", task_id)
        return None


def delete_task(task_id: str) -> None:
    task_dir = _task_dir(task_id)
    if task_dir is None:
        return
    shutil.rmtree(task_dir, ignore_errors=True)


def list_task_ids() -> list[str]:
    if not CHECKPOINT_DIR.exists():
        return []
    return sorted(p.name for p in CHECKPOINT_DIR.iterdir() if p.is_dir() and _TASK_ID_PATTERN.match(p.name))
