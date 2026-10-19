"""普通分析任务的启动/接回：新发起（`/start`）、进程重启后自动恢复
（task_recovery.py）、API故障后手动重试（`/tasks/{id}/retry`）三条路径共用
同一套runner——同一套检查点落盘、失败分类、会话锁释放逻辑，不会出现"新发起
的任务会存检查点，恢复出来的任务却不存"这类分叉。
"""

from __future__ import annotations

import asyncio
import hashlib
import time
import uuid
from datetime import UTC, datetime

from app.api.session_guard import release_in_flight_request
from app.core.config import settings
from app.services.agent import checkpoint, task_registry, trace_log
from app.services.agent.checkpoint import RetryableTaskError, Snapshot
from app.services.agent.llm_client import LLMResponse
from app.services.agent.loop import MAX_TURNS, run_agent_loop
from app.services.agent.system_prompt import SYSTEM_PROMPT

KIND_NORMAL = "normal"


class ResumeError(Exception):
    """存档存在但不能接着跑（版本不兼容/已过期/状态不对），message直接给用户看。"""


def _run_config() -> dict:
    return {
        "max_turns": MAX_TURNS,
        "temperature": None,
        "enable_compaction": True,
        # 下面三项只用于事后排查"恢复前后用的是不是同一套模型/Prompt"，
        # 恢复时不强制一致——部署新版本本身就是最常见的重启原因
        "provider": settings.llm_provider,
        "model": settings.llm_model,
        "prompt_sha": hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest()[:12],
    }


def start_normal_analysis(ticker: str, session_id: str | None) -> str:
    task_id = uuid.uuid4().hex
    started_at = time.time()
    config = _run_config()
    checkpoint.create_task(
        task_id, kind=KIND_NORMAL, ticker=ticker, session_id=session_id, started_at=started_at, config=config
    )
    _launch(task_id, ticker, session_id, config, started_at=started_at)
    return task_id


def resume_normal_analysis(task_id: str, *, reason: str, session_id: str | None) -> None:
    """从最新快照接着跑。`reason`是"server_restart"或"retry"，会写进`resumed`
    事件，前端据此显示不同的提示。不能接回时抛ResumeError。"""
    meta = checkpoint.read_meta(task_id)
    if meta is None or meta.get("kind") != KIND_NORMAL:
        raise ResumeError("找不到这次分析的检查点，请重新发起分析")
    if datetime.now(UTC) >= task_registry._next_trading_day_start(datetime.fromtimestamp(meta["started_at"], UTC)):
        raise ResumeError("检查点已过期（超过一个交易日），数据可能已经变化，请重新发起分析")
    snapshot = checkpoint.load_latest_snapshot(task_id)
    if snapshot is None:
        raise ResumeError("检查点版本不兼容或已损坏，无法恢复，请重新发起分析")

    initial_events = task_registry.rebuild_events_from_trace(task_id)[: snapshot.event_count]
    checkpoint.update_meta(task_id, status=checkpoint.STATUS_RUNNING, error=None, session_id=session_id)
    # 总用时只算真正在跑的时间：从原始发起到最后一次存档，加上恢复之后的时间——
    # 不算停机/等待用户点重试的空档，也不算存档之后被打断、要重做的那半截
    active_before = snapshot.saved_at - meta["started_at"]
    _launch(
        task_id,
        meta["ticker"],
        session_id,
        meta["config"],
        started_at=time.time() - active_before,
        resume=snapshot,
        resume_reason=reason,
        initial_events=initial_events,
    )


def _launch(
    task_id: str,
    ticker: str,
    session_id: str | None,
    config: dict,
    *,
    started_at: float,
    resume: Snapshot | None = None,
    resume_reason: str | None = None,
    initial_events: list[dict] | None = None,
) -> None:
    async def on_tool_result(tool_name: str, output: str) -> None:
        # task_registry对每个事件本身已经落一份到trace_log了，但那份是SSE同款
        # 的、截断到300字符的summary，事后排查"这个数字到底哪里对不上"意义
        # 有限——这里额外落完整未截断的工具输出，跟task_registry落的那份共用
        # 同一个trace_id文件，按发生顺序自然穿插
        trace_log.append_event(task_id, {"type": "tool_result_full", "tool_name": tool_name, "output": output})

    def on_checkpoint(phase: str, turn: int, state: checkpoint.LoopState, pending: LLMResponse | None) -> None:
        checkpoint.save_snapshot(
            task_id,
            phase=phase,
            turn=turn,
            state=state,
            pending_response=pending,
            event_count=task_registry.event_count(task_id),
        )

    async def runner(on_event: task_registry.OnEvent) -> dict:
        # session的"进行中"状态要跟着后台Agent Loop真正跑完的时机释放，不是
        # 跟着/start这次HTTP请求返回的时机——try/finally覆盖runner整个生命周期
        try:
            if resume is not None:
                await on_event(
                    {
                        "type": "resumed",
                        "reason": resume_reason,
                        "turn": resume.resume_point.turn,
                        "phase": resume.resume_point.phase,
                        # 预填进缓冲区的历史事件已经截断到存档点，这个值给
                        # rebuild_events_from_trace在trace日志里做同样的截断
                        "truncate_to": len(initial_events or []),
                    }
                )
            result = await run_agent_loop(
                ticker,
                max_turns=config["max_turns"],
                temperature=config["temperature"],
                enable_compaction=config["enable_compaction"],
                on_event=on_event,
                on_tool_result=on_tool_result,
                on_checkpoint=on_checkpoint,
                resume_from=resume.resume_point if resume is not None else None,
            )
        except asyncio.CancelledError:
            # 用户主动取消：检查点没用了，删掉。进程关闭/--reload导致的取消：
            # 保留，状态还是running，下次启动时自动恢复
            if task_registry.is_cancel_requested(task_id):
                checkpoint.delete_task(task_id)
            raise
        except Exception as exc:
            if checkpoint.is_retryable_error(exc) and checkpoint.list_snapshots(task_id):
                checkpoint.update_meta(
                    task_id, status=checkpoint.STATUS_FAILED_RETRYABLE, error=str(exc), failed_at=time.time()
                )
                raise RetryableTaskError(f"{exc}（已保存进度，可以从断点重试）") from exc
            checkpoint.update_meta(task_id, status=checkpoint.STATUS_FAILED, error=str(exc), failed_at=time.time())
            raise
        else:
            # 成功：trace日志已经足够还原这次运行，快照（含完整对话历史，单个
            # 任务可达几MB）不再保留
            checkpoint.delete_task(task_id)
            return result.model_dump()
        finally:
            release_in_flight_request(session_id)

    task_registry.start_task(ticker, runner, task_id=task_id, initial_events=initial_events, started_at=started_at)
