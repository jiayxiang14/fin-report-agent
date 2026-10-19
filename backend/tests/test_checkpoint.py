"""checkpoint.py：序列化往返、原子写入、读取最新快照、失败分类。"""

import json

import anthropic
import httpx
from pydantic import BaseModel

from app.models.agent import ReasoningNote, TranscriptEntry
from app.services.agent import checkpoint
from app.services.agent.checkpoint import LoopState
from app.services.agent.llm_client import LLMResponse, ToolCall

TASK_ID = "a" * 32


class _FakeSdkBlock(BaseModel):
    """模拟 anthropic SDK 的 content block（pydantic 对象，带可选的 None 字段）。"""

    type: str
    text: str | None = None
    citations: list | None = None


def _state() -> LoopState:
    return LoopState(
        messages=[
            {"role": "user", "content": "分析 AAPL"},
            {"role": "assistant", "content": [_FakeSdkBlock(type="text", text="先查数据")]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "{}"}]},
        ],
        transcript=[TranscriptEntry(turn=0, tool_name="get_financials", tool_input={"ticker": "AAPL"}, tool_output_summary="{}", is_error=False)],
        reasoning_notes=[ReasoningNote(turn=0, text="先查数据")],
        raw_tool_outputs=["{}"],
        verify_number_outcomes=[("revenue", "FY2025", True), ("eps", "Q2", None)],
        tool_call_names={"t1": "get_financials"},
        nudged_for_gate_batch=True,
        structure_gate_triggered=True,
        total_input_tokens=1200,
        total_output_tokens=80,
        last_model="deepseek-v4-flash",
    )


def test_snapshot_round_trip_restores_full_state_and_converts_sdk_blocks_to_dicts():
    pending = LLMResponse(
        stop_reason="tool_use",
        text="再核查一下",
        tool_calls=[ToolCall(id="t2", name="verify_number", input={"metric": "revenue"})],
        raw=None,
        model="deepseek-v4-flash",
    )
    checkpoint.save_snapshot(TASK_ID, phase="llm_done", turn=1, state=_state(), pending_response=pending, event_count=7)

    snapshot = checkpoint.load_latest_snapshot(TASK_ID)

    assert snapshot is not None
    assert snapshot.event_count == 7
    point = snapshot.resume_point
    assert (point.phase, point.turn) == ("llm_done", 1)
    restored = point.state
    # SDK 对象转成了 dict，None 字段被去掉（回传给供应商时不带多余的空字段）
    assert restored.messages[1]["content"] == [{"type": "text", "text": "先查数据"}]
    assert restored.transcript == _state().transcript
    assert restored.reasoning_notes == _state().reasoning_notes
    assert restored.verify_number_outcomes == [("revenue", "FY2025", True), ("eps", "Q2", None)]
    assert restored.nudged_for_gate_batch is True
    assert restored.structure_gate_triggered is True
    assert restored.total_input_tokens == 1200
    assert point.pending_response is not None
    assert point.pending_response.tool_calls == pending.tool_calls
    assert point.pending_response.text == "再核查一下"


def test_snapshots_are_kept_per_turn_and_latest_is_the_last_one_in_time_order():
    for turn in range(3):
        checkpoint.save_snapshot(TASK_ID, phase="turn_start", turn=turn, state=_state(), pending_response=None, event_count=turn)
        checkpoint.save_snapshot(TASK_ID, phase="llm_done", turn=turn, state=_state(), pending_response=None, event_count=turn + 100)

    names = [p.name for p in checkpoint.list_snapshots(TASK_ID)]
    assert names == [
        "turn-00-1-turn_start.json",
        "turn-00-2-llm_done.json",
        "turn-01-1-turn_start.json",
        "turn-01-2-llm_done.json",
        "turn-02-1-turn_start.json",
        "turn-02-2-llm_done.json",
    ]
    snapshot = checkpoint.load_latest_snapshot(TASK_ID)
    assert snapshot is not None
    assert (snapshot.resume_point.turn, snapshot.resume_point.phase, snapshot.event_count) == (2, "llm_done", 102)


def test_writes_leave_no_temp_files_behind():
    checkpoint.create_task(TASK_ID, kind="normal", ticker="AAPL", session_id="s", started_at=1.0, config={})
    checkpoint.save_snapshot(TASK_ID, phase="turn_start", turn=0, state=_state(), pending_response=None, event_count=0)

    files = sorted(p.name for p in (checkpoint.CHECKPOINT_DIR / TASK_ID).iterdir())
    assert files == ["meta.json", "turn-00-1-turn_start.json"]


def test_schema_version_mismatch_is_treated_as_unrecoverable():
    checkpoint.save_snapshot(TASK_ID, phase="turn_start", turn=0, state=_state(), pending_response=None, event_count=0)
    path = checkpoint.list_snapshots(TASK_ID)[0]
    data = json.loads(path.read_text())
    data["schema_version"] = checkpoint.SCHEMA_VERSION + 1
    path.write_text(json.dumps(data))

    assert checkpoint.load_latest_snapshot(TASK_ID) is None


def test_corrupted_snapshot_is_treated_as_unrecoverable_instead_of_raising():
    checkpoint.save_snapshot(TASK_ID, phase="turn_start", turn=0, state=_state(), pending_response=None, event_count=0)
    checkpoint.list_snapshots(TASK_ID)[0].write_text("{not json")

    assert checkpoint.load_latest_snapshot(TASK_ID) is None


def test_meta_create_update_and_delete():
    checkpoint.create_task(TASK_ID, kind="normal", ticker="AAPL", session_id="s1", started_at=10.0, config={"max_turns": 8})
    checkpoint.update_meta(TASK_ID, status=checkpoint.STATUS_FAILED_RETRYABLE, error="402")

    meta = checkpoint.read_meta(TASK_ID)
    assert meta is not None
    assert meta["status"] == checkpoint.STATUS_FAILED_RETRYABLE
    assert meta["error"] == "402"
    assert meta["ticker"] == "AAPL"
    assert checkpoint.list_task_ids() == [TASK_ID]

    checkpoint.delete_task(TASK_ID)
    assert checkpoint.read_meta(TASK_ID) is None
    assert checkpoint.list_task_ids() == []


def test_invalid_task_ids_are_ignored_to_prevent_path_traversal():
    checkpoint.save_snapshot("../../etc", phase="turn_start", turn=0, state=_state(), pending_response=None, event_count=0)

    assert checkpoint.list_task_ids() == []
    assert checkpoint.read_meta("../../etc") is None


def _status_error(status: int) -> anthropic.APIStatusError:
    request = httpx.Request("POST", "https://example.invalid")
    return anthropic.APIStatusError("boom", response=httpx.Response(status, request=request), body=None)


def test_retryable_error_classification():
    request = httpx.Request("POST", "https://example.invalid")
    assert checkpoint.is_retryable_error(anthropic.APIConnectionError(request=request))
    assert checkpoint.is_retryable_error(anthropic.APITimeoutError(request=request))
    for status in (402, 429, 500, 503, 401):
        assert checkpoint.is_retryable_error(_status_error(status)), status
    # 请求本身不合法，原样重试必然再失败
    assert not checkpoint.is_retryable_error(_status_error(400))
    assert not checkpoint.is_retryable_error(_status_error(422))
    # 其它异常多半是代码 bug，重试没用
    assert not checkpoint.is_retryable_error(RuntimeError("bug"))
    assert not checkpoint.is_retryable_error(KeyError("x"))
