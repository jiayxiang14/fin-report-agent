"""task_recovery.py：服务启动时自动接回"跑到一半被打断"的普通分析。

每个测试先手工造出"上一个进程留下的现场"：检查点目录（meta + 快照）和
trace日志（上一个进程推出去的事件，包括存档之后的半截事件），然后在新的
事件循环里调用 recover_pending_tasks()，模拟进程重启。
"""

import asyncio
import time
from unittest.mock import patch

import pytest

from app.api.session_guard import _in_flight_sessions
from app.models.agent import AgentRunResult
from app.services import task_recovery
from app.services.agent import checkpoint, task_registry, task_runner, trace_log
from app.services.agent.checkpoint import LoopState

TASK_ID = "d" * 32


@pytest.fixture(autouse=True)
def _isolate_trace_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(trace_log, "TRACE_DIR", tmp_path / "traces")


def _leave_interrupted_task(*, status=checkpoint.STATUS_RUNNING, auto_resume_count=0, started_at=None, session_id="s1"):
    checkpoint.create_task(
        TASK_ID,
        kind="normal",
        ticker="AAPL",
        session_id=session_id,
        started_at=started_at if started_at is not None else time.time() - 60,
        config=task_runner._run_config(),
    )
    checkpoint.update_meta(TASK_ID, status=status, auto_resume_count=auto_resume_count, failed_at=time.time())
    for text in ["a", "b"]:
        trace_log.append_event(TASK_ID, {"type": "reasoning", "turn": 0, "text": text})
    trace_log.append_event(TASK_ID, {"type": "tool_result_full", "tool_name": "get_financials", "output": "{}"})
    checkpoint.save_snapshot(
        TASK_ID, phase="turn_start", turn=1, state=LoopState(messages=[{"role": "user", "content": "x"}]), pending_response=None, event_count=2
    )
    # 存档之后、进程被杀之前推出去的半截事件
    trace_log.append_event(TASK_ID, {"type": "reasoning_delta", "turn": 1, "text": "半截"})


async def _collect(task_id: str) -> list[dict]:
    return [event async for event in task_registry._replay_and_tail(task_id)]


def _result() -> AgentRunResult:
    return AgentRunResult(
        ticker="AAPL", completed=True, stop_reason="end_turn", final_report="<conclusion>ok</conclusion>",
        reasoning_notes=[], transcript=[], turns_used=2,
    )


def test_interrupted_task_is_resumed_from_its_latest_snapshot_with_partial_events_dropped():
    _leave_interrupted_task()
    received = {}

    async def fake_loop(ticker, on_event=None, resume_from=None, **_):
        received["resume_turn"] = resume_from.turn
        received["session_locked"] = "s1" in _in_flight_sessions
        received["meta_during_run"] = checkpoint.read_meta(TASK_ID)
        await on_event({"type": "reasoning", "turn": 1, "text": "c"})
        return _result()

    async def run():
        resumed = task_recovery.recover_pending_tasks()
        return resumed, await _collect(TASK_ID)

    with patch("app.services.agent.task_runner.run_agent_loop", new=fake_loop):
        resumed, events = asyncio.run(run())

    assert resumed == [TASK_ID]
    assert [e["type"] for e in events] == ["reasoning", "reasoning", "resumed", "reasoning", "done"]
    assert [e.get("text") for e in events if e["type"] == "reasoning"] == ["a", "b", "c"]
    assert events[2]["reason"] == "server_restart"
    assert received["resume_turn"] == 1
    assert received["session_locked"] is True  # 重启后重新占住原来的会话锁
    assert received["meta_during_run"]["auto_resume_count"] == 1
    assert "s1" not in _in_flight_sessions  # 跑完释放
    assert checkpoint.read_meta(TASK_ID) is None


def test_gives_up_after_too_many_automatic_resumes_and_explains_why_on_replay():
    _leave_interrupted_task(auto_resume_count=checkpoint.MAX_AUTO_RESUMES)

    async def run():
        resumed = task_recovery.recover_pending_tasks()
        return resumed, await _collect(TASK_ID)

    with patch("app.services.agent.task_runner.run_agent_loop") as loop_mock:
        resumed, events = asyncio.run(run())

    assert resumed == []
    loop_mock.assert_not_called()
    assert checkpoint.read_meta(TASK_ID)["status"] == checkpoint.STATUS_FAILED
    # 进程内存里没有这个task，订阅时从trace日志重放：前面跑到哪 + 为什么停了
    assert events[-1]["type"] == "error"
    assert events[-1]["code"] == "resume_failed"
    assert "自动恢复" in events[-1]["message"]
    assert [e.get("text") for e in events if e["type"] == "reasoning"] == ["a", "b"]


def test_expired_checkpoint_is_not_resumed():
    _leave_interrupted_task(started_at=time.time() - 10 * 86400)

    with patch("app.services.agent.task_runner.run_agent_loop") as loop_mock:
        resumed = asyncio.run(_recover())

    assert resumed == []
    loop_mock.assert_not_called()
    meta = checkpoint.read_meta(TASK_ID)
    assert meta["status"] == checkpoint.STATUS_FAILED
    assert "过期" in meta["error"]


def test_incompatible_snapshot_is_not_resumed():
    _leave_interrupted_task()
    path = checkpoint.list_snapshots(TASK_ID)[0]
    path.write_text(path.read_text().replace(f'"schema_version": {checkpoint.SCHEMA_VERSION}', '"schema_version": 999', 1))

    with patch("app.services.agent.task_runner.run_agent_loop") as loop_mock:
        resumed = asyncio.run(_recover())

    assert resumed == []
    loop_mock.assert_not_called()
    assert "版本不兼容" in checkpoint.read_meta(TASK_ID)["error"]
    assert "s1" not in _in_flight_sessions


def test_failed_retryable_tasks_wait_for_manual_retry_instead_of_auto_resuming():
    _leave_interrupted_task(status=checkpoint.STATUS_FAILED_RETRYABLE)

    with patch("app.services.agent.task_runner.run_agent_loop") as loop_mock:
        resumed = asyncio.run(_recover())

    assert resumed == []
    loop_mock.assert_not_called()
    assert checkpoint.read_meta(TASK_ID)["status"] == checkpoint.STATUS_FAILED_RETRYABLE


def test_old_failed_checkpoints_are_cleaned_up_and_recent_ones_kept():
    _leave_interrupted_task(status=checkpoint.STATUS_FAILED)
    asyncio.run(_recover())
    assert checkpoint.read_meta(TASK_ID) is not None  # 7天内保留供排查

    checkpoint.update_meta(TASK_ID, failed_at=time.time() - checkpoint.FAILED_RETENTION_SECONDS - 1)
    asyncio.run(_recover())
    assert checkpoint.read_meta(TASK_ID) is None


def test_directory_without_meta_is_removed():
    checkpoint.save_snapshot(TASK_ID, phase="turn_start", turn=0, state=LoopState(messages=[]), pending_response=None, event_count=0)

    asyncio.run(_recover())

    assert checkpoint.list_task_ids() == []


async def _recover() -> list[str]:
    return task_recovery.recover_pending_tasks()
