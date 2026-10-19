"""task_runner.py：检查点落盘、失败分类、取消、从断点重试。直接在事件循环
里测服务层（不经过HTTP），后台task和订阅都在同一个asyncio.run里完成。"""

import asyncio
import time
from datetime import timedelta
from unittest.mock import patch

import anthropic
import httpx
import pytest

from app.api.session_guard import _in_flight_sessions
from app.models.agent import AgentRunResult
from app.services.agent import checkpoint, task_registry, task_runner, trace_log
from app.services.agent.checkpoint import LoopState


@pytest.fixture(autouse=True)
def _isolate_trace_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(trace_log, "TRACE_DIR", tmp_path / "traces")


def _result(ticker: str = "AAPL") -> AgentRunResult:
    return AgentRunResult(
        ticker=ticker, completed=True, stop_reason="end_turn", final_report="<conclusion>ok</conclusion>",
        reasoning_notes=[], transcript=[], turns_used=1,
    )


def _connection_error() -> anthropic.APIConnectionError:
    return anthropic.APIConnectionError(request=httpx.Request("POST", "https://example.invalid"))


async def _collect(task_id: str) -> list[dict]:
    return [event async for event in task_registry._replay_and_tail(task_id)]


def _types(events: list[dict]) -> list[str]:
    return [e["type"] for e in events]


async def _emit_and_checkpoint(on_event, on_checkpoint, *, before: list[str], after: list[str], turn: int = 0):
    """先推before里的事件，存一次档，再推after里的事件（模拟"存档之后、
    崩溃之前"推出去的半截事件）。"""
    for text in before:
        await on_event({"type": "reasoning", "turn": turn, "text": text})
    on_checkpoint("turn_start", turn, LoopState(messages=[{"role": "user", "content": "x"}]), None)
    for text in after:
        await on_event({"type": "reasoning", "turn": turn, "text": text})


def test_successful_run_saves_checkpoints_during_the_run_and_deletes_them_at_the_end():
    seen_snapshots = []

    async def fake_loop(ticker, on_event=None, on_tool_result=None, on_checkpoint=None, **_):
        await _emit_and_checkpoint(on_event, on_checkpoint, before=["a"], after=[])
        seen_snapshots.extend(p.name for p in checkpoint.list_snapshots(task_id_holder[0]))
        return _result(ticker)

    task_id_holder: list[str] = []

    async def run():
        task_id = task_runner.start_normal_analysis("AAPL", session_id=None)
        task_id_holder.append(task_id)
        return task_id, await _collect(task_id)

    with patch("app.services.agent.task_runner.run_agent_loop", new=fake_loop):
        task_id, events = asyncio.run(run())

    assert seen_snapshots == ["turn-00-1-turn_start.json"]
    assert _types(events) == ["reasoning", "done"]
    assert checkpoint.read_meta(task_id) is None  # 成功后整个检查点目录删除


def test_retryable_failure_keeps_checkpoint_and_marks_error_event_retryable():
    async def fake_loop(ticker, on_event=None, on_checkpoint=None, **_):
        await _emit_and_checkpoint(on_event, on_checkpoint, before=["a"], after=["b"])
        raise _connection_error()

    async def run():
        _in_flight_sessions.add("s1")
        task_id = task_runner.start_normal_analysis("AAPL", session_id="s1")
        return task_id, await _collect(task_id)

    with patch("app.services.agent.task_runner.run_agent_loop", new=fake_loop):
        task_id, events = asyncio.run(run())

    assert events[-1]["type"] == "error"
    assert events[-1]["retryable"] is True
    assert "可以从断点重试" in events[-1]["message"]
    meta = checkpoint.read_meta(task_id)
    assert meta is not None
    assert meta["status"] == checkpoint.STATUS_FAILED_RETRYABLE
    assert checkpoint.list_snapshots(task_id)
    assert "s1" not in _in_flight_sessions  # 失败也要释放会话锁


def test_non_retryable_failure_is_marked_failed_and_not_offered_a_retry():
    async def fake_loop(ticker, on_event=None, on_checkpoint=None, **_):
        await _emit_and_checkpoint(on_event, on_checkpoint, before=["a"], after=[])
        raise RuntimeError("代码bug")

    async def run():
        task_id = task_runner.start_normal_analysis("AAPL", session_id=None)
        return task_id, await _collect(task_id)

    with patch("app.services.agent.task_runner.run_agent_loop", new=fake_loop):
        task_id, events = asyncio.run(run())

    assert events[-1] == {"type": "error", "message": "代码bug", "elapsed_ms": events[-1]["elapsed_ms"]}
    meta = checkpoint.read_meta(task_id)
    assert meta is not None
    assert meta["status"] == checkpoint.STATUS_FAILED
    assert checkpoint.list_snapshots(task_id)  # 快照保留，供排查


def test_retryable_error_before_any_snapshot_is_not_offered_a_retry():
    async def fake_loop(ticker, **_):
        raise _connection_error()

    async def run():
        task_id = task_runner.start_normal_analysis("AAPL", session_id=None)
        return task_id, await _collect(task_id)

    with patch("app.services.agent.task_runner.run_agent_loop", new=fake_loop):
        task_id, events = asyncio.run(run())

    assert "retryable" not in events[-1]
    assert checkpoint.read_meta(task_id)["status"] == checkpoint.STATUS_FAILED


def test_user_cancel_emits_cancelled_event_deletes_checkpoint_and_releases_session():
    async def fake_loop(ticker, on_event=None, on_checkpoint=None, **_):
        await _emit_and_checkpoint(on_event, on_checkpoint, before=["a"], after=[])
        await asyncio.Event().wait()  # 永远等下去，直到被取消

    async def run():
        _in_flight_sessions.add("s1")
        task_id = task_runner.start_normal_analysis("AAPL", session_id="s1")
        await asyncio.sleep(0.01)
        assert task_registry.cancel_task(task_id) is True
        return task_id, await _collect(task_id)

    with patch("app.services.agent.task_runner.run_agent_loop", new=fake_loop):
        task_id, events = asyncio.run(run())

    assert events[-1]["type"] == "error"
    assert events[-1]["code"] == "cancelled"
    assert checkpoint.read_meta(task_id) is None
    assert "s1" not in _in_flight_sessions
    assert task_registry.cancel_task(task_id) is False  # 已经结束，再取消返回False


def test_cancellation_not_requested_by_user_keeps_checkpoint_running_for_recovery():
    """进程关闭/--reload 会取消后台task——不是用户的决定，不发终止事件，
    检查点保持 running，下次启动时自动恢复。"""

    async def fake_loop(ticker, on_event=None, on_checkpoint=None, **_):
        await _emit_and_checkpoint(on_event, on_checkpoint, before=["a"], after=[])
        await asyncio.Event().wait()

    async def run():
        task_id = task_runner.start_normal_analysis("AAPL", session_id=None)
        await asyncio.sleep(0.01)
        handle = task_registry._tasks[task_id].handle
        handle.cancel()
        with pytest.raises(asyncio.CancelledError):
            await handle
        return task_id

    with patch("app.services.agent.task_runner.run_agent_loop", new=fake_loop):
        task_id = asyncio.run(run())

    meta = checkpoint.read_meta(task_id)
    assert meta is not None
    assert meta["status"] == checkpoint.STATUS_RUNNING
    assert all(e["type"] != "error" for e in task_registry._tasks[task_id].events)


def test_retry_resumes_from_snapshot_with_events_truncated_to_the_save_point():
    received = {}

    async def failing_loop(ticker, on_event=None, on_checkpoint=None, **_):
        await _emit_and_checkpoint(on_event, on_checkpoint, before=["a", "b"], after=["半截"])
        raise _connection_error()

    async def resumed_loop(ticker, on_event=None, on_checkpoint=None, resume_from=None, **_):
        received["resume_from"] = resume_from
        await on_event({"type": "reasoning", "turn": 0, "text": "c"})
        return _result(ticker)

    async def run():
        with patch("app.services.agent.task_runner.run_agent_loop", new=failing_loop):
            task_id = task_runner.start_normal_analysis("AAPL", session_id=None)
            await _collect(task_id)
        with patch("app.services.agent.task_runner.run_agent_loop", new=resumed_loop):
            task_runner.resume_normal_analysis(task_id, reason="retry", session_id=None)
            return task_id, await _collect(task_id)

    task_id, events = asyncio.run(run())

    texts = [e.get("text") for e in events if e["type"] == "reasoning"]
    assert texts == ["a", "b", "c"]  # "半截"和失败的error事件都被截掉了
    resumed = next(e for e in events if e["type"] == "resumed")
    assert resumed["reason"] == "retry"
    assert resumed["truncate_to"] == 2
    assert events[-1]["type"] == "done"
    assert received["resume_from"].turn == 0
    # trace日志重建出来的序列跟订阅者看到的一致（重启后从trace重放靠它）
    assert task_registry.rebuild_events_from_trace(task_id) == events
    assert checkpoint.read_meta(task_id) is None


def test_resumed_task_elapsed_time_excludes_downtime(monkeypatch):
    async def resumed_loop(ticker, **_):
        return _result(ticker)

    # 这个测试只关心用时，不让"跨过交易日零点"把检查点判成过期
    monkeypatch.setattr(task_registry, "_next_trading_day_start", lambda dt: dt + timedelta(days=365))
    task_id = "b" * 32
    now = time.time()
    # 原始发起于1小时前，最后一次存档时已经跑了10分钟——中间50分钟是停机
    checkpoint.create_task(
        task_id, kind="normal", ticker="AAPL", session_id=None, started_at=now - 3600, config=task_runner._run_config()
    )
    with patch("app.services.agent.checkpoint.time.time", return_value=now - 3000):
        checkpoint.save_snapshot(
            task_id, phase="turn_start", turn=1, state=LoopState(messages=[]), pending_response=None, event_count=0
        )

    async def run():
        with patch("app.services.agent.task_runner.run_agent_loop", new=resumed_loop):
            task_runner.resume_normal_analysis(task_id, reason="server_restart", session_id=None)
            return await _collect(task_id)

    events = asyncio.run(run())

    assert 600_000 <= events[-1]["elapsed_ms"] <= 610_000


def test_resume_refuses_expired_or_missing_checkpoints():
    task_id = "c" * 32
    with pytest.raises(task_runner.ResumeError, match="找不到"):
        task_runner.resume_normal_analysis(task_id, reason="retry", session_id=None)

    checkpoint.create_task(task_id, kind="normal", ticker="AAPL", session_id=None, started_at=time.time() - 10 * 86400, config={})
    with pytest.raises(task_runner.ResumeError, match="过期"):
        task_runner.resume_normal_analysis(task_id, reason="retry", session_id=None)

    checkpoint.update_meta(task_id, started_at=time.time())
    with pytest.raises(task_runner.ResumeError, match="版本不兼容或已损坏"):
        task_runner.resume_normal_analysis(task_id, reason="retry", session_id=None)
