"""进程内任务注册表：把"发起一次分析"和"订阅这次分析的事件流"解耦，断线
重连接的是同一个后台task，不会重复触发真实花钱的LLM调用。这里直接测
`task_registry`模块本身（不经过HTTP路由层，`test_agent_stream_route.py`/
`test_best_of_n_route.py`已经覆盖了路由层的集成场景），覆盖过期策略、
重放机制、以及"订阅一个正在跑的task能不能收到后续新事件"这个核心行为。
"""

import asyncio
from datetime import UTC, datetime, timedelta
from unittest.mock import patch

from app.services.agent import task_registry, trace_log


def test_next_trading_day_start_skips_weekend():
    # 2026-08-14 是周五
    friday = datetime(2026, 8, 14, 15, 0, tzinfo=UTC)
    next_start = task_registry._next_trading_day_start(friday)

    assert next_start == datetime(2026, 8, 17, 0, 0, tzinfo=UTC)  # 跳过周六周日，落在下周一


def test_next_trading_day_start_from_a_weekday_is_just_tomorrow():
    # 2026-08-17 是周一
    monday = datetime(2026, 8, 17, 9, 0, tzinfo=UTC)
    next_start = task_registry._next_trading_day_start(monday)

    assert next_start == datetime(2026, 8, 18, 0, 0, tzinfo=UTC)


def test_start_task_returns_a_unique_task_id_each_time():
    async def runner(on_event):
        return {"ok": True}

    async def run():
        return task_registry.start_task("AAPL", runner), task_registry.start_task("AAPL", runner)

    id_a, id_b = asyncio.run(run())

    assert id_a != id_b


async def _collect(task_id: str) -> list[dict]:
    return [event async for event in task_registry._replay_and_tail(task_id)]


def test_subscribing_to_unknown_task_id_yields_a_single_error_event():
    events = asyncio.run(_collect("does-not-exist"))

    assert events == [
        {"type": "error", "code": "task_not_found", "message": "task_id 不存在或已过期，请重新发起分析"}
    ]


def test_late_subscriber_receives_full_history_replay_then_live_events():
    """先跑完一个task（产生3个事件+done），断开之后再订阅同一个task_id——
    应该收到完整的历史重放，不用等新事件（因为task已经跑完了）。"""

    async def runner(on_event):
        await on_event({"type": "reasoning", "turn": 0, "text": "推理中"})
        await on_event({"type": "tool_call_started", "turn": 0, "tool_name": "get_financials"})
        return {"final_report": "ok"}

    async def run():
        task_id = task_registry.start_task("AAPL", runner)
        # 给后台任务一次调度机会跑完——测试环境下没有真实I/O延迟，一次
        # sleep(0)级别的让步就够
        await asyncio.sleep(0.05)
        return await _collect(task_id)

    events = asyncio.run(run())

    assert [e["type"] for e in events] == ["reasoning", "tool_call_started", "done"]
    assert events[-1]["result"] == {"final_report": "ok"}


def test_terminal_events_carry_server_side_elapsed_ms():
    """前端刷新页面接回一个早已跑完的task时，没法自己算出这次分析的真实用时
    （只知道发起时间，算出来的会把跑完之后的空闲时间也算进去），所以总用时
    必须由后端在done/error事件里给出。"""

    async def slow_runner(on_event):
        await asyncio.sleep(0.03)
        return {"final_report": "ok"}

    async def failing_runner(on_event):
        await asyncio.sleep(0.03)
        raise RuntimeError("boom")

    async def run():
        done_id = task_registry.start_task("AAPL", slow_runner)
        error_id = task_registry.start_task("AAPL", failing_runner)
        await asyncio.sleep(0.1)
        return await _collect(done_id), await _collect(error_id)

    done_events, error_events = asyncio.run(run())

    assert done_events[-1]["type"] == "done"
    assert 30 <= done_events[-1]["elapsed_ms"] < 1000
    assert error_events[-1]["type"] == "error"
    assert 30 <= error_events[-1]["elapsed_ms"] < 1000


def test_subscriber_connected_before_completion_receives_events_as_they_happen():
    """订阅发生在task还没跑完的时候——不是"跑完了再读"，是"边跑边收"，
    模拟真实场景里前端在分析进行中就已经连着的情况。"""

    async def run():
        started = asyncio.Event()
        finish = asyncio.Event()

        async def runner(on_event):
            await on_event({"type": "reasoning", "turn": 0, "text": "第一步"})
            started.set()
            await finish.wait()
            await on_event({"type": "reasoning", "turn": 1, "text": "第二步"})
            return {"final_report": "ok"}

        task_id = task_registry.start_task("AAPL", runner)

        collected: list[dict] = []

        async def subscriber():
            async for event in task_registry._replay_and_tail(task_id):
                collected.append(event)

        sub_task = asyncio.create_task(subscriber())
        await started.wait()
        finish.set()
        await sub_task
        return collected

    events = asyncio.run(run())

    assert [e["type"] for e in events] == ["reasoning", "reasoning", "done"]


def test_start_task_persists_every_event_to_trace_log(tmp_path):
    """task过期被清理、或者进程重启之后，`_tasks`这个内存字典里的历史就没了——
    这就是trace_log存在的意义：每个经过on_event的事件都该额外落一份到磁盘，
    按task_id（当trace_id用，不发明第二个标识符）事后能查。"""

    async def runner(on_event):
        await on_event({"type": "reasoning", "turn": 0, "text": "推理中"})
        await on_event({"type": "tool_call_started", "turn": 0, "tool_name": "get_financials"})
        return {"final_report": "ok"}

    async def run():
        task_id = task_registry.start_task("AAPL", runner)
        await asyncio.sleep(0.05)
        return task_id

    with patch.object(trace_log, "TRACE_DIR", tmp_path):
        task_id = asyncio.run(run())
        events = trace_log.read_trace(task_id)

    assert [e["type"] for e in events] == ["reasoning", "tool_call_started", "done"]


def test_start_task_accepts_a_caller_supplied_task_id():
    """agent.py的路由需要在调用start_task之前就知道task_id（好把它闭包进
    on_tool_result回调里），这里验证传入的task_id会被原样使用，不会被
    内部生成的另一个值覆盖掉。"""

    async def runner(on_event):
        return {"ok": True}

    async def run():
        return task_registry.start_task("AAPL", runner, task_id="a" * 32)

    task_id = asyncio.run(run())

    assert task_id == "a" * 32
    assert "a" * 32 in task_registry._tasks


def test_evicted_task_is_no_longer_subscribable():
    """过期的task应该被当成不存在处理，不能因为进程一直没重启就无限期占着
    内存——这里直接把expires_at改到过去模拟"已经过期"，不用真的等一个交易日。"""

    async def runner(on_event):
        return {"ok": True}

    async def run():
        task_id = task_registry.start_task("AAPL", runner)
        task_registry._tasks[task_id].expires_at = datetime.now(UTC) - timedelta(seconds=1)
        # 触发一次_evict_expired（start_task/后面的订阅都会顺带清理），这里
        # 直接调用新的start_task来触发清理，比等真实定时任务更直接
        task_registry.start_task("MSFT", runner)
        return task_id

    task_id = asyncio.run(run())

    assert task_id not in task_registry._tasks


def test_completed_task_no_longer_in_memory_is_replayed_from_trace_log(tmp_path):
    """进程重启后内存里的任务表是空的——已经跑完的task应该能从trace日志重放，
    而不是让用户看到"找不到任务"。"""
    task_id = "f" * 32
    with patch.object(trace_log, "TRACE_DIR", tmp_path):
        trace_log.append_event(task_id, {"type": "reasoning", "turn": 0, "text": "a"})
        trace_log.append_event(task_id, {"type": "tool_result_full", "tool_name": "x", "output": "{}"})
        trace_log.append_event(task_id, {"type": "done", "result": {"final_report": "ok"}})
        events = asyncio.run(_collect(task_id))

    assert events == [{"type": "reasoning", "turn": 0, "text": "a"}, {"type": "done", "result": {"final_report": "ok"}}]


def test_unfinished_task_in_trace_log_is_still_reported_as_not_found(tmp_path):
    task_id = "f" * 32
    with patch.object(trace_log, "TRACE_DIR", tmp_path):
        trace_log.append_event(task_id, {"type": "reasoning", "turn": 0, "text": "a"})
        events = asyncio.run(_collect(task_id))

    assert events[-1]["code"] == "task_not_found"


def test_rebuild_from_trace_truncates_at_resumed_markers(tmp_path):
    task_id = "f" * 32
    with patch.object(trace_log, "TRACE_DIR", tmp_path):
        for event in [
            {"type": "reasoning", "text": "a"},
            {"type": "reasoning", "text": "半截"},
            {"type": "error", "message": "502"},
            {"type": "resumed", "truncate_to": 1},
            {"type": "reasoning", "text": "b"},
        ]:
            trace_log.append_event(task_id, event)
        events = task_registry.rebuild_events_from_trace(task_id)

    assert [e["type"] for e in events] == ["reasoning", "resumed", "reasoning"]
    assert [e.get("text") for e in events] == ["a", None, "b"]
