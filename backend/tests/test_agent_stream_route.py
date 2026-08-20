"""Stage 4 + 断线重连改造：验证 `POST /api/analyze/{ticker}/start` +
`GET /api/analyze/stream/{task_id}` 这一对路由能把 run_agent_loop 的
on_event 回调正确转换成SSE事件流，并且异常会变成一个 error 事件而不是让
连接直接断掉或者返回非200状态码（SSE一旦开始推流，状态码已经定死是200了）。

之前是一个 `GET /{ticker}/stream` 端点同时做"发起分析"和"订阅事件流"两件
事，断线重连等于重新触发一次真实花钱的分析。现在拆成 `/start`（发起，返回
task_id）+ `/stream/{task_id}`（订阅，可以断了重连），这里额外补了"订阅一个
不存在的task_id"和"重连到已经跑完的task能收到完整历史"两个新场景。
"""

import asyncio
import json
from unittest.mock import patch

import httpx
from fastapi.testclient import TestClient

from app.main import app
from app.models.agent import AgentRunResult
from app.services.agent import trace_log

client = TestClient(app)


def _parse_sse_events(body: str) -> list[dict]:
    events = []
    for line in body.splitlines():
        if line.startswith("data: "):
            events.append(json.loads(line[len("data: ") :]))
    return events


def _start(ticker: str = "AAPL") -> str:
    response = client.post(f"/api/analyze/{ticker}/start")
    assert response.status_code == 200
    return response.json()["task_id"]


def test_stream_route_emits_progress_events_then_done():
    async def fake_run_agent_loop(ticker, max_turns=8, on_event=None, on_tool_result=None):
        if on_event:
            await on_event({"type": "reasoning", "turn": 0, "text": "先查财务数据"})
            await on_event(
                {
                    "type": "tool_call_started",
                    "turn": 0,
                    "tool_name": "get_financials",
                    "tool_input": {"ticker": ticker},
                }
            )
            await on_event(
                {
                    "type": "tool_call_finished",
                    "turn": 0,
                    "tool_name": "get_financials",
                    "is_error": False,
                    "summary": "{...}",
                }
            )
        return AgentRunResult(
            ticker=ticker,
            completed=True,
            stop_reason="end_turn",
            final_report="<conclusion>结论</conclusion>",
            reasoning_notes=[],
            transcript=[],
            turns_used=1,
        )

    with patch("app.api.routes.agent.run_agent_loop", new=fake_run_agent_loop):
        task_id = _start()
        response = client.get(f"/api/analyze/stream/{task_id}")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")

    events = _parse_sse_events(response.text)
    event_types = [e["type"] for e in events]
    assert event_types == ["reasoning", "tool_call_started", "tool_call_finished", "done"]
    assert events[-1]["result"]["final_report"] == "<conclusion>结论</conclusion>"
    assert events[-1]["result"]["completed"] is True


def test_stream_route_emits_error_event_on_exception():
    async def fake_run_agent_loop(ticker, max_turns=8, on_event=None, on_tool_result=None):
        raise RuntimeError("LLM 调用失败")

    with patch("app.api.routes.agent.run_agent_loop", new=fake_run_agent_loop):
        task_id = _start()
        response = client.get(f"/api/analyze/stream/{task_id}")

    assert response.status_code == 200  # SSE已经开始推流，状态码只能是200
    events = _parse_sse_events(response.text)
    assert events == [{"type": "error", "message": "LLM 调用失败"}]


def test_stream_route_returns_error_event_for_unknown_task_id():
    """task_id不存在（乱猜的、或者已经过期被清理掉了）——不能让前端的
    EventSource死等一个永远不会来的响应，直接发一个error事件说明情况，
    前端可以据此判断"这个task接不回去了，需要重新发起"。"""
    response = client.get("/api/analyze/stream/does-not-exist")

    assert response.status_code == 200
    events = _parse_sse_events(response.text)
    assert events == [
        {"type": "error", "code": "task_not_found", "message": "task_id 不存在或已过期，请重新发起分析"}
    ]


def test_start_persists_full_untruncated_tool_output_via_trace_log(tmp_path):
    """SSE事件里的tool_call_finished只带_summarize截断到300字符的summary——
    事后排查"这个数字到底哪里对不上"意义有限。路由应该把run_agent_loop的
    on_tool_result钩子接到trace_log，额外落一条带完整未截断输出的记录，
    跟task_registry落的那份summary共用同一个trace_id文件。"""
    long_output = "x" * 500  # 明显超过SUMMARY_LENGTH=300，确认没有被截断

    async def fake_run_agent_loop(ticker, max_turns=8, on_event=None, on_tool_result=None):
        if on_tool_result:
            await on_tool_result("get_financials", long_output)
        return AgentRunResult(
            ticker=ticker,
            completed=True,
            stop_reason="end_turn",
            final_report="<conclusion>结论</conclusion>",
            reasoning_notes=[],
            transcript=[],
            turns_used=1,
        )

    with (
        patch("app.api.routes.agent.run_agent_loop", new=fake_run_agent_loop),
        patch.object(trace_log, "TRACE_DIR", tmp_path),
    ):
        task_id = _start()
        events = trace_log.read_trace(task_id)

    full_output_events = [e for e in events if e["type"] == "tool_result_full"]
    assert len(full_output_events) == 1
    assert full_output_events[0]["output"] == long_output
    assert full_output_events[0]["tool_name"] == "get_financials"


def test_second_start_with_same_session_id_is_rejected_while_first_is_in_flight(monkeypatch):
    """真实模拟"同一个session几乎同时点了两次开始分析"（StrictMode双开、
    或者用户手快点两下）：第二次/start在第一次的后台Agent Loop还没跑完时
    发生，应该被明确拒绝，不是悄悄跑两次真实花钱的分析。

    改用httpx.AsyncClient直接跑在一个由本测试自己掌控的事件循环上（不经过
    同步版TestClient）——同步TestClient每次调用背后是否复用同一个事件循环
    是未定义的实现细节，跨调用用threading.Event同步会绑定到这个不确定性上，
    真实跑过会偶发挂起。这里用普通asyncio.Event，全程只有一个事件循环，
    "先卡住第一个请求的后台task，同时发第二个请求，再放行"这个时序是完全
    确定的。
    """
    monkeypatch.setattr("app.api.session_guard._in_flight_sessions", set())
    still_running = asyncio.Event()

    async def fake_run_agent_loop(ticker, max_turns=8, on_event=None, on_tool_result=None):
        await still_running.wait()
        return AgentRunResult(
            ticker=ticker,
            completed=True,
            stop_reason="end_turn",
            final_report="<conclusion>结论</conclusion>",
            reasoning_notes=[],
            transcript=[],
            turns_used=1,
        )

    async def run():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as async_client:
            first = await async_client.post("/api/analyze/AAPL/start", headers={"X-Session-Id": "shared-session"})
            second = await async_client.post("/api/analyze/MSFT/start", headers={"X-Session-Id": "shared-session"})
            still_running.set()
            # 排空后台task，避免它带着一个仍处于patch上下文里的引用泄漏到下一个测试
            await asyncio.sleep(0)
            return first, second

    with patch("app.api.routes.agent.run_agent_loop", new=fake_run_agent_loop):
        first, second = asyncio.run(run())

    assert first.status_code == 200
    assert second.status_code == 409
    assert "已经有一个分析正在进行" in second.json()["detail"]


def test_session_is_released_once_the_background_analysis_completes(monkeypatch):
    monkeypatch.setattr("app.api.session_guard._in_flight_sessions", set())

    async def fake_run_agent_loop(ticker, max_turns=8, on_event=None, on_tool_result=None):
        return AgentRunResult(
            ticker=ticker,
            completed=True,
            stop_reason="end_turn",
            final_report="<conclusion>结论</conclusion>",
            reasoning_notes=[],
            transcript=[],
            turns_used=1,
        )

    with patch("app.api.routes.agent.run_agent_loop", new=fake_run_agent_loop):
        first = client.post("/api/analyze/AAPL/start", headers={"X-Session-Id": "shared-session-2"})
        # 完整拉一次stream，确保后台task真的跑完了（_replay_and_tail只有在
        # state.done之后才会让生成器结束，TestClient.get()会等生成器耗尽）
        client.get(f"/api/analyze/stream/{first.json()['task_id']}")

        second = client.post("/api/analyze/MSFT/start", headers={"X-Session-Id": "shared-session-2"})

    assert first.status_code == 200
    assert second.status_code == 200


def test_start_without_a_session_id_header_is_never_blocked(monkeypatch):
    monkeypatch.setattr("app.api.session_guard._in_flight_sessions", set())

    async def fake_run_agent_loop(ticker, max_turns=8, on_event=None, on_tool_result=None):
        return AgentRunResult(
            ticker=ticker,
            completed=True,
            stop_reason="end_turn",
            final_report="<conclusion>结论</conclusion>",
            reasoning_notes=[],
            transcript=[],
            turns_used=1,
        )

    with patch("app.api.routes.agent.run_agent_loop", new=fake_run_agent_loop):
        first = client.post("/api/analyze/AAPL/start")
        second = client.post("/api/analyze/MSFT/start")

    assert first.status_code == 200
    assert second.status_code == 200


def test_reconnecting_after_completion_replays_full_history():
    """真实模拟"断线重连"这个场景：第一次订阅完整收完事件流（模拟页面
    正常跑完），断开连接之后再订阅同一个task_id——因为task_registry里
    events是按task持久保留的（进程内，不因为没人订阅就清空），第二次订阅
    应该能收到完整的历史事件，不是空的。"""

    async def fake_run_agent_loop(ticker, max_turns=8, on_event=None, on_tool_result=None):
        if on_event:
            await on_event({"type": "reasoning", "turn": 0, "text": "推理中"})
        return AgentRunResult(
            ticker=ticker,
            completed=True,
            stop_reason="end_turn",
            final_report="<conclusion>结论</conclusion>",
            reasoning_notes=[],
            transcript=[],
            turns_used=1,
        )

    with patch("app.api.routes.agent.run_agent_loop", new=fake_run_agent_loop):
        task_id = _start()
        first = client.get(f"/api/analyze/stream/{task_id}")
        second = client.get(f"/api/analyze/stream/{task_id}")

    first_events = _parse_sse_events(first.text)
    second_events = _parse_sse_events(second.text)
    assert first_events == second_events
    assert [e["type"] for e in second_events] == ["reasoning", "done"]
