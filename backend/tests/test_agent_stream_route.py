"""Stage 4：验证 GET /api/analyze/{ticker}/stream 这条SSE路由能把 run_agent_loop
的 on_event 回调正确转换成SSE事件流，并且异常会变成一个 error 事件而不是让连接
直接断掉或者返回非200状态码（SSE一旦开始推流，状态码已经定死是200了）。
"""

import json
from unittest.mock import patch

from fastapi.testclient import TestClient

from app.main import app
from app.models.agent import AgentRunResult

client = TestClient(app)


def _parse_sse_events(body: str) -> list[dict]:
    events = []
    for line in body.splitlines():
        if line.startswith("data: "):
            events.append(json.loads(line[len("data: ") :]))
    return events


def test_stream_route_emits_progress_events_then_done():
    async def fake_run_agent_loop(ticker, max_turns=8, on_event=None):
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
        response = client.get("/api/analyze/AAPL/stream")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")

    events = _parse_sse_events(response.text)
    event_types = [e["type"] for e in events]
    assert event_types == ["reasoning", "tool_call_started", "tool_call_finished", "done"]
    assert events[-1]["result"]["final_report"] == "<conclusion>结论</conclusion>"
    assert events[-1]["result"]["completed"] is True


def test_stream_route_emits_error_event_on_exception():
    async def fake_run_agent_loop(ticker, max_turns=8, on_event=None):
        raise RuntimeError("LLM 调用失败")

    with patch("app.api.routes.agent.run_agent_loop", new=fake_run_agent_loop):
        response = client.get("/api/analyze/AAPL/stream")

    assert response.status_code == 200  # SSE已经开始推流，状态码只能是200
    events = _parse_sse_events(response.text)
    assert events == [{"type": "error", "message": "LLM 调用失败"}]
