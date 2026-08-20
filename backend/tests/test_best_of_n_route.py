"""`POST /api/analyze/{ticker}/best-of-n/start` + `GET .../best-of-n/stream/
{task_id}` 的SSE路由测试，跟 test_agent_stream_route.py 是同一套验证思路：
事件按预期顺序推送、异常变成error事件而不是非200状态码，断线重连接的是
同一个task_id。
"""

import asyncio
import json
from unittest.mock import patch

import httpx
from fastapi.testclient import TestClient

from app.main import app
from app.models.agent import AgentRunResult
from app.models.best_of_n import BestOfNResult, CandidateSummary, RuleScoreBreakdown

client = TestClient(app)


def _parse_sse_events(body: str) -> list[dict]:
    events = []
    for line in body.splitlines():
        if line.startswith("data: "):
            events.append(json.loads(line[len("data: ") :]))
    return events


def _rule_score() -> RuleScoreBreakdown:
    return RuleScoreBreakdown(
        traceability=50.0,
        traceability_matched=1,
        traceability_total=1,
        self_verification=20.0,
        structure=20.0,
        length=10.0,
        total=100.0,
    )


def _start(ticker: str = "AAPL") -> str:
    response = client.post(f"/api/analyze/{ticker}/best-of-n/start")
    assert response.status_code == 200
    return response.json()["task_id"]


def test_stream_route_emits_candidate_events_then_done():
    async def fake_run_best_of_n(ticker, on_event=None, trace_id=None):
        if on_event:
            await on_event({"type": "reasoning", "turn": 0, "candidate_index": 0, "text": "候选0推理中"})
            await on_event(
                {
                    "type": "candidate_scored",
                    "candidate_index": 0,
                    "temperature": 0.3,
                    "total_score": 100.0,
                    "rule_score": _rule_score().model_dump(),
                    "llm_score": 90.0,
                    "llm_reason": "很好",
                }
            )
        selected = AgentRunResult(
            ticker=ticker,
            completed=True,
            stop_reason="end_turn",
            final_report="<conclusion>结论</conclusion>",
            reasoning_notes=[],
            transcript=[],
            turns_used=1,
        )
        candidate = CandidateSummary(
            index=0,
            temperature=0.3,
            completed=True,
            final_report="<conclusion>结论</conclusion>",
            rule_score=_rule_score(),
            llm_score=90.0,
            llm_reason="很好",
            total_score=100.0,
        )
        return BestOfNResult(ticker=ticker, candidates=[candidate], selected_index=0, selected=selected)

    with patch("app.api.routes.best_of_n.run_best_of_n", new=fake_run_best_of_n):
        task_id = _start()
        response = client.get(f"/api/analyze/best-of-n/stream/{task_id}")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")

    events = _parse_sse_events(response.text)
    event_types = [e["type"] for e in events]
    assert event_types == ["reasoning", "candidate_scored", "done"]
    assert events[-1]["result"]["selected_index"] == 0
    assert events[-1]["result"]["selected"]["final_report"] == "<conclusion>结论</conclusion>"


def test_stream_route_emits_error_event_on_exception():
    async def fake_run_best_of_n(ticker, on_event=None, trace_id=None):
        raise RuntimeError("Best-of-N 运行失败")

    with patch("app.api.routes.best_of_n.run_best_of_n", new=fake_run_best_of_n):
        task_id = _start()
        response = client.get(f"/api/analyze/best-of-n/stream/{task_id}")

    assert response.status_code == 200
    events = _parse_sse_events(response.text)
    assert events == [{"type": "error", "message": "Best-of-N 运行失败"}]


def test_second_start_with_same_session_id_is_rejected_while_first_is_in_flight(monkeypatch):
    """跟test_agent_stream_route.py同一个场景、同一套解法（见那边的详细注释：
    用httpx.AsyncClient掌控唯一的事件循环，避免同步TestClient跨调用是否
    复用事件循环这个未定义行为导致的偶发挂起）。深度分析成本比普通分析更高
    （3次Agent Loop+最多3次裁判调用），同一个session更不该被意外双开。"""
    monkeypatch.setattr("app.api.session_guard._in_flight_sessions", set())
    still_running = asyncio.Event()

    async def fake_run_best_of_n(ticker, on_event=None, trace_id=None):
        await still_running.wait()
        selected = AgentRunResult(
            ticker=ticker,
            completed=True,
            stop_reason="end_turn",
            final_report="<conclusion>结论</conclusion>",
            reasoning_notes=[],
            transcript=[],
            turns_used=1,
        )
        candidate = CandidateSummary(
            index=0,
            temperature=0.3,
            completed=True,
            final_report="<conclusion>结论</conclusion>",
            rule_score=_rule_score(),
            llm_score=90.0,
            llm_reason="很好",
            total_score=100.0,
        )
        return BestOfNResult(ticker=ticker, candidates=[candidate], selected_index=0, selected=selected)

    async def run():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as async_client:
            first = await async_client.post(
                "/api/analyze/AAPL/best-of-n/start", headers={"X-Session-Id": "shared-session"}
            )
            second = await async_client.post(
                "/api/analyze/MSFT/best-of-n/start", headers={"X-Session-Id": "shared-session"}
            )
            still_running.set()
            await asyncio.sleep(0)
            return first, second

    with patch("app.api.routes.best_of_n.run_best_of_n", new=fake_run_best_of_n):
        first, second = asyncio.run(run())

    assert first.status_code == 200
    assert second.status_code == 409


def test_session_is_released_once_best_of_n_completes(monkeypatch):
    monkeypatch.setattr("app.api.session_guard._in_flight_sessions", set())

    async def fake_run_best_of_n(ticker, on_event=None, trace_id=None):
        selected = AgentRunResult(
            ticker=ticker,
            completed=True,
            stop_reason="end_turn",
            final_report="<conclusion>结论</conclusion>",
            reasoning_notes=[],
            transcript=[],
            turns_used=1,
        )
        candidate = CandidateSummary(
            index=0,
            temperature=0.3,
            completed=True,
            final_report="<conclusion>结论</conclusion>",
            rule_score=_rule_score(),
            llm_score=90.0,
            llm_reason="很好",
            total_score=100.0,
        )
        return BestOfNResult(ticker=ticker, candidates=[candidate], selected_index=0, selected=selected)

    with patch("app.api.routes.best_of_n.run_best_of_n", new=fake_run_best_of_n):
        first_response = client.post("/api/analyze/AAPL/best-of-n/start", headers={"X-Session-Id": "session-3"})
        client.get(f"/api/analyze/best-of-n/stream/{first_response.json()['task_id']}")

        second_response = client.post("/api/analyze/MSFT/best-of-n/start", headers={"X-Session-Id": "session-3"})

    assert second_response.status_code == 200


def test_a_normal_analysis_and_a_best_of_n_analysis_share_the_same_session_guard(monkeypatch):
    """两个路由共用同一份session状态（`session_guard._in_flight_sessions`）——
    同一个浏览器标签页不该同时既跑一次普通分析又跑一次深度分析，两者抢的是
    同一批Alpha Vantage/Polygon配额。"""
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
            normal = await async_client.post("/api/analyze/AAPL/start", headers={"X-Session-Id": "cross-route"})
            deep = await async_client.post(
                "/api/analyze/MSFT/best-of-n/start", headers={"X-Session-Id": "cross-route"}
            )
            still_running.set()
            await asyncio.sleep(0)
            return normal, deep

    with patch("app.api.routes.agent.run_agent_loop", new=fake_run_agent_loop):
        normal, deep = asyncio.run(run())

    assert normal.status_code == 200
    assert deep.status_code == 409
