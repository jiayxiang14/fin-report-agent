"""GET /api/analyze/{ticker}/best-of-n/stream 的SSE路由测试，跟
test_agent_stream_route.py 是同一套验证思路：事件按预期顺序推送、
异常变成error事件而不是非200状态码。
"""

import json
from unittest.mock import patch

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


def test_stream_route_emits_candidate_events_then_done():
    async def fake_run_best_of_n(ticker, on_event=None):
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
        response = client.get("/api/analyze/AAPL/best-of-n/stream")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")

    events = _parse_sse_events(response.text)
    event_types = [e["type"] for e in events]
    assert event_types == ["reasoning", "candidate_scored", "done"]
    assert events[-1]["result"]["selected_index"] == 0
    assert events[-1]["result"]["selected"]["final_report"] == "<conclusion>结论</conclusion>"


def test_stream_route_emits_error_event_on_exception():
    async def fake_run_best_of_n(ticker, on_event=None):
        raise RuntimeError("Best-of-N 运行失败")

    with patch("app.api.routes.best_of_n.run_best_of_n", new=fake_run_best_of_n):
        response = client.get("/api/analyze/AAPL/best-of-n/stream")

    assert response.status_code == 200
    events = _parse_sse_events(response.text)
    assert events == [{"type": "error", "message": "Best-of-N 运行失败"}]
