"""`POST /api/analyze/tasks/{task_id}/cancel` 和 `/retry` 的路由层行为：
参数校验、状态检查、会话锁、以及"失败 → 重试 → 同一条流接着跑完"的完整链路。"""

import json
from unittest.mock import patch

import anthropic
import httpx
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.models.agent import AgentRunResult
from app.services.agent import checkpoint, trace_log
from app.services.agent.checkpoint import LoopState

client = TestClient(app)


@pytest.fixture(autouse=True)
def _isolate_trace_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(trace_log, "TRACE_DIR", tmp_path / "traces")


def _events(task_id: str) -> list[dict]:
    body = client.get(f"/api/analyze/stream/{task_id}").text
    return [json.loads(line[len("data: ") :]) for line in body.splitlines() if line.startswith("data: ")]


def _result() -> AgentRunResult:
    return AgentRunResult(
        ticker="AAPL", completed=True, stop_reason="end_turn", final_report="<conclusion>ok</conclusion>",
        reasoning_notes=[], transcript=[], turns_used=1,
    )


async def _failing_loop(ticker, on_event=None, on_checkpoint=None, resume_from=None, **_):
    if resume_from is None:
        await on_event({"type": "reasoning", "turn": 0, "text": "a"})
        on_checkpoint("turn_start", 1, LoopState(messages=[{"role": "user", "content": "x"}]), None)
        raise anthropic.APIConnectionError(request=httpx.Request("POST", "https://example.invalid"))
    await on_event({"type": "reasoning", "turn": 1, "text": "b"})
    return _result()


def test_failed_task_can_be_retried_and_continues_on_the_same_stream():
    with patch("app.services.agent.task_runner.run_agent_loop", new=_failing_loop):
        task_id = client.post("/api/analyze/AAPL/start", headers={"X-Session-Id": "s1"}).json()["task_id"]
        first = _events(task_id)
        retry = client.post(f"/api/analyze/tasks/{task_id}/retry", headers={"X-Session-Id": "s1"})
        second = _events(task_id)

    assert first[-1]["type"] == "error"
    assert first[-1]["retryable"] is True
    assert retry.status_code == 200
    assert retry.json() == {"task_id": task_id}
    assert [e["type"] for e in second] == ["reasoning", "resumed", "reasoning", "done"]


def test_retry_is_rejected_when_the_failure_is_not_retryable_and_releases_the_session_lock():
    async def buggy_loop(ticker, on_event=None, on_checkpoint=None, **_):
        on_checkpoint("turn_start", 0, LoopState(messages=[]), None)
        raise RuntimeError("bug")

    with patch("app.services.agent.task_runner.run_agent_loop", new=buggy_loop):
        task_id = client.post("/api/analyze/AAPL/start").json()["task_id"]
        _events(task_id)
        response = client.post(f"/api/analyze/tasks/{task_id}/retry", headers={"X-Session-Id": "s2"})

    assert response.status_code == 409
    assert "不支持从断点重试" in response.json()["detail"]
    # 被拒绝的重试不能把会话锁占着不放，否则这个标签页再也发不起新分析
    with patch("app.services.agent.task_runner.run_agent_loop", new=_failing_loop):
        assert client.post("/api/analyze/AAPL/start", headers={"X-Session-Id": "s2"}).status_code == 200


def test_retry_of_an_expired_checkpoint_returns_410():
    with patch("app.services.agent.task_runner.run_agent_loop", new=_failing_loop):
        task_id = client.post("/api/analyze/AAPL/start").json()["task_id"]
        _events(task_id)
    checkpoint.update_meta(task_id, started_at=0.0)

    response = client.post(f"/api/analyze/tasks/{task_id}/retry")

    assert response.status_code == 410
    assert "过期" in response.json()["detail"]


def test_retry_of_unknown_task_returns_409():
    response = client.post(f"/api/analyze/tasks/{'e' * 32}/retry")
    assert response.status_code == 409


def test_cancel_of_unknown_or_finished_task_returns_404():
    assert client.post(f"/api/analyze/tasks/{'e' * 32}/cancel").status_code == 404


@pytest.mark.parametrize("action", ["cancel", "retry"])
def test_task_id_format_is_validated(action):
    assert client.post(f"/api/analyze/tasks/not-a-task-id/{action}").status_code == 422
