"""入站限流：保护`/api/analyze/*`这几个真的会触发LLM调用的端点，不被无限
调用打爆成本。用真实TestClient走完整的FastAPI依赖注入（不是单独测
`enforce_rate_limit`这个函数），确认限流是真的接在路由上，不是只在单测里
自证。
"""

from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

from app.api import rate_limit
from app.main import app

client = TestClient(app)


def test_requests_within_limit_are_not_blocked():
    with patch("app.api.routes.agent.run_agent_loop", new=AsyncMock(side_effect=RuntimeError("boom"))):
        for _ in range(rate_limit.RATE_LIMIT_MAX_REQUESTS):
            response = client.post("/api/analyze/AAPL")
            assert response.status_code != 429


def test_request_beyond_limit_is_rejected_with_429():
    with patch("app.api.routes.agent.run_agent_loop", new=AsyncMock(side_effect=RuntimeError("boom"))):
        for _ in range(rate_limit.RATE_LIMIT_MAX_REQUESTS):
            client.post("/api/analyze/AAPL")
        response = client.post("/api/analyze/AAPL")

    assert response.status_code == 429


def test_limit_is_shared_across_normal_and_best_of_n_endpoints():
    """两个路由文件（agent.py/best_of_n.py）各自是独立的APIRouter实例，但都
    依赖同一个`enforce_rate_limit`函数——这里验证限流状态确实是共享的，不是
    各路由各算各的，不然总的LLM调用成本还是防不住。"""
    with patch("app.api.routes.agent.run_agent_loop", new=AsyncMock(side_effect=RuntimeError("boom"))):
        for _ in range(rate_limit.RATE_LIMIT_MAX_REQUESTS):
            client.post("/api/analyze/AAPL")

    response = client.post("/api/analyze/AAPL/best-of-n/start")

    assert response.status_code == 429


def test_stream_subscribe_endpoint_is_exempt_from_rate_limit():
    """`/stream/{task_id}`只是订阅/重放已经在跑或跑完的task，不产生新的
    LLM调用——限流保护的是"发起分析"这个真花钱的动作（挂在`/start`上），
    重连重放不该占用户的限流额度，不然刷新页面几次就会把自己锁死。"""
    with patch("app.api.routes.agent.run_agent_loop", new=AsyncMock(side_effect=RuntimeError("boom"))):
        for _ in range(rate_limit.RATE_LIMIT_MAX_REQUESTS):
            client.post("/api/analyze/AAPL/start")
        # /start 本身应该已经被限流打到429了，但订阅一个（不存在的）task_id
        # 不该受影响——用一个乱猜的task_id，只关心状态码不是429
        response = client.get("/api/analyze/stream/does-not-exist")

    assert response.status_code == 200


def test_window_expiry_allows_requests_again(monkeypatch):
    fake_now = 1000.0
    monkeypatch.setattr(rate_limit.time, "monotonic", lambda: fake_now)

    with patch("app.api.routes.agent.run_agent_loop", new=AsyncMock(side_effect=RuntimeError("boom"))):
        for _ in range(rate_limit.RATE_LIMIT_MAX_REQUESTS):
            client.post("/api/analyze/AAPL")
        assert client.post("/api/analyze/AAPL").status_code == 429

        fake_now += rate_limit.RATE_LIMIT_WINDOW_SECONDS + 1
        response = client.post("/api/analyze/AAPL")

    assert response.status_code != 429


def test_different_client_ips_are_tracked_independently():
    with patch("app.api.routes.agent.run_agent_loop", new=AsyncMock(side_effect=RuntimeError("boom"))):
        for _ in range(rate_limit.RATE_LIMIT_MAX_REQUESTS):
            rate_limit.enforce_rate_limit(_FakeRequest("1.2.3.4"))
        blocked = _check_raises_429(_FakeRequest("1.2.3.4"))
        not_blocked = _check_raises_429(_FakeRequest("5.6.7.8"))

    assert blocked is True
    assert not_blocked is False


class _FakeClient:
    def __init__(self, host: str):
        self.host = host


class _FakeRequest:
    def __init__(self, host: str):
        self.client = _FakeClient(host)


def _check_raises_429(request: _FakeRequest) -> bool:
    from fastapi import HTTPException

    try:
        rate_limit.enforce_rate_limit(request)  # type: ignore[arg-type]
    except HTTPException as exc:
        return exc.status_code == 429
    return False
