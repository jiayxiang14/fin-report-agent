"""共享的 Polygon 日线拉取（`fetch_daily_bars`）本身的错误包装：非404状态码/连接错误
都应该统一抛 PolygonClientError，不管调用方是 sector_rotation、price_reaction 还是
company_profile。各调用方自己把这个异常包装成各自领域异常的行为，在各自模块的测试里
覆盖（比如 test_upstream_error_wrapping.py 里的 sector_rotation 那个测试）。
"""

import asyncio
from unittest.mock import patch

import httpx
import pytest

from app.services.polygon_client import PolygonClientError, fetch_daily_bars

FAKE_TICKER = "ZZZZTEST"  # 确保命中不了任何真实的本地缓存文件


def _error_response(url: str, status_code: int = 503) -> httpx.Response:
    return httpx.Response(status_code, request=httpx.Request("GET", url), text="upstream error")


def test_fetch_daily_bars_wraps_non_404_status_as_polygon_client_error():
    async def fake_throttled_get(client, url, params):
        return _error_response(url)

    with patch("app.services.polygon_client.throttled_get", new=fake_throttled_get):
        with pytest.raises(PolygonClientError):
            asyncio.run(fetch_daily_bars(FAKE_TICKER, client=None))


def test_fetch_daily_bars_wraps_connection_error_as_polygon_client_error():
    async def fake_throttled_get(client, url, params):
        raise httpx.ConnectError("network is unreachable")

    with patch("app.services.polygon_client.throttled_get", new=fake_throttled_get):
        with pytest.raises(PolygonClientError):
            asyncio.run(fetch_daily_bars(FAKE_TICKER, client=None))


def test_fetch_daily_bars_raises_on_404():
    async def fake_throttled_get(client, url, params):
        return _error_response(url, status_code=404)

    with patch("app.services.polygon_client.throttled_get", new=fake_throttled_get):
        with pytest.raises(PolygonClientError):
            asyncio.run(fetch_daily_bars(FAKE_TICKER, client=None))


def test_fetch_daily_bars_returns_close_high_low_volume_columns(monkeypatch):
    monkeypatch.setattr("app.services.polygon_client.settings.polygon_api_key", "test-key")

    async def fake_throttled_get(client, url, params):
        payload = {
            "results": [
                {"t": 1785038400000, "c": 100.0, "h": 105.0, "l": 98.0, "v": 1_000_000},
                {"t": 1785124800000, "c": 102.0, "h": 103.0, "l": 99.0, "v": 1_200_000},
            ]
        }
        return httpx.Response(200, request=httpx.Request("GET", url), json=payload)

    with patch("app.services.polygon_client.throttled_get", new=fake_throttled_get):
        bars = asyncio.run(fetch_daily_bars("ZZZZTEST2", client=None))

    assert list(bars.columns) == ["close", "high", "low", "volume"]
    assert len(bars) == 2
    assert bars["close"].iloc[0] == 100.0
    assert bars["high"].iloc[1] == 103.0
