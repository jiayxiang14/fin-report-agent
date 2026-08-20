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


def test_fetch_daily_bars_date_is_timezone_independent(monkeypatch):
    """真实复现过的bug：Polygon日线的t字段是"这个交易日美东零点"换算成UTC的
    毫秒时间戳（真实拉取过AAPL数据验证：t=1786334400000对应UTC 04:00，也就是
    2026-08-10这天美东零点在夏令时下的表示），不是UTC零点。之前用
    datetime.fromtimestamp()不传时区，会按服务器本地系统时区解释——服务器
    时区是America/Los_Angeles这类西向偏移较大的时区时，04:00 UTC减7/8小时
    会跨到前一个自然日，日线日期系统性地少算一天。这个断言不依赖测试运行
    时的系统时区（不管在哪个时区跑pytest结果都应该一样），因为修复后是显式
    按America/New_York解析，不受服务器本地时区影响。"""
    monkeypatch.setattr("app.services.polygon_client.settings.polygon_api_key", "test-key")

    async def fake_throttled_get(client, url, params):
        payload = {"results": [{"t": 1786334400000, "c": 100.0, "h": 105.0, "l": 98.0, "v": 1_000_000}]}
        return httpx.Response(200, request=httpx.Request("GET", url), json=payload)

    with patch("app.services.polygon_client.throttled_get", new=fake_throttled_get):
        bars = asyncio.run(fetch_daily_bars("ZZZZTEST3", client=None))

    assert bars.index[0].date().isoformat() == "2026-08-10"
