"""alpha_vantage_client 的缓存命中/未命中 + "200状态码但响应体里塞错误提示"这个
Alpha Vantage 特有的坑（限速/无效key/未知symbol都不是用HTTP错误状态码表达的）。
"""

import asyncio
import json
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.services.alpha_vantage_client import AlphaVantageClientError, fetch_json, throttled_get


def test_fetch_json_reads_fresh_cache_without_network_call(tmp_path, monkeypatch):
    monkeypatch.setattr("app.services.alpha_vantage_client.CACHE_DIR", tmp_path)
    cache_file = tmp_path / "alphavantage_EARNINGS_AAPL.json"
    cache_file.write_text(json.dumps({"quarterlyEarnings": []}))

    mock_client = MagicMock()
    mock_client.get = AsyncMock(side_effect=AssertionError("不应该发起网络请求"))

    result = asyncio.run(fetch_json("EARNINGS", "AAPL", mock_client))

    assert result == {"quarterlyEarnings": []}
    mock_client.get.assert_not_called()


def test_fetch_json_ignores_stale_cache_and_refetches(tmp_path, monkeypatch):
    monkeypatch.setattr("app.services.alpha_vantage_client.CACHE_DIR", tmp_path)
    monkeypatch.setattr("app.core.config.settings.alpha_vantage_api_key", "fake-key")
    cache_file = tmp_path / "alphavantage_EARNINGS_AAPL.json"
    cache_file.write_text(json.dumps({"quarterlyEarnings": ["stale"]}))
    stale_time = (datetime.now() - timedelta(days=30)).timestamp()
    import os

    os.utime(cache_file, (stale_time, stale_time))

    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.json = MagicMock(return_value={"quarterlyEarnings": ["fresh"]})
    mock_client = MagicMock()
    mock_client.get = AsyncMock(return_value=mock_response)

    with patch("app.services.alpha_vantage_client.throttled_get", new=AsyncMock(return_value=mock_response)):
        result = asyncio.run(fetch_json("EARNINGS", "AAPL", mock_client))

    assert result == {"quarterlyEarnings": ["fresh"]}


def test_fetch_json_raises_on_rate_limit_note_in_200_response(tmp_path, monkeypatch):
    monkeypatch.setattr("app.services.alpha_vantage_client.CACHE_DIR", tmp_path)
    monkeypatch.setattr("app.core.config.settings.alpha_vantage_api_key", "fake-key")

    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.json = MagicMock(
        return_value={"Note": "Thank you for using Alpha Vantage! Our standard API rate limit is..."}
    )
    mock_client = MagicMock()

    with patch("app.services.alpha_vantage_client.throttled_get", new=AsyncMock(return_value=mock_response)):
        with pytest.raises(AlphaVantageClientError):
            asyncio.run(fetch_json("EARNINGS", "AAPL", mock_client))


def test_require_api_key_raises_when_not_configured(monkeypatch):
    monkeypatch.setattr("app.core.config.settings.alpha_vantage_api_key", "")
    from app.services.alpha_vantage_client import require_api_key

    with pytest.raises(AlphaVantageClientError):
        require_api_key()


def test_throttled_get_does_not_sleep_between_back_to_back_calls_within_quota(monkeypatch):
    """回归测试：这里曾经用"86400秒/25次"均匀摊开限速（照搬polygon_client.py那种
    按分钟计的限速模式），导致配额内的第二次调用也要硬等将近1小时——Alpha Vantage
    限的是每日配额，不是速率，两次调用只要在25次配额内就应该立刻放行，不应该有
    任何人为的sleep。"""
    monkeypatch.setattr("app.services.alpha_vantage_client._request_timestamps", [])

    mock_response = MagicMock()
    mock_client = MagicMock()
    mock_client.get = AsyncMock(return_value=mock_response)

    async def run_two_calls():
        await throttled_get(mock_client, "https://www.alphavantage.co/query", {})
        await throttled_get(mock_client, "https://www.alphavantage.co/query", {})

    start = datetime.now()
    asyncio.run(run_two_calls())
    elapsed = (datetime.now() - start).total_seconds()

    assert elapsed < 1.0
    assert mock_client.get.call_count == 2


def test_throttled_get_raises_immediately_once_daily_quota_exhausted(monkeypatch):
    """配额用满后应该立刻报错，而不是阻塞到下一个可用时间点——阻塞会让Agent Loop
    整个卡住且没有任何超时能打断它。"""
    now = datetime.now()
    monkeypatch.setattr("app.services.alpha_vantage_client._request_timestamps", [now] * 25)

    mock_client = MagicMock()
    mock_client.get = AsyncMock(side_effect=AssertionError("配额用满时不应该发起网络请求"))

    with pytest.raises(AlphaVantageClientError):
        asyncio.run(throttled_get(mock_client, "https://www.alphavantage.co/query", {}))


def test_throttled_get_prunes_timestamps_older_than_24_hours(monkeypatch):
    stale_timestamps = [datetime.now() - timedelta(days=2)] * 25
    monkeypatch.setattr("app.services.alpha_vantage_client._request_timestamps", stale_timestamps)

    mock_response = MagicMock()
    mock_client = MagicMock()
    mock_client.get = AsyncMock(return_value=mock_response)

    asyncio.run(throttled_get(mock_client, "https://www.alphavantage.co/query", {}))

    mock_client.get.assert_called_once()
