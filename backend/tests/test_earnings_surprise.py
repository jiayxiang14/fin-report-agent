"""get_earnings_surprise 的回归测试：verdict 是纯代码判断（不经过LLM），核心逻辑是
"surprise 的正负号 -> 超预期/低于预期/符合预期"这个映射，以及缺数据时的软降级。
"""

import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from app.services.earnings_surprise import (
    EarningsSurpriseError,
    _classify_verdict,
    _parse_float,
    get_earnings_surprise,
)


def test_classify_verdict_beats_estimate():
    assert _classify_verdict(0.15) == "超预期"


def test_classify_verdict_misses_estimate():
    assert _classify_verdict(-0.3) == "低于预期"


def test_classify_verdict_inline_within_epsilon():
    assert _classify_verdict(0.005) == "符合预期"


def test_classify_verdict_none_when_no_surprise_data():
    assert _classify_verdict(None) is None


def test_parse_float_handles_none_string():
    assert _parse_float("None") is None
    assert _parse_float(None) is None
    assert _parse_float("2.65") == 2.65


def test_get_earnings_surprise_picks_latest_quarter_and_computes_verdict():
    payload = {
        "quarterlyEarnings": [
            {
                "fiscalDateEnding": "2025-03-31",
                "reportedDate": "2025-04-24",
                "reportedEPS": "2.40",
                "estimatedEPS": "2.50",
                "surprise": "-0.10",
                "surprisePercentage": "-4.0",
            },
            {
                "fiscalDateEnding": "2025-06-30",
                "reportedDate": "2025-07-23",
                "reportedEPS": "2.65",
                "estimatedEPS": "2.50",
                "surprise": "0.15",
                "surprisePercentage": "6.0",
            },
        ]
    }
    with patch(
        "app.services.earnings_surprise.fetch_json", new=AsyncMock(return_value=payload)
    ):
        result = asyncio.run(get_earnings_surprise("AAPL"))

    assert result.has_data is True
    assert result.fiscal_date_ending == "2025-06-30"  # 最新一期，不是列表第一条
    assert result.reported_eps == 2.65
    assert result.estimated_eps == 2.50
    assert result.verdict == "超预期"


def test_get_earnings_surprise_returns_no_data_when_quarterly_empty():
    with patch(
        "app.services.earnings_surprise.fetch_json",
        new=AsyncMock(return_value={"quarterlyEarnings": []}),
    ):
        result = asyncio.run(get_earnings_surprise("NOPE"))

    assert result.has_data is False
    assert result.verdict is None


def test_get_earnings_surprise_wraps_client_error():
    from app.services.alpha_vantage_client import AlphaVantageClientError

    with patch(
        "app.services.earnings_surprise.fetch_json",
        new=AsyncMock(side_effect=AlphaVantageClientError("缺少 ALPHA_VANTAGE_API_KEY")),
    ):
        with pytest.raises(EarningsSurpriseError):
            asyncio.run(get_earnings_surprise("AAPL"))
