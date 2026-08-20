"""公司概览（简介/市值/细分行业/20日ADR）的回归测试。ADR计算和"查不到ticker时的降级"
是这里最值得覆盖的两块，Polygon响应本身的解析用mock隔离，不发真实网络请求。
"""

import asyncio
from unittest.mock import AsyncMock, patch

import pandas as pd
import pytest

from app.models.financials import FinancialMetric, FinancialsResponse, MetricPoint
from app.services.company_profile import (
    CompanyProfileError,
    _compute_adr_pct,
    get_company_profile,
)
from app.services.polygon_client import PolygonClientError
from app.services.sec_client import SecClientError


def _fake_financials(eps: float | None, unit: str = "USD/shares") -> FinancialsResponse:
    metrics = {}
    if eps is not None:
        metrics["eps_diluted"] = FinancialMetric(
            tag="EarningsPerShareDiluted",
            label="稀释每股收益",
            unit=unit,
            latest_annual=MetricPoint(end="2025-12-31", val=eps, form="10-K", filed="2026-02-01"),
        )
    return FinancialsResponse(
        ticker="AMZN",
        cik="0001018724",
        entity_name="Amazon.Com Inc",
        metrics=metrics,
        retrieved_at="2026-08-04T00:00:00Z",
    )


def _bars(closes, highs, lows) -> pd.DataFrame:
    dates = pd.bdate_range(start="2026-06-01", periods=len(closes))
    return pd.DataFrame({"close": closes, "high": highs, "low": lows, "volume": [0] * len(closes)}, index=dates)


def test_compute_adr_pct_averages_last_20_trading_days():
    # 前面故意放一堆振幅很小的数据，最后20天振幅固定是收盘价的10%
    closes = [100.0] * 30 + [100.0] * 20
    highs = [100.5] * 30 + [105.0] * 20
    lows = [99.5] * 30 + [95.0] * 20
    bars = _bars(closes, highs, lows)

    adr = _compute_adr_pct(bars, window=20)

    assert adr == 10.0  # (105-95)/100*100 = 10%，前面的小振幅数据不应该拉低这个值


def test_get_company_profile_returns_has_data_false_when_ticker_not_found():
    with patch(
        "app.services.company_profile.fetch_ticker_details",
        new=AsyncMock(return_value=None),
    ):
        result = asyncio.run(get_company_profile("NOPE"))

    assert result.has_data is False
    assert result.note is not None


def test_get_company_profile_happy_path():
    details = {
        "name": "Amazon.Com Inc",
        "description": "在线零售与云计算公司",
        "market_cap": 2_921_415_780_628.88,
        "sic_description": "RETAIL-CATALOG & MAIL-ORDER HOUSES",
        "homepage_url": "https://www.amazon.com",
        "total_employees": 1_576_000,
    }
    bars = _bars([100.0] * 25, [105.0] * 25, [95.0] * 25)

    with (
        patch("app.services.company_profile.fetch_ticker_details", new=AsyncMock(return_value=details)),
        patch("app.services.company_profile.fetch_daily_bars", new=AsyncMock(return_value=bars)),
        patch(
            "app.services.company_profile.get_financials",
            new=AsyncMock(return_value=_fake_financials(eps=5.0)),
        ),
    ):
        result = asyncio.run(get_company_profile("AMZN"))

    assert result.has_data is True
    assert result.name == "Amazon.Com Inc"
    assert result.market_cap == 2_921_415_780_628.88
    assert result.adr_20d_pct == 10.0
    assert result.pe_ratio == 20.0  # 收盘价100.0 / EPS 5.0


def test_get_company_profile_wraps_bars_fetch_error():
    details = {"name": "Amazon.Com Inc"}
    with (
        patch("app.services.company_profile.fetch_ticker_details", new=AsyncMock(return_value=details)),
        patch(
            "app.services.company_profile.fetch_daily_bars",
            new=AsyncMock(side_effect=PolygonClientError("上游错误")),
        ),
    ):
        with pytest.raises(CompanyProfileError):
            asyncio.run(get_company_profile("AMZN"))


def test_pe_ratio_is_none_when_eps_is_negative():
    details = {"name": "Amazon.Com Inc"}
    bars = _bars([100.0] * 25, [105.0] * 25, [95.0] * 25)
    with (
        patch("app.services.company_profile.fetch_ticker_details", new=AsyncMock(return_value=details)),
        patch("app.services.company_profile.fetch_daily_bars", new=AsyncMock(return_value=bars)),
        patch(
            "app.services.company_profile.get_financials",
            new=AsyncMock(return_value=_fake_financials(eps=-2.0)),
        ),
    ):
        result = asyncio.run(get_company_profile("AMZN"))

    assert result.pe_ratio is None  # 亏损公司的市盈率没有意义，不是负数


def test_pe_ratio_is_none_when_eps_unit_is_not_usd():
    """真实复现过的bug（NBIS）：股价（latest_close）永远是Polygon给的美元，
    但境外发行人的稀释EPS可能只有本币计价的数据（比如"RUB/shares"）——直接
    相除会算出一个跨货币、完全没有意义的市盈率（比如259.2美元/53.26卢布=4.87，
    看起来正常实际毫无意义）。这里不该硬算，应该如实返回None并在note里说明。"""
    details = {"name": "Nebius Group N.V."}
    bars = _bars([259.2] * 25, [265.0] * 25, [255.0] * 25)
    with (
        patch("app.services.company_profile.fetch_ticker_details", new=AsyncMock(return_value=details)),
        patch("app.services.company_profile.fetch_daily_bars", new=AsyncMock(return_value=bars)),
        patch(
            "app.services.company_profile.get_financials",
            new=AsyncMock(return_value=_fake_financials(eps=53.26, unit="RUB/shares")),
        ),
    ):
        result = asyncio.run(get_company_profile("NBIS"))

    assert result.pe_ratio is None
    assert result.note is not None
    assert "RUB/shares" in result.note


def test_pe_ratio_is_none_when_financials_fetch_fails():
    """算不出市盈率不应该拖垮整个公司概览请求，是软性降级不是硬错误。"""
    details = {"name": "Amazon.Com Inc"}
    bars = _bars([100.0] * 25, [105.0] * 25, [95.0] * 25)
    with (
        patch("app.services.company_profile.fetch_ticker_details", new=AsyncMock(return_value=details)),
        patch("app.services.company_profile.fetch_daily_bars", new=AsyncMock(return_value=bars)),
        patch(
            "app.services.company_profile.get_financials",
            new=AsyncMock(side_effect=SecClientError("SEC挂了")),
        ),
    ):
        result = asyncio.run(get_company_profile("AMZN"))

    assert result.has_data is True
    assert result.pe_ratio is None
