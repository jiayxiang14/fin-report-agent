import asyncio
from unittest.mock import AsyncMock, patch

from app.models.financials import FinancialMetric, FinancialsResponse, MetricPoint
from app.services.verify_number import verify_number


def _fake_financials() -> FinancialsResponse:
    annual = MetricPoint(end="2025-09-27", val=416_161_000_000.0, form="10-K", filed="2025-10-31")
    quarterly = MetricPoint(end="2026-06-27", val=109_417_000_000.0, form="10-Q", filed="2026-07-31")
    return FinancialsResponse(
        ticker="AAPL",
        cik="0000320193",
        entity_name="Apple Inc.",
        metrics={
            "revenue": FinancialMetric(
                tag="RevenueFromContractWithCustomerExcludingAssessedTax",
                label="营业收入",
                unit="USD",
                latest_annual=annual,
                latest_quarterly=quarterly,
            )
        },
        retrieved_at="2026-08-03T00:00:00Z",
    )


def _run(*args, **kwargs):
    with patch(
        "app.services.verify_number.get_financials", new=AsyncMock(return_value=_fake_financials())
    ):
        return asyncio.run(verify_number(*args, **kwargs))


def test_matching_claim_within_tolerance():
    result = _run("AAPL", "revenue", 109_400_000_000.0, "quarterly")
    assert result.matches is True
    assert result.actual_value == 109_417_000_000.0


def test_wildly_wrong_claim_does_not_match():
    result = _run("AAPL", "revenue", 364_357_000_000.0, "quarterly")
    assert result.matches is False
    assert result.difference_pct > 1.0


def test_unknown_metric_returns_none_not_false():
    result = _run("AAPL", "made_up_metric", 100.0, "annual")
    assert result.matches is None
    assert "不支持的指标" in result.note


def test_unsupported_period_returns_none():
    result = _run("AAPL", "revenue", 100.0, "monthly")
    assert result.matches is None
    assert "不支持的周期" in result.note


def test_metric_without_data_for_period_returns_none():
    # revenue 只 mock 了 annual/quarterly，换一个真实存在但没数据的场景：
    # 用一个 financials 里压根没有的 metric key 之外的情况——这里用 net_income
    # 因为 fake_financials 没有提供它，应该返回 matches=None
    result = _run("AAPL", "net_income", 100.0, "annual")
    assert result.matches is None
    assert "没有可用的" in result.note
