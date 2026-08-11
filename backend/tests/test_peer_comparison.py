import asyncio
from unittest.mock import AsyncMock, patch

from app.models.financials import FinancialMetric, FinancialsResponse, MetricPoint
from app.services.peer_comparison import find_peers, get_peer_comparison


def test_find_peers_returns_same_sector_tickers_excluding_self():
    peers = find_peers("AAPL")
    assert "AAPL" not in peers
    assert len(peers) == 3  # XLK 板块预设池里有7家以上大盘股，够取满3个
    assert peers == sorted(peers)  # 按字母序取前N个


def test_find_peers_returns_empty_for_unknown_ticker():
    assert find_peers("NOTATICKER") == []


def _fake_financials(ticker: str, entity_name: str, revenue: float, net_income: float) -> FinancialsResponse:
    point = MetricPoint(end="2025-09-27", val=revenue, form="10-K", filed="2025-10-31", yoy_change_pct=5.0)
    ni_point = MetricPoint(
        end="2025-09-27", val=net_income, form="10-K", filed="2025-10-31", yoy_change_pct=10.0
    )
    return FinancialsResponse(
        ticker=ticker,
        cik="0000000000",
        entity_name=entity_name,
        metrics={
            "revenue": FinancialMetric(tag="Revenues", label="营业收入", unit="USD", latest_annual=point),
            "net_income": FinancialMetric(
                tag="NetIncomeLoss", label="净利润", unit="USD", latest_annual=ni_point
            ),
        },
        retrieved_at="2026-08-03T00:00:00Z",
    )


def test_get_peer_comparison_aggregates_peer_financials():
    fake_responses = {
        "MSFT": _fake_financials("MSFT", "MICROSOFT CORP", 300_000_000_000, 100_000_000_000),
        "NVDA": _fake_financials("NVDA", "NVIDIA CORP", 100_000_000_000, 40_000_000_000),
    }

    async def fake_get_financials(ticker):
        if ticker not in fake_responses:
            raise AssertionError(f"unexpected peer lookup: {ticker}")
        return fake_responses[ticker]

    with patch(
        "app.services.peer_comparison.get_financials", new=AsyncMock(side_effect=fake_get_financials)
    ), patch("app.services.peer_comparison.find_peers", return_value=["MSFT", "NVDA"]):
        result = asyncio.run(get_peer_comparison("AAPL"))

    assert result.sector_etf == "XLK"
    assert result.note is None
    assert {p.ticker for p in result.peers} == {"MSFT", "NVDA"}
    msft = next(p for p in result.peers if p.ticker == "MSFT")
    assert msft.revenue == 300_000_000_000
    assert msft.net_income_yoy_pct == 10.0


def test_get_peer_comparison_unknown_ticker_returns_note_not_error():
    result = asyncio.run(get_peer_comparison("RARE123"))
    assert result.peers == []
    assert result.sector_etf is None
    assert "不在预设股票池内" in result.note
