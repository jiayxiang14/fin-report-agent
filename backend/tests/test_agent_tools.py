import asyncio
from unittest.mock import AsyncMock, patch

from app.services.agent.tools import TOOL_SCHEMAS, execute_tool
from app.services.sec_edgar import TickerNotFoundError


def test_all_eight_tools_are_registered():
    names = {t["name"] for t in TOOL_SCHEMAS}
    assert names == {
        "get_financials",
        "get_sector_position",
        "get_peer_comparison",
        "get_filing_text",
        "get_price_reaction",
        "get_thematic_flow",
        "get_earnings_surprise",
        "verify_number",
    }


def test_unknown_tool_name_returns_error_not_exception():
    content, is_error = asyncio.run(execute_tool("does_not_exist", {}))
    assert is_error is True
    assert "未知工具" in content


def test_tool_failure_is_caught_and_reported_as_error():
    with patch(
        "app.services.agent.tools.get_financials",
        new=AsyncMock(side_effect=TickerNotFoundError("找不到")),
    ):
        content, is_error = asyncio.run(execute_tool("get_financials", {"ticker": "NOPE"}))
    assert is_error is True
    assert "工具调用失败" in content


def test_get_price_reaction_dispatch_returns_json_content():
    from app.models.price_reaction import PriceReactionResponse

    fake = PriceReactionResponse(ticker="AMZN", has_data=True, price_change_pct=15.2)
    with patch("app.services.agent.tools.get_price_reaction", new=AsyncMock(return_value=fake)):
        content, is_error = asyncio.run(execute_tool("get_price_reaction", {"ticker": "AMZN"}))
    assert is_error is False
    assert '"price_change_pct":15.2' in content


def test_get_thematic_flow_dispatch_passes_ticker_through():
    from app.models.thematic_flow import ThematicFlowResponse

    fake = ThematicFlowResponse(
        ticker="NVDA",
        matched_themes=["AI芯片/GPU"],
        sic_matched_themes=[],
        benchmark="SPY",
        as_of="2026-08-05",
        themes=[],
        note="test",
    )
    with patch(
        "app.services.agent.tools.get_thematic_flow", new=AsyncMock(return_value=fake)
    ) as mock_get_thematic_flow:
        content, is_error = asyncio.run(execute_tool("get_thematic_flow", {"ticker": "NVDA"}))
    assert is_error is False
    assert '"matched_themes":["AI芯片/GPU"]' in content
    mock_get_thematic_flow.assert_awaited_once_with("NVDA")


def test_get_earnings_surprise_dispatch_returns_json_content():
    from app.models.earnings_surprise import EarningsSurpriseResponse

    fake = EarningsSurpriseResponse(ticker="AAPL", has_data=True, verdict="超预期")
    with patch(
        "app.services.agent.tools.get_earnings_surprise", new=AsyncMock(return_value=fake)
    ) as mock_get_earnings_surprise:
        content, is_error = asyncio.run(execute_tool("get_earnings_surprise", {"ticker": "AAPL"}))
    assert is_error is False
    assert '"verdict":"超预期"' in content
    mock_get_earnings_surprise.assert_awaited_once_with("AAPL")


def test_get_financials_dispatch_returns_json_content():
    from app.models.financials import FinancialsResponse

    fake = FinancialsResponse(
        ticker="AAPL", cik="0000320193", entity_name="Apple Inc.", metrics={}, retrieved_at="2026-08-03T00:00:00Z"
    )
    with patch("app.services.agent.tools.get_financials", new=AsyncMock(return_value=fake)):
        content, is_error = asyncio.run(execute_tool("get_financials", {"ticker": "AAPL"}))
    assert is_error is False
    assert '"entity_name":"Apple Inc."' in content
