from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

from app.main import app
from app.models.financials import FinancialsHistoryResponse, MetricHistory
from app.services.sec_edgar import TickerNotFoundError

client = TestClient(app)


def test_financials_history_route_returns_data():
    fake = FinancialsHistoryResponse(
        ticker="AAPL",
        cik="0000320193",
        entity_name="Apple Inc.",
        history={"revenue": MetricHistory(label="营业收入", unit="USD", points=[])},
        retrieved_at="2026-08-07T00:00:00Z",
    )
    with patch(
        "app.api.routes.financials_history.get_financials_history", new=AsyncMock(return_value=fake)
    ):
        response = client.get("/api/financials-history/AAPL")

    assert response.status_code == 200
    assert response.json()["history"]["revenue"]["label"] == "营业收入"


def test_financials_history_route_returns_404_for_unknown_ticker():
    with patch(
        "app.api.routes.financials_history.get_financials_history",
        new=AsyncMock(side_effect=TickerNotFoundError("找不到")),
    ):
        response = client.get("/api/financials-history/NOPE")

    assert response.status_code == 404
