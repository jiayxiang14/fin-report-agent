from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

from app.main import app
from app.models.peer import PeerComparisonResponse

client = TestClient(app)


def test_peer_comparison_route_returns_data():
    fake = PeerComparisonResponse(ticker="AMZN", sector_etf="XLY", sector_name="Consumer Discretionary", peers=[])
    with patch("app.api.routes.peer.get_peer_comparison", new=AsyncMock(return_value=fake)):
        response = client.get("/api/peer-comparison/AMZN")

    assert response.status_code == 200
    assert response.json()["sector_etf"] == "XLY"
