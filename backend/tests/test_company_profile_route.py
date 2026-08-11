from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

from app.main import app
from app.models.company_profile import CompanyProfileResponse
from app.services.company_profile import CompanyProfileError

client = TestClient(app)


def test_company_profile_route_returns_data():
    fake = CompanyProfileResponse(ticker="AMZN", has_data=True, name="Amazon.Com Inc")
    with patch(
        "app.api.routes.company_profile.get_company_profile", new=AsyncMock(return_value=fake)
    ):
        response = client.get("/api/company-profile/AMZN")

    assert response.status_code == 200
    assert response.json()["name"] == "Amazon.Com Inc"


def test_company_profile_route_maps_service_error_to_502():
    with patch(
        "app.api.routes.company_profile.get_company_profile",
        new=AsyncMock(side_effect=CompanyProfileError("Polygon 挂了")),
    ):
        response = client.get("/api/company-profile/AMZN")

    assert response.status_code == 502
