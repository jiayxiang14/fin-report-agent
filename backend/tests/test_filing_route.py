"""路由层测试：确认 resolve_cik 抛出的 TickerNotFoundError 能被正确映射成 404，
而不是未捕获异常导致的裸 500（第2阶段联调时发现的真实 bug：filing 路由最初只
catch 了 FilingTextError，没盖住 TickerNotFoundError）。"""

from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

from app.main import app
from app.services.sec_edgar import TickerNotFoundError

client = TestClient(app)


def test_unknown_ticker_returns_404_not_500():
    with patch(
        "app.api.routes.filing.get_filing_text",
        new=AsyncMock(side_effect=TickerNotFoundError("找不到 ticker")),
    ):
        response = client.get("/api/filing-text/ZZRARE")

    assert response.status_code == 404
