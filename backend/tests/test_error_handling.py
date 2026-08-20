"""全局异常处理：SecClientError基类（比如SEC_EDGAR_USER_AGENT没配置）不会被
各路由里`except SecEdgarError`（子类）挡住，靠main.py注册的全局handler兜底
转成502——真实CI里跑到过这个缺口（缺SEC_EDGAR_USER_AGENT时裸露500堆栈），
这里锁定修复后的行为。
"""

from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

from app.main import app
from app.services.sec_client import SecClientError

client = TestClient(app)


def test_sec_client_error_base_class_is_caught_by_global_handler_not_just_the_subclass():
    with patch(
        "app.api.routes.financials.get_financials",
        new=AsyncMock(side_effect=SecClientError("缺少 SEC_EDGAR_USER_AGENT 环境变量")),
    ):
        response = client.get("/api/financials/AAPL")

    assert response.status_code == 502
    assert response.json()["detail"] == "缺少 SEC_EDGAR_USER_AGENT 环境变量"
