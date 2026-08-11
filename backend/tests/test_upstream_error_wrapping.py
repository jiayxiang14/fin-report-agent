"""复盘发现的问题：SEC EDGAR / Polygon 返回非404错误（500/503/限速等）时，服务层
原本对 `response.raise_for_status()` 不做任何包装，未捕获的 `httpx.HTTPStatusError`
会直接穿透到路由层——而路由层只 catch 了各自的自定义异常类型，接不住 httpx 原生
异常，最终变成裸的 500 Internal Server Error，而不是设计好的干净 502 降级。

这里直接在服务层验证：非404的上游错误现在会被统一包装成对应的领域异常
（SecEdgarError / FilingTextError / SectorDataError），路由层原有的
`except SecEdgarError` 等分支就能接住，不需要改路由代码。
"""

import asyncio
from unittest.mock import patch

import httpx
import pytest

from app.services.filing_text import FilingTextError, fetch_filing_document, fetch_submissions
from app.services.polygon_client import PolygonClientError
from app.services.sec_edgar import SecEdgarError, fetch_company_facts
from app.services.sector_rotation import SectorDataError, _fetch_closes

FAKE_CIK = "9999999999"  # 确保命中不了任何真实的本地缓存文件
FAKE_TICKER = "ZZZZTEST"


def _error_response(url: str, status_code: int = 503) -> httpx.Response:
    return httpx.Response(status_code, request=httpx.Request("GET", url), text="upstream error")


def test_fetch_company_facts_wraps_non_404_status_as_sec_edgar_error():
    async def fake_throttled_get(client, url, params=None):
        return _error_response(url)

    with patch("app.services.sec_edgar.throttled_get", new=fake_throttled_get):
        with pytest.raises(SecEdgarError):
            asyncio.run(fetch_company_facts(FAKE_CIK, client=None))


def test_fetch_company_facts_wraps_connection_error_as_sec_edgar_error():
    async def fake_throttled_get(client, url, params=None):
        raise httpx.ConnectError("network is unreachable")

    with patch("app.services.sec_edgar.throttled_get", new=fake_throttled_get):
        with pytest.raises(SecEdgarError):
            asyncio.run(fetch_company_facts(FAKE_CIK, client=None))


def test_fetch_submissions_wraps_non_404_status_as_filing_text_error():
    async def fake_throttled_get(client, url, params=None):
        return _error_response(url)

    with patch("app.services.filing_text.throttled_get", new=fake_throttled_get):
        with pytest.raises(FilingTextError):
            asyncio.run(fetch_submissions(FAKE_CIK, client=None))


def test_fetch_filing_document_wraps_non_404_status_as_filing_text_error():
    async def fake_throttled_get(client, url, params=None):
        return _error_response(url)

    with patch("app.services.filing_text.throttled_get", new=fake_throttled_get):
        with pytest.raises(FilingTextError):
            asyncio.run(
                fetch_filing_document(FAKE_CIK, "0000000000-26-000000", "fake.htm", client=None)
            )


def test_sector_rotation_wraps_polygon_client_error_as_sector_data_error():
    """`sector_rotation.py`现在调用共享的`polygon_client.fetch_daily_bars`（在
    test_polygon_client.py里单独测过它自己的错误包装），这里只验证`_fetch_closes`
    在自己的边界上把 PolygonClientError 转成 SectorDataError，不需要重新模拟一次
    完整的HTTP错误响应。"""

    async def fake_fetch_daily_bars(ticker, client):
        raise PolygonClientError("上游错误")

    with patch("app.services.sector_rotation.fetch_daily_bars", new=fake_fetch_daily_bars):
        with pytest.raises(SectorDataError):
            asyncio.run(_fetch_closes(FAKE_TICKER, client=None))
