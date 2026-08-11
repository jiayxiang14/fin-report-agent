"""所有 sec.gov 域名请求共享的 User-Agent 校验和限速器。

SEC 的"每秒不超过10次请求"限制是按调用方（也就是我们这一个后端进程）算的，
不是按 data.sec.gov / www.sec.gov 分开算的，所以 sec_edgar.py 和 filing_text.py
必须共用同一个节流状态，不能各起一份、各算各的。
"""

import asyncio
import time

import httpx

from app.core.config import settings

CACHE_DIR_NAME = ".cache"

_MIN_REQUEST_INTERVAL = 1.0 / 10
_rate_limit_lock = asyncio.Lock()
_last_request_at = 0.0


class SecClientError(Exception):
    pass


def require_user_agent() -> dict[str, str]:
    if not settings.sec_edgar_user_agent:
        raise SecClientError(
            "缺少 SEC_EDGAR_USER_AGENT 环境变量，SEC EDGAR 要求所有请求带上可识别的 "
            "User-Agent（格式：'你的名字 你的邮箱'），请在 .env 里配置。"
        )
    return {"User-Agent": settings.sec_edgar_user_agent}


async def throttled_get(
    client: httpx.AsyncClient, url: str, params: dict | None = None
) -> httpx.Response:
    """在共享的最小请求间隔约束下发起 GET，保证不超过 SEC 的每秒10次限速。"""
    global _last_request_at
    async with _rate_limit_lock:
        elapsed = time.monotonic() - _last_request_at
        wait = _MIN_REQUEST_INTERVAL - elapsed
        if wait > 0:
            await asyncio.sleep(wait)
        response = await client.get(url, params=params, headers=require_user_agent())
        _last_request_at = time.monotonic()
    return response
