"""Alpha Vantage 请求共享的限速器和磁盘缓存。

跟 `polygon_client.py` 是同一个模式（共享限速状态、按 ticker 缓存到磁盘），但参数
差很多：Alpha Vantage 免费层限的是**每天25次请求配额**（不是Polygon那种按分钟算的
速率），所以限速器也不能照搬"均匀摊开"的实现，见下面 `throttled_get` 的注释。
缓存 TTL 因此要设得远比 Polygon 的20小时长——EPS 分析师预期这类数据一个季度才更新一次，
没必要每天重新拉，拉一次能扛一整个财报周期才划算。
"""

import asyncio
import json
from datetime import datetime, timedelta
from pathlib import Path

import httpx

from app.core.config import settings
from app.services.cache_lock import get_lock

CACHE_DIR = Path(__file__).resolve().parent.parent.parent / ".cache"
EARNINGS_CACHE_TTL = timedelta(days=7)

# 每天25次请求是配额上限，不是"每秒/每分钟"式的速率——不能照搬polygon_client.py
# 那种"均匀摊开"限速器（86400/25秒≈57.6分钟一次的强制间隔会让第二次调用在一次
# Agent Loop内部硬生生卡住将近一小时，且工具调用链路上没有任何超时能打断这个
# sleep）。这里改成滚动24小时窗口配额检查：额度内立即放行，超额直接报错，让
# Agent/用户马上看到明确失败，而不是无声地卡住
_DAILY_QUOTA = 25
_QUOTA_WINDOW = timedelta(days=1)
_rate_limit_lock = asyncio.Lock()
_request_timestamps: list[datetime] = []


class AlphaVantageClientError(Exception):
    pass


def require_api_key() -> str:
    if not settings.alpha_vantage_api_key:
        raise AlphaVantageClientError("缺少 ALPHA_VANTAGE_API_KEY 环境变量，请在 .env 里配置")
    return settings.alpha_vantage_api_key


async def throttled_get(client: httpx.AsyncClient, url: str, params: dict) -> httpx.Response:
    async with _rate_limit_lock:
        now = datetime.now()
        cutoff = now - _QUOTA_WINDOW
        while _request_timestamps and _request_timestamps[0] < cutoff:
            _request_timestamps.pop(0)
        if len(_request_timestamps) >= _DAILY_QUOTA:
            retry_at = _request_timestamps[0] + _QUOTA_WINDOW
            raise AlphaVantageClientError(
                f"已达到 Alpha Vantage 每日{_DAILY_QUOTA}次请求配额，请在 {retry_at:%H:%M} 后重试"
            )
        response = await client.get(url, params=params)
        _request_timestamps.append(now)
    return response


def _cache_file(function: str, ticker: str) -> Path:
    return CACHE_DIR / f"alphavantage_{function}_{ticker}.json"


async def fetch_json(function: str, ticker: str, client: httpx.AsyncClient) -> dict:
    """按 (function, ticker) 缓存 Alpha Vantage 的原始 JSON 响应。任何失败（网络错误、
    非2xx状态码、返回体里带 "Note"/"Error Message" 这类限速或参数错误提示）统一抛
    AlphaVantageClientError，调用方自己决定要包装成什么领域异常。

    整个函数体在 get_lock(function+ticker专属key) 保护下执行——每天只有25次配额，
    并发场景（比如Best-of-N并行跑多个候选）如果不加锁，会出现多个协程同时判定
    "缓存不存在"、一起发请求，凭空多花好几次本来就紧张的配额去查同一份数据。
    """
    cache_file = _cache_file(function, ticker)
    async with get_lock(f"alphavantage_{function}_{ticker}"):
        if cache_file.exists():
            age = datetime.now() - datetime.fromtimestamp(cache_file.stat().st_mtime)
            if age < EARNINGS_CACHE_TTL:
                return json.loads(cache_file.read_text())

        api_key = require_api_key()
        params = {"function": function, "symbol": ticker, "apikey": api_key}

        try:
            response = await throttled_get(client, "https://www.alphavantage.co/query", params)
        except httpx.HTTPError as exc:
            raise AlphaVantageClientError(f"请求 Alpha Vantage 失败：{exc}") from exc
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise AlphaVantageClientError(f"Alpha Vantage 返回错误状态码 {response.status_code}") from exc

        payload = response.json()
        # Alpha Vantage 对限速/无效key/未知symbol 不用 HTTP 错误状态码表达，而是在
        # 200响应体里塞一句提示文字，键名各不相同，只能挨个认
        for error_key in ("Note", "Error Message", "Information"):
            if error_key in payload:
                raise AlphaVantageClientError(f"Alpha Vantage 返回提示：{payload[error_key]}")

        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(json.dumps(payload))
        return payload
