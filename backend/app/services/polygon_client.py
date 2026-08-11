"""所有 Polygon.io 请求共享的限速器和日线数据拉取。

限速状态必须共享：`sector_rotation.py`（板块ETF日线）、`price_reaction.py`（个股日线+
成交量）、`company_profile.py`（个股日线算ADR + ticker详情）都打Polygon的同一个免费
tier，不能各起一份限速状态，不然多个模块分别按"每分钟5次"算，实际请求速率会变成成倍——
跟 `sec_client.py` 给SEC相关模块共享限速状态是同一个道理。

日线拉取也收敛成一份共享实现：三个消费者原本各自需要的字段不同（收盘价 / 收盘价+成交量 /
最高最低价算ADR），与其各写一份几乎相同的拉取+缓存+错误处理逻辑，不如共享同一次拉取——
副作用是同一次分析里如果多个功能都要用到同一个ticker的日线数据，会命中同一份缓存，不会
各打一次Polygon。这里只负责拉取和统一的错误包装（`PolygonClientError`），各调用方在自己
的边界上把它包装成各自的领域异常，跟 `require_api_key()` 现在的做法一致。
"""

import asyncio
import json
import time
from datetime import date, datetime, timedelta
from pathlib import Path

import httpx
import pandas as pd

from app.core.config import settings
from app.services.cache_lock import get_lock

CACHE_DIR = Path(__file__).resolve().parent.parent.parent / ".cache"
# Polygon免费版是EOD数据，一天刷新一次足够——但20小时比一个自然日短，只要两次
# 使用间隔略微超过20小时（哪怕只差一两小时），缓存就会判定过期，导致几乎每天
# 第一次使用都触发"板块轮动12个ETF+主题轮动27个ticker，共35+个标的一起过期，
# 排队等同一个限速器逐个重新拉取"的冷启动，实测能拖到8-10分钟。调到30小时，
# 只要两次使用间隔不超过30小时就不会触发这种全量冷启动，不影响数据新鲜度
# （同一天内数据本来就不会变）
DAILY_BARS_CACHE_TTL = timedelta(hours=30)
# 取三个消费者里最大的需求：RRG的130天滚动窗口+历史展示需要约450个日历天
LOOKBACK_DAYS = 450

_MIN_REQUEST_INTERVAL = 60.0 / 5
_rate_limit_lock = asyncio.Lock()
_last_request_at = 0.0


class PolygonClientError(Exception):
    pass


def require_api_key() -> str:
    if not settings.polygon_api_key:
        raise PolygonClientError("缺少 POLYGON_API_KEY 环境变量，请在 .env 里配置")
    return settings.polygon_api_key


async def throttled_get(client: httpx.AsyncClient, url: str, params: dict) -> httpx.Response:
    global _last_request_at
    async with _rate_limit_lock:
        elapsed = time.monotonic() - _last_request_at
        wait = _MIN_REQUEST_INTERVAL - elapsed
        if wait > 0:
            await asyncio.sleep(wait)
        response = await client.get(url, params=params)
        _last_request_at = time.monotonic()
    return response


def _bars_cache_file(ticker: str) -> Path:
    return CACHE_DIR / f"polygon_bars_{ticker}.json"


async def fetch_daily_bars(ticker: str, client: httpx.AsyncClient) -> pd.DataFrame:
    """拉取某个 ticker 近 LOOKBACK_DAYS 天的日线数据（收盘/最高/最低/成交量），
    磁盘缓存30小时。任何失败（网络错误、非404状态码、404、空结果）统一抛
    PolygonClientError，调用方自己决定要包装成什么领域异常。

    整个函数体在 get_lock(ticker专属key) 保护下执行：同一个ticker的并发请求
    （比如Best-of-N并行跑多个候选、或者前端同时加载好几个用到同一ticker行情
    的面板）会排队而不是一起去发请求——不只是防止并发写同一个缓存文件损坏
    数据，第一个请求写完缓存后，排在后面的请求拿到锁时会直接命中新鲜缓存，
    不会重复消耗本来就紧张的限速配额。
    """
    cache_file = _bars_cache_file(ticker)
    async with get_lock(f"polygon_bars_{ticker}"):
        if cache_file.exists():
            age = datetime.now() - datetime.fromtimestamp(cache_file.stat().st_mtime)
            if age < DAILY_BARS_CACHE_TTL:
                cached = json.loads(cache_file.read_text())
                return pd.DataFrame(
                    {
                        "close": cached["close"],
                        "high": cached["high"],
                        "low": cached["low"],
                        "volume": cached["volume"],
                    },
                    index=pd.to_datetime(cached["date"]),
                )

        end = date.today()
        start = end - timedelta(days=LOOKBACK_DAYS)
        url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/day/{start}/{end}"
        api_key = require_api_key()
        params = {"adjusted": "true", "sort": "asc", "limit": 50000, "apiKey": api_key}

        try:
            response = await throttled_get(client, url, params)
        except httpx.HTTPError as exc:
            raise PolygonClientError(f"请求 Polygon 行情数据失败：{exc}") from exc
        if response.status_code == 404:
            raise PolygonClientError(f"Polygon 没有 {ticker} 的行情数据")
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise PolygonClientError(f"Polygon 返回错误状态码 {response.status_code}") from exc
        payload = response.json()
        results = payload.get("results") or []
        if not results:
            raise PolygonClientError(f"Polygon 返回了 {ticker} 的空行情数据")

        dates = [datetime.fromtimestamp(r["t"] / 1000).date().isoformat() for r in results]
        closes = [r["c"] for r in results]
        highs = [r["h"] for r in results]
        lows = [r["l"] for r in results]
        volumes = [r["v"] for r in results]

        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(
            json.dumps({"date": dates, "close": closes, "high": highs, "low": lows, "volume": volumes})
        )

        return pd.DataFrame(
            {"close": closes, "high": highs, "low": lows, "volume": volumes},
            index=pd.to_datetime(dates),
        )
