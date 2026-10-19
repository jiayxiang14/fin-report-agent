"""后台定期预热行情缓存。

板块轮动（`sector_rotation.py`）和细分主题轮动（`thematic_flow.py`）用到的
ETF/成分股是固定集合，不依赖用户这次分析的是哪家公司——可以在用户真正发起
分析之前就提前拉热磁盘缓存，避免"缓存超过30小时TTL全部过期后，第一个真实
请求撞上Polygon限速器（5次/分钟）排队逐个刷新，拖到几分钟"这种冷启动体验
（真实复现过：连续一周多没人跑分析，12个板块ETF缓存全过期，单是板块轮动
一项就要排队2分钟以上）。

只预热"固定、不依赖用户输入"的这部分标的（板块ETF+基准+主题篮子成分股）——
公司自己的财务数据/同行对比这些是用户驱动的、没法提前预热，不在这个模块
范围内。
"""

from __future__ import annotations

import asyncio
import logging
from datetime import timedelta

import httpx

from app.services.polygon_client import fetch_daily_bars
from app.services.sector_rotation import BENCHMARK, SECTOR_ETFS
from app.services.thematic_flow import THEMATIC_BASKETS

logger = logging.getLogger(__name__)

# 远小于 polygon_client.DAILY_BARS_CACHE_TTL（30小时），保证只要进程一直
# 跑着，缓存实际上永远不会真的过期到需要冷启动排队的地步
WARM_INTERVAL = timedelta(hours=6)


def _warm_ticker_set() -> set[str]:
    """两处调用点（板块轮动、主题轮动）各自的固定标的去重合并成一个集合——
    用set而不是两份list相加，哪天两边篮子出现重叠ticker（比如某个板块ETF
    同时也是某个主题篮子成分股）不会被重复预热两次。"""
    tickers = {BENCHMARK, *SECTOR_ETFS.keys()}
    for basket in THEMATIC_BASKETS.values():
        tickers.update(basket)
    return tickers


async def warm_daily_bars_cache() -> None:
    """挨个预热。单个ticker失败（网络抖动、这只票临时下市）不该打断其它
    标的的预热——记日志，不往外抛，这是锦上添花的后台任务，不是关键路径，
    调用方（`run_periodic_cache_warming`）也不应该因为一次预热部分失败就
    整个循环退出。"""
    async with httpx.AsyncClient(timeout=15.0) as client:
        for ticker in sorted(_warm_ticker_set()):
            try:
                await fetch_daily_bars(ticker, client)
            except Exception:  # noqa: BLE001 - 预热任务的失败边界，一个ticker出错不能连累其它
                logger.warning("缓存预热失败：%s", ticker, exc_info=True)


async def run_periodic_cache_warming() -> None:
    """常驻后台协程，`main.py`的lifespan里用`asyncio.create_task`启动、
    应用关闭时cancel掉。启动后立刻做一轮预热（不是先睡满一个WARM_INTERVAL
    才开始），这样进程刚重启/刚部署时也能尽快把冷缓存捂热，不用等6小时。"""
    while True:
        await warm_daily_bars_cache()
        await asyncio.sleep(WARM_INTERVAL.total_seconds())
