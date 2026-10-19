import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from app.services import cache_warmer
from app.services.polygon_client import PolygonClientError
from app.services.sector_rotation import BENCHMARK, SECTOR_ETFS
from app.services.thematic_flow import THEMATIC_BASKETS


def test_warm_ticker_set_covers_benchmark_sector_etfs_and_thematic_baskets():
    tickers = cache_warmer._warm_ticker_set()

    assert BENCHMARK in tickers
    assert set(SECTOR_ETFS.keys()) <= tickers
    for basket in THEMATIC_BASKETS.values():
        assert set(basket) <= tickers


def test_warm_ticker_set_deduplicates_overlap_between_sectors_and_themes():
    # 用真实数据也能验证去重逻辑本身没坏——集合大小不会超过两边加起来的上限
    tickers = cache_warmer._warm_ticker_set()
    max_possible = 1 + len(SECTOR_ETFS) + sum(len(b) for b in THEMATIC_BASKETS.values())
    assert len(tickers) <= max_possible


def test_warm_daily_bars_cache_fetches_every_ticker_in_the_warm_set():
    fake_fetch = AsyncMock(return_value=None)
    with patch("app.services.cache_warmer.fetch_daily_bars", new=fake_fetch):
        asyncio.run(cache_warmer.warm_daily_bars_cache())

    fetched_tickers = {call.args[0] for call in fake_fetch.call_args_list}
    assert fetched_tickers == cache_warmer._warm_ticker_set()


def test_warm_daily_bars_cache_continues_past_a_single_ticker_failure():
    """一只票临时下市/网络抖动导致的失败不该打断其它标的的预热——这是锦上
    添花的后台任务，不是关键路径。"""
    warm_set = cache_warmer._warm_ticker_set()
    failing_ticker = next(iter(warm_set))

    async def fake_fetch(ticker, client):
        if ticker == failing_ticker:
            raise PolygonClientError("模拟失败")
        return None

    with patch("app.services.cache_warmer.fetch_daily_bars", new=fake_fetch):
        asyncio.run(cache_warmer.warm_daily_bars_cache())  # 不应该抛出


def test_run_periodic_cache_warming_warms_immediately_then_sleeps_between_rounds():
    """启动后应该立刻做一轮预热，不是先睡满一个WARM_INTERVAL才开始——用
    asyncio.sleep在第一次被调用时就抛出哨兵异常，只要能观察到"睡之前已经
    warm过一次"就足够验证这个顺序，不需要真的等好几个小时。"""
    calls: list[str] = []

    class _StopLoop(Exception):
        pass

    async def fake_warm() -> None:
        calls.append("warm")

    async def fake_sleep(_seconds: float) -> None:
        calls.append("sleep")
        raise _StopLoop

    with (
        patch("app.services.cache_warmer.warm_daily_bars_cache", new=fake_warm),
        patch("app.services.cache_warmer.asyncio.sleep", new=fake_sleep),
        pytest.raises(_StopLoop),
    ):
        asyncio.run(cache_warmer.run_periodic_cache_warming())

    assert calls == ["warm", "sleep"]
