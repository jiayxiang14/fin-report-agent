"""cache_lock.get_lock 本身的基础行为，加上一个真实的并发场景验证：Best-of-N
并行跑多个候选时，同一个ticker的并发请求应该被去重（只有一个真正发网络请求，
另一个等锁之后直接读到新鲜缓存），而不是都各自发一次请求。
"""

import asyncio
from unittest.mock import patch

import httpx

from app.services.cache_lock import get_lock
from app.services.polygon_client import fetch_daily_bars


def test_get_lock_returns_the_same_lock_instance_for_the_same_key():
    assert get_lock("AAPL") is get_lock("AAPL")


def test_get_lock_returns_different_lock_instances_for_different_keys():
    assert get_lock("AAPL") is not get_lock("MSFT")


def test_concurrent_fetch_daily_bars_for_same_ticker_only_hits_network_once(tmp_path, monkeypatch):
    """模拟Best-of-N并行跑多个候选、都要同一个ticker数据的场景：缓存冷启动时，
    两个几乎同时发起的请求不该各自都真的打一次Polygon——第一个进锁的去发请求
    +写缓存，第二个等锁之后应该直接命中新鲜缓存，不再发第二次请求。
    """
    monkeypatch.setattr("app.services.polygon_client.CACHE_DIR", tmp_path)
    call_count = 0

    async def fake_throttled_get(client, url, params):
        nonlocal call_count
        call_count += 1
        # 故意让"发请求"这一步耗时，扩大竞态窗口——如果锁没生效，两个协程会
        # 都在这个await点之前判定"缓存不存在"，然后都真的走到这里发请求
        await asyncio.sleep(0.05)
        payload = {"results": [{"t": 1785038400000, "c": 100.0, "h": 105.0, "l": 98.0, "v": 1_000_000}]}
        return httpx.Response(200, request=httpx.Request("GET", url), json=payload)

    async def run_concurrently():
        with patch("app.services.polygon_client.throttled_get", new=fake_throttled_get):
            await asyncio.gather(
                fetch_daily_bars("CONCURRENTTEST", client=None),
                fetch_daily_bars("CONCURRENTTEST", client=None),
            )

    asyncio.run(run_concurrently())

    assert call_count == 1
