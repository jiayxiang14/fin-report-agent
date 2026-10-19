"""app.main的lifespan：按settings.enable_cache_warming决定要不要启动后台
缓存预热任务，应用关闭时要cancel掉，不能让它变成孤儿协程。

不用`TestClient(app)`触发真实lifespan——本项目FastAPI/Starlette版本下，
`TestClient(app)`不加`with`不会触发lifespan事件（其它测试文件也都是这么用的，
不是这次才发现），真触发的话会在每次跑测试时发出真实的Polygon.io网络请求，
这里改成直接把`lifespan`当成普通async context manager调用，全程mock掉
`asyncio.create_task`和`cache_warmer.run_periodic_cache_warming`本身。
"""

import asyncio
from unittest.mock import MagicMock, patch

from app.core.config import settings
from app.main import app, lifespan


def test_lifespan_starts_and_cancels_the_cache_warmer_when_enabled(monkeypatch):
    monkeypatch.setattr(settings, "enable_cache_warming", True)
    fake_task = MagicMock()

    with (
        # patch()对async函数默认生成AsyncMock——调用它返回的是一个新的协程
        # 对象（不是`return_value`本身，那个只有真的await之后才拿得到），
        # 所以不去比较create_task收到的具体协程对象是不是某个特定实例，
        # 只确认"warmer被调用过一次、create_task确实拿着一个协程去建了
        # task"这件事本身
        patch("app.main.cache_warmer.run_periodic_cache_warming") as mock_warmer,
        patch("app.main.asyncio.create_task", return_value=fake_task) as mock_create_task,
    ):

        async def run() -> None:
            async with lifespan(app):
                pass

        asyncio.run(run())

    mock_warmer.assert_called_once_with()
    assert mock_create_task.call_count == 1
    passed_coro = mock_create_task.call_args.args[0]
    assert asyncio.iscoroutine(passed_coro)
    passed_coro.close()  # 从没被真的await过（create_task本身被mock掉了），显式关掉避免"coroutine从未被await"的警告
    fake_task.cancel.assert_called_once()


def test_lifespan_does_not_start_the_cache_warmer_when_disabled(monkeypatch):
    monkeypatch.setattr(settings, "enable_cache_warming", False)

    with patch("app.main.asyncio.create_task") as mock_create_task:

        async def run() -> None:
            async with lifespan(app):
                pass

        asyncio.run(run())

    mock_create_task.assert_not_called()
