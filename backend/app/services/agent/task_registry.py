"""进程内任务注册表：把"发起一次分析"和"订阅这次分析的事件流"两件事解耦。

之前 `/stream` 端点一个HTTP请求同时做这两件事——SSE连接一断（用户刷新页面、
网络抖动），`event_generator` 的 `finally` 里直接 `task.cancel()` 把还在跑的
Agent Loop 杀掉，已经花掉的LLM调用全部白费，重连等于重新发起一次真实花钱的
分析，不是接回原来那次。

现在：`start_task` 创建一个不属于任何HTTP连接的后台 `asyncio.Task`，返回
`task_id`；`stream_task` 订阅这个task——连接上先重放已经发生过的全部事件
（缓冲区），再继续实时推送新事件；断开时只把这条连接自己的订阅摘掉，不影响
后台任务本身。刷新页面、网络抖动重连，只要 `task_id` 还在（没过期），接回
的是同一次分析，不会重复花钱。

只在进程内存里，不做跨进程/跨重启持久化——真要扛多实例部署或进程重启，
需要 Redis/真正的任务队列（后台执行本身还是绑在某一个进程的事件循环里，
Redis 只能保证已经产生的事件/结果不因为进程重启而丢，救不回"跑到一半"
这件事本身），是完全不同量级的工程量，讨论过，这里明确不做。

过期策略按"交易日"而不是滚动24小时：`_next_trading_day_start` 只跳过周末，
不处理NYSE法定假日（一年约9天）——这是刻意的简化，假期当天会多保留一天
缓存，代价很小，不值得为这几天/年接一份完整的交易日历。
"""

from __future__ import annotations

import asyncio
import json
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta

from fastapi.responses import StreamingResponse

from app.services.agent import trace_log

OnEvent = Callable[[dict], Awaitable[None]]
# 调用方包好的闭包：拿到 on_event 后跑真正的业务逻辑（run_agent_loop /
# run_best_of_n），返回值必须已经是可JSON序列化的dict（调用方自己
# .model_dump()），task_registry 不关心跑的是哪一种分析
TaskRunner = Callable[[OnEvent], Awaitable[dict]]


def _next_trading_day_start(dt: datetime) -> datetime:
    next_day = dt.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)
    while next_day.weekday() >= 5:  # 5=Saturday, 6=Sunday
        next_day += timedelta(days=1)
    return next_day


@dataclass
class _TaskState:
    ticker: str
    events: list[dict] = field(default_factory=list)
    done: bool = False
    # 当前正在监听的连接，可以有0个（后台任务不依赖任何人正在看着）
    subscribers: list[asyncio.Queue] = field(default_factory=list)
    expires_at: datetime = field(init=False)

    def __post_init__(self) -> None:
        self.expires_at = _next_trading_day_start(datetime.now(UTC))


_tasks: dict[str, _TaskState] = {}


def _evict_expired() -> None:
    now = datetime.now(UTC)
    expired_ids = [task_id for task_id, state in _tasks.items() if state.expires_at <= now]
    for task_id in expired_ids:
        del _tasks[task_id]


def start_task(ticker: str, runner: TaskRunner, task_id: str | None = None) -> str:
    """创建后台任务，立刻返回task_id，不等待任务本身完成。

    `task_id`可以由调用方预先生成并传入（`agent.py`的路由需要在调用这个函数
    之前就知道task_id，好把它闭包进一个`on_tool_result`回调里去落盘完整的
    工具输出）——不传时保持原来的行为，内部自己生成一个。
    """
    _evict_expired()
    task_id = task_id or uuid.uuid4().hex
    state = _TaskState(ticker=ticker)
    _tasks[task_id] = state

    async def on_event(event: dict) -> None:
        # 除了进程内内存缓冲区（服务SSE重放），额外落一份到磁盘——这样即使
        # 这个task过期被清理、或者进程重启，事后依然能按task_id查到完整轨迹。
        # 见 trace_log.py 顶部注释：这里复用task_id本身当trace_id，不发明
        # 第二个并行的标识符。
        trace_log.append_event(task_id, event)
        state.events.append(event)
        for queue in state.subscribers:
            await queue.put(event)

    async def run() -> None:
        try:
            result = await runner(on_event)
            await on_event({"type": "done", "result": result})
        except Exception as exc:  # noqa: BLE001 - 后台任务的失败边界，必须转成事件，不能无声消失
            await on_event({"type": "error", "message": str(exc)})
        finally:
            state.done = True
            for queue in state.subscribers:
                await queue.put(None)  # 结束哨兵

    # 故意不保留这个task的引用——它完全独立于发起它的HTTP请求，不依赖任何
    # 调用方拿着引用去管理生命周期，这正是要解决的问题本身
    asyncio.create_task(run())
    return task_id


async def _replay_and_tail(task_id: str) -> AsyncIterator[dict]:
    state = _tasks.get(task_id)
    if state is None:
        # code字段给前端识别用，不能靠匹配中文文案字符串去判断"这个task该
        # 重新发起了"，那样文案一改前端逻辑就悄悄失效
        yield {"type": "error", "code": "task_not_found", "message": "task_id 不存在或已过期，请重新发起分析"}
        return

    queue: asyncio.Queue = asyncio.Queue()
    events_snapshot = list(state.events)
    # 这两行之间没有await，asyncio单线程协作式调度下不会有其它协程插进来，
    # 不会漏事件（新事件在snapshot之后才发生，会进新注册的队列）也不会重复
    # 收到同一个事件（snapshot里已经有的不会再从队列里收第二次）
    state.subscribers.append(queue)
    try:
        for event in events_snapshot:
            yield event
        if state.done:
            # 连接上的时候任务已经跑完了——重放完历史事件就结束，不用等
            # 队列，本来也不会再有东西进来（新订阅者注册太晚，错过了那次
            # finally里发给旧订阅者的结束哨兵）
            return
        while True:
            event = await queue.get()
            if event is None:
                break
            yield event
    finally:
        if queue in state.subscribers:
            state.subscribers.remove(queue)


def _sse(event: dict) -> str:
    return f"data: {json.dumps(event, ensure_ascii=False)}\n\n"


def stream_task(task_id: str) -> StreamingResponse:
    async def event_generator() -> AsyncIterator[str]:
        async for event in _replay_and_tail(task_id):
            yield _sse(event)

    return StreamingResponse(event_generator(), media_type="text/event-stream")
