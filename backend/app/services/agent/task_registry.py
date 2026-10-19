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

进程重启/手动重试后接回同一个task_id（见checkpoint.py、task_runner.py）：
`start_task`可以传入`initial_events`（从trace日志重建、截断到存档点的历史事件）
和原始`started_at`，订阅者照样先重放这些历史事件再接收新事件；内存里已经没有
的、已经跑完的task，订阅时从trace日志重建事件重放（`rebuild_events_from_trace`）。

取消：`cancel_task`只对用户主动取消生效——进程关闭/开发时--reload也会取消
后台task，那种情况不发终止事件、原样向上抛，检查点保持`running`，下次启动
时自动恢复。

过期策略按"交易日"而不是滚动24小时：`_next_trading_day_start` 只跳过周末，
不处理NYSE法定假日（一年约9天）——这是刻意的简化，假期当天会多保留一天
缓存，代价很小，不值得为这几天/年接一份完整的交易日历。
"""

from __future__ import annotations

import asyncio
import json
import time
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


_TERMINAL_EVENT_TYPES = frozenset({"done", "error"})


@dataclass
class _TaskState:
    ticker: str
    # 墙钟时间（time.time()），不是monotonic——重启/重试接回时要跟存下来的
    # 原始发起时间比较，monotonic的零点每次进程启动都不一样
    started_at: float = field(default_factory=time.time)
    events: list[dict] = field(default_factory=list)
    done: bool = False
    cancel_requested: bool = False
    handle: asyncio.Task | None = None
    # 当前正在监听的连接，可以有0个（后台任务不依赖任何人正在看着）
    subscribers: list[asyncio.Queue] = field(default_factory=list)
    expires_at: datetime = field(init=False)

    def __post_init__(self) -> None:
        self.expires_at = _next_trading_day_start(datetime.fromtimestamp(self.started_at, UTC))


_tasks: dict[str, _TaskState] = {}


def _evict_expired() -> None:
    now = datetime.now(UTC)
    expired_ids = [task_id for task_id, state in _tasks.items() if state.expires_at <= now]
    for task_id in expired_ids:
        del _tasks[task_id]


def start_task(
    ticker: str,
    runner: TaskRunner,
    task_id: str | None = None,
    *,
    initial_events: list[dict] | None = None,
    started_at: float | None = None,
) -> str:
    """创建后台任务，立刻返回task_id，不等待任务本身完成。

    `task_id`可以由调用方预先生成并传入（`agent.py`的路由需要在调用这个函数
    之前就知道task_id，好把它闭包进一个`on_tool_result`回调里去落盘完整的
    工具输出）——不传时保持原来的行为，内部自己生成一个。

    `initial_events`/`started_at`只在接回已有task（重启恢复/手动重试）时传：
    前者是要先填进缓冲区的历史事件（不会重复写trace日志），后者让总用时从
    原始发起时刻算起。
    """
    _evict_expired()
    task_id = task_id or uuid.uuid4().hex
    state = _TaskState(ticker=ticker, started_at=started_at if started_at is not None else time.time())
    state.events.extend(initial_events or [])
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
        # 总用时由后端算、随终止事件下发：前端刷新页面接回一个早已跑完的task时，
        # 只靠前端自己记的发起时间只能算出"从发起到现在过了多久"（含跑完之后的
        # 空闲时间），不是这次分析真正的用时
        def elapsed_ms() -> int:
            return round((time.time() - state.started_at) * 1000)

        try:
            result = await runner(on_event)
            await on_event({"type": "done", "result": result, "elapsed_ms": elapsed_ms()})
        except asyncio.CancelledError:
            if not state.cancel_requested:
                # 进程关闭/--reload：不是用户的决定，不发终止事件，原样往上抛
                raise
            await on_event({"type": "error", "code": "cancelled", "message": "分析已取消", "elapsed_ms": elapsed_ms()})
        except Exception as exc:  # noqa: BLE001 - 后台任务的失败边界，必须转成事件，不能无声消失
            error_event: dict = {"type": "error", "message": str(exc), "elapsed_ms": elapsed_ms()}
            # 可重试的外部故障（见checkpoint.RetryableTaskError），前端据此显示
            # "从断点重试"按钮
            if getattr(exc, "retryable", False):
                error_event["retryable"] = True
            await on_event(error_event)
        finally:
            state.done = True
            for queue in state.subscribers:
                await queue.put(None)  # 结束哨兵

    # 引用只存在注册表自己的状态里（给cancel_task用，也防止task被垃圾回收），
    # 不交给发起它的HTTP请求——它的生命周期完全独立于那次请求，这正是要
    # 解决的问题本身
    state.handle = asyncio.create_task(run())
    return task_id


def event_count(task_id: str) -> int:
    """已经推送过的事件数，存检查点时一起存下来，恢复时据此截断事件历史。"""
    state = _tasks.get(task_id)
    return len(state.events) if state is not None else 0


def is_running(task_id: str) -> bool:
    state = _tasks.get(task_id)
    return state is not None and not state.done


def is_cancel_requested(task_id: str) -> bool:
    state = _tasks.get(task_id)
    return state is not None and state.cancel_requested


def cancel_task(task_id: str) -> bool:
    """用户主动取消。返回False表示没有正在跑的这个task（不存在/已经结束）。"""
    state = _tasks.get(task_id)
    if state is None or state.done or state.handle is None:
        return False
    state.cancel_requested = True
    state.handle.cancel()
    return True


def rebuild_events_from_trace(task_id: str) -> list[dict]:
    """从trace日志还原出订阅者应该看到的事件序列：去掉不是SSE事件的
    `tool_result_full`和落盘时间戳；遇到`resumed`事件时，按它的`truncate_to`
    把前面的事件截断——被打断的那一轮推出去的半截事件（推理文字推到一半、
    工具开始了没结束），恢复后会整轮重做，不截断的话同一轮会出现两遍。"""
    events: list[dict] = []
    for raw in trace_log.read_trace(task_id):
        if raw.get("type") == "tool_result_full":
            continue
        event = {k: v for k, v in raw.items() if k != "logged_at"}
        if event.get("type") == "resumed" and isinstance(event.get("truncate_to"), int):
            events = events[: event["truncate_to"]]
        events.append(event)
    return events


async def _replay_and_tail(task_id: str) -> AsyncIterator[dict]:
    state = _tasks.get(task_id)
    if state is None:
        # 内存里没有（过期被清理/进程重启过），但trace日志里能还原出一次完整
        # 结束的运行——直接重放，不用让用户重新发起
        trace_events = rebuild_events_from_trace(task_id)
        if trace_events and trace_events[-1].get("type") in _TERMINAL_EVENT_TYPES:
            for event in trace_events:
                yield event
            return
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
