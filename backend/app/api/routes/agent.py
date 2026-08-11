import asyncio
import contextlib
import json

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from app.models.agent import AgentRunResult
from app.services.agent.loop import run_agent_loop

router = APIRouter(prefix="/api/analyze", tags=["agent"])


@router.post("/{ticker}", response_model=AgentRunResult)
async def analyze_ticker(ticker: str) -> AgentRunResult:
    try:
        return await run_agent_loop(ticker)
    except Exception as exc:  # noqa: BLE001 - LLM/网络层的意外失败，转成清晰的502
        raise HTTPException(status_code=502, detail=f"Agent Loop 运行失败：{exc}") from exc


def _sse(event: dict) -> str:
    return f"data: {json.dumps(event, ensure_ascii=False)}\n\n"


@router.get("/{ticker}/stream")
async def analyze_ticker_stream(ticker: str) -> StreamingResponse:
    """流式展示Agent Loop的推理过程（Stage 4）：`run_agent_loop`本身不知道有
    SSE这回事，这里用一个 asyncio.Queue 把它的 on_event 回调桥接成SSE事件流——
    "跑Agent Loop"和"把事件转成SSE帧"是两个并发任务，前者往队列里塞事件，
    后者从队列里取事件就地yield出去。

    失败走一个 `{"type": "error"}` 事件而不是HTTP错误状态码：SSE响应一旦开始
    推流，状态码已经是200了，无法再改成502，只能在流里带一个error事件告诉
    前端这次分析失败了。
    """

    async def event_generator():
        queue: asyncio.Queue = asyncio.Queue()

        async def on_event(event: dict) -> None:
            await queue.put(event)

        async def run() -> None:
            try:
                result = await run_agent_loop(ticker, on_event=on_event)
                await queue.put({"type": "done", "result": result.model_dump()})
            except Exception as exc:  # noqa: BLE001 - 流式端点的失败边界
                await queue.put({"type": "error", "message": str(exc)})
            finally:
                await queue.put(None)  # 结束哨兵

        task = asyncio.create_task(run())
        try:
            while True:
                event = await queue.get()
                if event is None:
                    break
                yield _sse(event)
        finally:
            # 客户端提前断开连接时（比如关掉浏览器标签页），不能只是发起取消就
            # 撒手不管——`task.cancel()`只是请求取消，真正让 run_agent_loop
            # 内部的 await 点抛出 CancelledError 并让协程收尾，需要再 await 一次
            # 才算数，否则这里的生成器可能在后台任务还没真正停下来之前就退出了
            if not task.done():
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task

    return StreamingResponse(event_generator(), media_type="text/event-stream")
