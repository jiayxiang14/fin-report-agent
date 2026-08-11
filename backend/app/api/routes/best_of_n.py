import asyncio
import contextlib
import json

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from app.services.agent.best_of_n import run_best_of_n

router = APIRouter(prefix="/api/analyze", tags=["best_of_n"])


def _sse(event: dict) -> str:
    return f"data: {json.dumps(event, ensure_ascii=False)}\n\n"


@router.get("/{ticker}/best-of-n/stream")
async def analyze_ticker_best_of_n_stream(ticker: str) -> StreamingResponse:
    """跟 agent.py 的 `/stream` 端点同一套"asyncio.Queue桥接on_event"实现，
    是一个独立的新端点而不是替换普通分析：Best-of-N一次要跑3次完整Agent Loop
    +最多3次裁判调用，API调用量/花费大约是普通分析的4倍，不能默默变成默认
    行为，用户需要能明确选择要不要用这个更贵的"深度分析"模式。

    run_best_of_n内部现在是并行跑3个候选（asyncio.gather，靠cache_lock保证
    并发访问同一份磁盘缓存安全），所以墙钟延迟不再是"4倍"那么夸张——三个候选
    的事件会按各自真实产生的时间交错到达这个队列里，不是排队等前一个候选
    完全跑完才轮到下一个。这里只是原样把队列里的事件转发成SSE帧，不关心
    到达顺序，前端本来就按candidate_index分桶处理，天然兼容交错到达。

    事件在 run_best_of_n/run_agent_loop 已有的 reasoning/tool_call_started/
    tool_call_finished 基础上多带一个 candidate_index 字段区分是哪个候选，
    新增 candidate_scored（一个候选跑完+打分后发出），最终 done 携带完整的
    BestOfNResult。
    """

    async def event_generator():
        queue: asyncio.Queue = asyncio.Queue()

        async def on_event(event: dict) -> None:
            await queue.put(event)

        async def run() -> None:
            try:
                result = await run_best_of_n(ticker, on_event=on_event)
                await queue.put({"type": "done", "result": result.model_dump()})
            except Exception as exc:  # noqa: BLE001 - 流式端点的失败边界
                await queue.put({"type": "error", "message": str(exc)})
            finally:
                await queue.put(None)

        task = asyncio.create_task(run())
        try:
            while True:
                event = await queue.get()
                if event is None:
                    break
                yield _sse(event)
        finally:
            if not task.done():
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task

    return StreamingResponse(event_generator(), media_type="text/event-stream")
