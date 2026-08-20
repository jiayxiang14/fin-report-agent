import uuid

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from app.api.rate_limit import enforce_rate_limit
from app.api.session_guard import enforce_single_in_flight_request, release_in_flight_request
from app.api.ticker_path import TickerPath
from app.services.agent import task_registry
from app.services.agent.best_of_n import run_best_of_n

router = APIRouter(prefix="/api/analyze", tags=["best_of_n"])


@router.post("/{ticker}/best-of-n/start", dependencies=[Depends(enforce_rate_limit)])
async def start_best_of_n(
    ticker: TickerPath,
    session_id: str | None = Depends(enforce_single_in_flight_request),
) -> dict:
    """跟 agent.py 的 `/start` 同一套task_registry实现——Best-of-N一次要跑3次
    完整Agent Loop+最多3次裁判调用，成本比普通分析高得多，断线重连更不能
    默默变成"重新触发一次"，跟普通分析共用同一个任务注册表（task_id全局
    唯一，注册表不关心是哪种分析）。session并发保护同样共用一份状态——同一个
    浏览器标签页不该同时既跑一次普通分析又跑一次深度分析，两者对Alpha
    Vantage/Polygon是同一批共享配额。

    提前生成task_id（不让task_registry.start_task内部生成）并传给
    run_best_of_n当trace_id——胜出候选的完整工具输出会被提升进这个task_id
    对应的trace文件，跟task_registry落的事件级记录共用同一份。
    """
    task_id = uuid.uuid4().hex

    async def runner(on_event: task_registry.OnEvent) -> dict:
        try:
            result = await run_best_of_n(ticker, on_event=on_event, trace_id=task_id)
            return result.model_dump()
        finally:
            release_in_flight_request(session_id)

    task_registry.start_task(ticker, runner, task_id=task_id)
    return {"task_id": task_id}


@router.get("/best-of-n/stream/{task_id}")
async def analyze_best_of_n_stream(task_id: str) -> StreamingResponse:
    return task_registry.stream_task(task_id)
