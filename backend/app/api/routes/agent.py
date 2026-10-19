from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Path
from fastapi.responses import StreamingResponse

from app.api.rate_limit import enforce_rate_limit
from app.api.routes.trace import TRACE_ID_PATTERN
from app.api.session_guard import enforce_single_in_flight_request, release_in_flight_request
from app.api.ticker_path import TickerPath
from app.models.agent import AgentRunResult
from app.services.agent import checkpoint, task_registry, task_runner
from app.services.agent.loop import run_agent_loop

# task_id 跟 trace_id 是同一个值（uuid4().hex），复用同一个格式校验
TaskIdPath = Annotated[str, Path(pattern=TRACE_ID_PATTERN, description="发起分析时返回的task_id")]

router = APIRouter(prefix="/api/analyze", tags=["agent"])


@router.post("/{ticker}", response_model=AgentRunResult, dependencies=[Depends(enforce_rate_limit)])
async def analyze_ticker(ticker: TickerPath) -> AgentRunResult:
    try:
        return await run_agent_loop(ticker)
    except Exception as exc:  # noqa: BLE001 - LLM/网络层的意外失败，转成清晰的502
        raise HTTPException(status_code=502, detail=f"Agent Loop 运行失败：{exc}") from exc


@router.post("/{ticker}/start", dependencies=[Depends(enforce_rate_limit)])
async def start_analysis(
    ticker: TickerPath,
    session_id: str | None = Depends(enforce_single_in_flight_request),
) -> dict:
    """发起一次分析、立刻返回task_id，不等待跑完——这一步会真实触发LLM调用，
    所以限流挂在这里，不是挂在订阅端点上（`/stream/{task_id}`纯粹是订阅/
    重放已经在跑的task，不产生新的LLM调用，不该占用户的限流额度）。
    检查点、失败分类、会话锁释放都在task_runner里，跟重启恢复/手动重试共用。"""
    return {"task_id": task_runner.start_normal_analysis(ticker, session_id)}


@router.post("/tasks/{task_id}/cancel")
async def cancel_analysis(task_id: TaskIdPath) -> dict:
    """用户主动取消一次正在跑的分析（普通分析和深度分析共用）。不挂限流：
    取消只会减少花费。task_id是不可猜测的uuid，拿到它的只有发起者本人。"""
    if not task_registry.cancel_task(task_id):
        raise HTTPException(status_code=404, detail="没有正在进行的这次分析（可能已经结束）")
    return {"cancelled": True}


@router.post("/tasks/{task_id}/retry", dependencies=[Depends(enforce_rate_limit)])
async def retry_analysis(
    task_id: TaskIdPath,
    session_id: str | None = Depends(enforce_single_in_flight_request),
) -> dict:
    """API持续故障（余额不足、服务宕机）导致的失败，从最近的存档点接着跑，
    沿用同一个task_id——前端重新订阅同一条流即可。会真实触发LLM调用，所以
    跟/start一样挂限流和会话锁。"""
    try:
        if task_registry.is_running(task_id):
            raise HTTPException(status_code=409, detail="这次分析还在进行中，不需要重试")
        meta = checkpoint.read_meta(task_id)
        if meta is None or meta.get("status") != checkpoint.STATUS_FAILED_RETRYABLE:
            raise HTTPException(status_code=409, detail="这次分析不支持从断点重试，请重新发起分析")
        task_runner.resume_normal_analysis(task_id, reason="retry", session_id=session_id)
    except task_runner.ResumeError as exc:
        release_in_flight_request(session_id)
        raise HTTPException(status_code=410, detail=str(exc)) from exc
    except HTTPException:
        release_in_flight_request(session_id)
        raise
    return {"task_id": task_id}


@router.get("/stream/{task_id}")
async def analyze_stream(task_id: str) -> StreamingResponse:
    """订阅一次分析的事件流——不管这个task是刚发起的、正在跑的、还是已经
    跑完的，连接上先重放全部历史事件（缓冲区），再继续实时推送。刷新页面/
    网络抖动重连，接的是同一个task_id，不会重复触发LLM调用。"""
    return task_registry.stream_task(task_id)
