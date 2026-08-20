import uuid

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse

from app.api.rate_limit import enforce_rate_limit
from app.api.session_guard import enforce_single_in_flight_request, release_in_flight_request
from app.api.ticker_path import TickerPath
from app.models.agent import AgentRunResult
from app.services.agent import task_registry, trace_log
from app.services.agent.loop import run_agent_loop

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
    重放已经在跑的task，不产生新的LLM调用，不该占用户的限流额度）。"""
    # 提前生成task_id（不让task_registry.start_task内部生成），这样才能把它
    # 闭包进on_tool_result里——task_registry对每个事件本身已经落一份到trace_log
    # 了，但那份是SSE同款的、给verify_number/get_filing_text这类工具截断到
    # 300字符的summary，事后排查"这个数字到底哪里对不上"意义有限。这里额外
    # 用on_tool_result拿到完整未截断的工具输出，单独落一条trace事件，跟
    # task_registry落的那份共用同一个trace_id文件，按发生顺序自然穿插。
    task_id = uuid.uuid4().hex

    async def on_tool_result(tool_name: str, output: str) -> None:
        trace_log.append_event(task_id, {"type": "tool_result_full", "tool_name": tool_name, "output": output})

    async def runner(on_event: task_registry.OnEvent) -> dict:
        # session的"进行中"状态要跟着后台Agent Loop真正跑完的时机释放，不是
        # 跟着这次/start的HTTP请求返回的时机——那个几乎立刻就结束了，这个
        # try/finally覆盖的是runner的整个生命周期，成功/失败都会释放。
        try:
            result = await run_agent_loop(ticker, on_event=on_event, on_tool_result=on_tool_result)
            return result.model_dump()
        finally:
            release_in_flight_request(session_id)

    task_registry.start_task(ticker, runner, task_id=task_id)
    return {"task_id": task_id}


@router.get("/stream/{task_id}")
async def analyze_stream(task_id: str) -> StreamingResponse:
    """订阅一次分析的事件流——不管这个task是刚发起的、正在跑的、还是已经
    跑完的，连接上先重放全部历史事件（缓冲区），再继续实时推送。刷新页面/
    网络抖动重连，接的是同一个task_id，不会重复触发LLM调用。"""
    return task_registry.stream_task(task_id)
