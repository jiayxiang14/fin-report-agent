"""按trace_id（就是发起分析时task_registry返回的task_id）查询一次Agent运行的
完整持久化事件轨迹——跟`/api/analyze/stream/{task_id}`不同，这个端点读的是
磁盘上的记录，不依赖task还在`task_registry._tasks`内存字典里存活，task过期
或进程重启之后依然能查。
"""

from typing import Annotated

from fastapi import APIRouter, HTTPException, Path

from app.services.agent import trace_log

# trace_id 就是 uuid.uuid4().hex，跟 ticker_path.py 校验 ticker 格式是同一个
# 目的：把非法输入（比如路径穿越字符）挡在路由层，trace_log.py 内部虽然也有
# 一道防御性校验，但错误信息应该在这里就给出清晰的400，而不是让请求钻到
# service层才发现格式不对。
TRACE_ID_PATTERN = r"^[0-9a-f]{32}$"

TraceIdPath = Annotated[str, Path(pattern=TRACE_ID_PATTERN, description="trace_id，即发起分析时返回的task_id")]

router = APIRouter(prefix="/api/trace", tags=["trace"])


@router.get("/{trace_id}")
async def get_trace(trace_id: TraceIdPath) -> list[dict]:
    events = trace_log.read_trace(trace_id)
    if not events:
        raise HTTPException(status_code=404, detail=f"trace_id '{trace_id}' 不存在或还没有任何记录")
    return events
