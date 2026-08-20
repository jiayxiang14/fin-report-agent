"""极简的单进程内存限流：保护"真金白银会被打爆"的分析类端点——每次调用都是
一次真实的多轮LLM Agent Loop（Best-of-N 端点更是3倍），之前这几个端点谁都能
无限调用，没有任何防线。

用固定窗口（fixed window）按客户端IP计数，不引入Redis/第三方限流库——这是
个人demo项目单进程部署现状下足够的方案。局限（如实说明，不是遗漏）：这份
状态只存在当前进程内存里，多worker/多实例部署时each进程各算各的，起不到
统一限流的效果，到时候需要换成基于Redis的共享计数器。
"""

import time
from collections import defaultdict

from fastapi import HTTPException, Request

RATE_LIMIT_WINDOW_SECONDS = 60.0
RATE_LIMIT_MAX_REQUESTS = 5  # 单IP每个窗口最多5次，够手动测试用，挡住失控脚本式调用

_request_timestamps: dict[str, list[float]] = defaultdict(list)


def _client_key(request: Request) -> str:
    return request.client.host if request.client else "unknown"


def enforce_rate_limit(request: Request) -> None:
    key = _client_key(request)
    now = time.monotonic()
    window_start = now - RATE_LIMIT_WINDOW_SECONDS
    timestamps = _request_timestamps[key]
    while timestamps and timestamps[0] < window_start:
        timestamps.pop(0)
    if len(timestamps) >= RATE_LIMIT_MAX_REQUESTS:
        raise HTTPException(status_code=429, detail="请求过于频繁，请稍后再试")
    timestamps.append(now)
