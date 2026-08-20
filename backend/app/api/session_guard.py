"""同一个session同时只允许1个"进行中"的分析请求——不是防滥用（`X-Session-Id`
是前端自己生成、没有任何验证的，故意换一个新session_id就能绕过，这不是安全
机制），是防意外：这个项目真实踩过"同一个逻辑动作被意外触发两次"的坑（React
StrictMode导致SSE连接被开两次，`useAgentAnalysis.js`专门用`start()`函数模式
+`sessionStorage`记录task_id规避过前端这一侧），这里补的是后端这一侧对应的
保险丝——万一前端的防重复逻辑失效（或者用户手快点了两次"开始分析"），不该
让同一个session同时跑两次真实花钱的Agent Loop，抢占本来就紧张的Alpha
Vantage/Polygon配额。

跟 `rate_limit.py` 的IP限速是两回事、互相补充，不是重复：那个限的是"一段
时间窗口内请求总数"（挡持续滥用），5次/60秒这个窗口宽松到完全挡不住"同一个
动作几乎同时触发了两次"这种瞬时重复提交场景——这个限的正是这个场景。

用 session_id（不是IP）做key：同一个IP背后可能是同一个办公网络下的好几个
不同用户各自分析不同ticker，按IP做"同时只能1个"会把这些人互相挡住，产生
大量误伤。session_id由前端生成、存在`sessionStorage`（不是`localStorage`）
里，天然按"一个浏览器标签页"这个粒度隔离，两个标签页各自分析不同ticker不
该互相挡住。

模块级内存集合存状态（跟`alpha_vantage_client.py`的配额计数器同一个模式），
依赖"uvicorn单进程"这个项目已有的架构前提（`cache_lock.py`顶部注释点破的
同一个假设）——多worker部署时这份状态不共享，到时候需要换成Redis。没有
`X-Session-Id`请求头时（curl测试、老版本前端）直接放行，不强制要求这个头，
这是一个可选的额外保护，不是硬性门槛。
"""

from fastapi import Header, HTTPException

_in_flight_sessions: set[str] = set()


def enforce_single_in_flight_request(x_session_id: str | None = Header(default=None)) -> str | None:
    """FastAPI依赖：既做门禁（同一个session已有进行中的请求时拒绝），也把
    session_id原样返回给路由——路由需要拿着同一个值，在分析真正跑完（不是
    HTTP请求本身返回）的时候调用`release_in_flight_request`。"""
    if x_session_id is None:
        return None
    if x_session_id in _in_flight_sessions:
        raise HTTPException(
            status_code=409,
            detail="这个session已经有一个分析正在进行，请等它完成或刷新页面后重试",
        )
    _in_flight_sessions.add(x_session_id)
    return x_session_id


def release_in_flight_request(session_id: str | None) -> None:
    """必须在后台Agent Loop真正跑完（成功或失败）时调用，不是`/start`这个
    HTTP请求返回的时候——`/start`本身立刻返回task_id，真正耗时的分析在
    `task_registry`的后台task里跑，"进行中"这个状态要跟着那个task的生命
    周期走，不是跟着这次HTTP请求/响应的生命周期走。"""
    if session_id is not None:
        _in_flight_sessions.discard(session_id)
