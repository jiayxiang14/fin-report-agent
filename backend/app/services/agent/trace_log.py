"""每次Agent运行的SSE事件，除了实时推给前端，额外落一份到磁盘，支持事后按
trace_id（复用task_registry已有的task_id，不发明第二个并行的标识符）查询完整
轨迹——之前这些事件只活在`task_registry._tasks`这个进程内内存字典里，跑完/
过期（按交易日）/进程重启就彻底没了，没有任何可持久化查询的记录。

写入实现是纯同步阻塞I/O（不是async函数），直接复用loop.py里`_append_compaction_log`
已经在用的同一个模式：asyncio单线程协作式调度下，一段没有await的同步代码块
本身就是原子的，不会被同一事件循环里的其它协程打断——不需要额外引入
`cache_lock`那类锁，只要保证"从组装这行JSON到写完文件"这段代码里不出现任何
await就够了。JSONL的行序天然就是事件的发生顺序，不需要额外的序号字段。

落盘失败绝不能拖垮正在跑的Agent Loop——这是锦上添花的可观测性数据，不是
核心链路，失败直接吞掉，不重试不往外抛。这不只是"磁盘满/权限问题"这类
`OSError`：event字典里混进不可JSON序列化的内容时`json.dumps`会抛
`TypeError`，实测验证过这条路径原来没被兜住，会真的往外传——在Best-of-N
场景下这个代价被放大成"一次纯粹的可观测性写入失败，白白报废一个已经真实
花了钱的候选"（`on_tool_result`失败会被`_run_candidate`外层的`except`
兜住，但兜底动作是把整个候选判定为失败），所以这里改成兜住所有异常，不是
只兜`OSError`。
"""

import json
import re
from datetime import UTC, datetime
from pathlib import Path

from app.services.polygon_client import CACHE_DIR

TRACE_DIR = CACHE_DIR / "traces"

# trace_id 在正常流程里永远是 task_registry 生成的 uuid4().hex，但这个模块的
# 读写函数本身不应该假设调用方一定守规矩——尤其是 read_trace 最终会被查询
# 路由拿externally-supplied的字符串直接调用，不校验格式就拼进文件路径会有
# 路径穿越风险（比如传"../../../etc/passwd"）。路由层已经用 Path(pattern=...)
# 挡了一道（跟 ticker_path.py 同一个模式），这里再挡一道是防御性的第二道关卡。
_TRACE_ID_PATTERN = re.compile(r"^[0-9a-f]{32}$")


def _trace_file(trace_id: str) -> Path | None:
    if not _TRACE_ID_PATTERN.match(trace_id):
        return None
    return TRACE_DIR / f"{trace_id}.jsonl"


def append_event(trace_id: str, event: dict) -> None:
    trace_file = _trace_file(trace_id)
    if trace_file is None:
        return
    try:
        TRACE_DIR.mkdir(parents=True, exist_ok=True)
        line = json.dumps({"logged_at": datetime.now(UTC).isoformat(), **event}, ensure_ascii=False)
        with trace_file.open("a") as f:
            f.write(line + "\n")
    except Exception:  # noqa: BLE001 - 落盘是锦上添花的可观测性数据，任何失败都要吞掉，不能拖垮Agent Loop
        pass


def read_trace(trace_id: str) -> list[dict]:
    trace_file = _trace_file(trace_id)
    if trace_file is None or not trace_file.exists():
        return []
    events = []
    for line in trace_file.read_text().splitlines():
        if not line:
            continue
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return events
