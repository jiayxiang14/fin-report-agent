"""按 key 复用的进程内互斥锁，给"检查磁盘缓存是否存在→不存在就发请求→写缓存"
这套逻辑做并发保护。

不是操作系统级别的文件锁（那是跨进程场景才需要的，比如多个 worker 进程共享同一份
缓存目录）——这个项目从始至终是单进程跑（uvicorn 没有开多 worker），所有并发都
发生在同一个 asyncio 事件循环里，asyncio.Lock 已经足够，不需要引入额外依赖。

真正要防的场景：Best-of-N 并行跑 N 个候选时，同一个 ticker 的多个候选会几乎
同时请求同一份数据（比如都要 AAPL 的财务数据）——如果不做互斥，缓存冷启动时
会出现"都检查到不存在→都发起网络请求→都写同一个文件"的竞态，既重复消耗上游
限速配额，也有并发写同一个文件时数据损坏、或读到另一个协程写到一半的文件的
风险。加锁之后，同一个 key 的并发请求会排队，第一个真正去发请求+写缓存，
后面几个拿到锁时缓存已经是新鲜的，直接读缓存，不重复发请求——顺带也省了
重复的上游调用，不只是"安全"，还更省配额。
"""

import asyncio
from collections import defaultdict

_locks: defaultdict[str, asyncio.Lock] = defaultdict(asyncio.Lock)


def get_lock(key: str) -> asyncio.Lock:
    return _locks[key]
