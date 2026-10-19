"""服务启动时扫描检查点目录：自动接回"跑到一半被打断"的普通分析，清理过期
的失败记录。在 main.py 的 lifespan 里调用，跟缓存预热同一个位置。

只处理 `running` 状态——进程重启后还是 `running`，说明任务被打断时没有走到
任何终止分支（崩溃/kill/部署重启/--reload）。`failed_retryable` 不自动恢复：
那是外部故障（比如余额不足），重启服务本身解决不了，要等用户确认问题已解决
再手动点重试。

只支持单进程：多个实例同时启动会抢着恢复同一个任务，这要等 B 阶段用 Redis
锁解决（见 docs/开发日志.md）。
"""

from __future__ import annotations

import logging
import time

from app.api.session_guard import acquire_in_flight_request, release_in_flight_request
from app.services.agent import checkpoint, task_runner, trace_log

logger = logging.getLogger(__name__)


def _give_up(task_id: str, reason: str) -> None:
    checkpoint.update_meta(task_id, status=checkpoint.STATUS_FAILED, error=reason, failed_at=time.time())
    # 写进trace日志：前端刷新后订阅这个task_id，能从trace日志重放出
    # "前面跑到哪 + 为什么没能接着跑"，而不是一个笼统的"找不到任务"
    trace_log.append_event(task_id, {"type": "error", "code": "resume_failed", "message": reason})


def recover_pending_tasks() -> list[str]:
    """返回成功接回的task_id列表（给日志和测试用）。必须在事件循环里调用——
    接回的任务会用asyncio.create_task在后台跑。"""
    resumed: list[str] = []
    now = time.time()
    for task_id in checkpoint.list_task_ids():
        meta = checkpoint.read_meta(task_id)
        if meta is None:
            checkpoint.delete_task(task_id)
            continue

        status = meta.get("status")
        if status in (checkpoint.STATUS_FAILED, checkpoint.STATUS_FAILED_RETRYABLE):
            if now - meta.get("failed_at", now) > checkpoint.FAILED_RETENTION_SECONDS:
                checkpoint.delete_task(task_id)
            continue
        if status != checkpoint.STATUS_RUNNING:
            continue

        if meta.get("auto_resume_count", 0) >= checkpoint.MAX_AUTO_RESUMES:
            _give_up(task_id, f"服务重启后已自动恢复过 {checkpoint.MAX_AUTO_RESUMES} 次仍未完成，已停止自动恢复，请重新发起分析")
            continue
        # 先记一次再启动：如果恰恰是这个任务把进程搞崩，下次启动时计数已经加上了
        checkpoint.update_meta(task_id, auto_resume_count=meta.get("auto_resume_count", 0) + 1)

        session_id = meta.get("session_id")
        acquired = acquire_in_flight_request(session_id)
        try:
            task_runner.resume_normal_analysis(
                task_id, reason="server_restart", session_id=session_id if acquired else None
            )
        except task_runner.ResumeError as exc:
            if acquired:
                release_in_flight_request(session_id)
            _give_up(task_id, str(exc))
            continue
        resumed.append(task_id)
        logger.info("服务重启后自动恢复分析 task_id=%s ticker=%s", task_id, meta.get("ticker"))
    return resumed
