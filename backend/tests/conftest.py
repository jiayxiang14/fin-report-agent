"""全局测试隔离：`rate_limit.py`的限流状态是进程级的模块全局字典，不清空的
话，不同测试文件对`/api/analyze/*`的真实请求次数会在同一次pytest进程里累加，
某个测试单独跑是绿的，但跟别的测试一起跑可能因为"额度被前面的测试用掉了"
而失败——这跟这次CI复盘发现的"测试依赖未重置的共享状态"是同一类问题，这里
直接用autouse fixture在每个测试前清空，从源头避免。
"""

import pytest

from app.api.rate_limit import _request_timestamps
from app.api.session_guard import _in_flight_sessions
from app.services.agent import best_of_n, checkpoint, loop, task_registry, trace_log


@pytest.fixture(autouse=True)
def _reset_rate_limit_state():
    _request_timestamps.clear()
    yield
    _request_timestamps.clear()


@pytest.fixture(autouse=True)
def _isolate_runtime_records(tmp_path, monkeypatch):
    """所有"运行时落盘的记录"统一重定向到临时目录，不让测试写进仓库的
    .cache/。真实踩过的坑：之前只有个别测试自己重定向了best_of_n_runs.jsonl，
    其余测试每跑一次pytest就往真实日志里追加4条假的Best-of-N运行记录（简报
    内容是"候选0"这类占位符），日积月累636条假数据混在8条真实记录里，项目
    总结书据此写出了"133条真实运行记录"这种错误结论；trace日志每跑一次多出
    22个文件，上下文压缩审计日志同样被污染。检查点不重定向的话，下次本地
    启动服务时还会被当成"跑到一半被打断"的任务去自动恢复。

    数据源缓存（companyfacts/polygon_bars等）不在这里：测试本来就mock掉了
    外部请求，不会写它们。"""
    monkeypatch.setattr(checkpoint, "CHECKPOINT_DIR", tmp_path / "checkpoints")
    monkeypatch.setattr(trace_log, "TRACE_DIR", tmp_path / "traces")
    monkeypatch.setattr(best_of_n, "RUNS_LOG_PATH", tmp_path / "best_of_n_runs.jsonl")
    monkeypatch.setattr(best_of_n, "CANDIDATE_TRACE_DIR", tmp_path / "candidate_traces")
    monkeypatch.setattr(loop, "CONTEXT_COMPACTION_LOG_PATH", tmp_path / "context_compaction.jsonl")
    yield


@pytest.fixture(autouse=True)
def _reset_session_guard_state():
    _in_flight_sessions.clear()
    yield
    _in_flight_sessions.clear()


@pytest.fixture(autouse=True)
def _reset_task_registry():
    """task_registry的任务表也是模块级全局状态：用固定task_id的测试之间会
    互相串（前一个测试留下的已完成task，会让后一个测试从内存而不是trace
    日志重放）。"""
    task_registry._tasks.clear()
    yield
    task_registry._tasks.clear()
