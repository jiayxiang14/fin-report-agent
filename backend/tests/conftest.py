"""全局测试隔离：`rate_limit.py`的限流状态是进程级的模块全局字典，不清空的
话，不同测试文件对`/api/analyze/*`的真实请求次数会在同一次pytest进程里累加，
某个测试单独跑是绿的，但跟别的测试一起跑可能因为"额度被前面的测试用掉了"
而失败——这跟这次CI复盘发现的"测试依赖未重置的共享状态"是同一类问题，这里
直接用autouse fixture在每个测试前清空，从源头避免。
"""

import pytest

from app.api.rate_limit import _request_timestamps


@pytest.fixture(autouse=True)
def _reset_rate_limit_state():
    _request_timestamps.clear()
    yield
    _request_timestamps.clear()
