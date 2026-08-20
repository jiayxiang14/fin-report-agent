"""session_guard.py：同一个session（前端生成、放在X-Session-Id请求头里的
标识符）同时只允许1个"进行中"的分析请求。这里直接测依赖函数本身，路由层的
集成场景（真实通过FastAPI依赖注入触发409、后台任务跑完后释放）在
`test_agent_stream_route.py`/`test_best_of_n_route.py`里覆盖。
"""

import pytest
from fastapi import HTTPException

from app.api.session_guard import enforce_single_in_flight_request, release_in_flight_request


@pytest.fixture(autouse=True)
def _reset_in_flight_sessions(monkeypatch):
    monkeypatch.setattr("app.api.session_guard._in_flight_sessions", set())


def test_missing_session_id_is_always_allowed_and_returns_none():
    assert enforce_single_in_flight_request(x_session_id=None) is None
    assert enforce_single_in_flight_request(x_session_id=None) is None  # 反复调用也不该被拦


def test_first_request_for_a_session_id_is_allowed_and_marks_it_in_flight():
    result = enforce_single_in_flight_request(x_session_id="s1")
    assert result == "s1"


def test_second_request_for_the_same_session_id_is_rejected_before_release():
    enforce_single_in_flight_request(x_session_id="s1")

    with pytest.raises(HTTPException) as exc_info:
        enforce_single_in_flight_request(x_session_id="s1")

    assert exc_info.value.status_code == 409


def test_different_session_ids_do_not_block_each_other():
    enforce_single_in_flight_request(x_session_id="s1")

    result = enforce_single_in_flight_request(x_session_id="s2")

    assert result == "s2"


def test_session_id_is_usable_again_after_release():
    enforce_single_in_flight_request(x_session_id="s1")
    release_in_flight_request("s1")

    result = enforce_single_in_flight_request(x_session_id="s1")

    assert result == "s1"


def test_releasing_a_session_id_that_was_never_marked_does_not_raise():
    release_in_flight_request("never-seen")  # 不该抛异常，比如None作为session_id传进来的场景
    release_in_flight_request(None)
