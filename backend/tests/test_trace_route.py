"""GET /api/trace/{trace_id}：按task_id事后查询一次分析的完整持久化事件轨迹，
不依赖task还在task_registry的内存字典里存活。
"""

from unittest.mock import patch

from fastapi.testclient import TestClient

from app.main import app
from app.services.agent import trace_log

client = TestClient(app)


def test_get_trace_returns_persisted_events(tmp_path):
    trace_id = "b" * 32
    with patch.object(trace_log, "TRACE_DIR", tmp_path):
        trace_log.append_event(trace_id, {"type": "reasoning", "turn": 0, "text": "推理中"})
        trace_log.append_event(trace_id, {"type": "done", "result": {"final_report": "ok"}})

        response = client.get(f"/api/trace/{trace_id}")

    assert response.status_code == 200
    events = response.json()
    assert [e["type"] for e in events] == ["reasoning", "done"]


def test_get_trace_returns_404_for_unknown_trace_id(tmp_path):
    with patch.object(trace_log, "TRACE_DIR", tmp_path):
        response = client.get(f"/api/trace/{'c' * 32}")

    assert response.status_code == 404


def test_get_trace_rejects_malformed_trace_id():
    """路径穿越类字符（比如"../"）应该在路由层就被 Path(pattern=...) 挡掉，
    返回422而不是钻到service层——这条不需要mock TRACE_DIR，请求根本到不了
    trace_log.read_trace。"""
    response = client.get("/api/trace/not-a-valid-trace-id")

    assert response.status_code == 422
