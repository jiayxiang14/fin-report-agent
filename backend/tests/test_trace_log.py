"""trace_log.py：把task_registry的事件额外落盘成JSONL，支持事后按trace_id
查询。这里直接测模块本身（写入/读回/非法trace_id防护/落盘失败的软降级），
`test_task_registry.py`覆盖它跟task_registry的接线，`test_trace_route.py`
覆盖查询端点。
"""

from unittest.mock import patch

from app.services.agent import trace_log

VALID_TRACE_ID = "a" * 32


def test_append_then_read_round_trips_events(tmp_path):
    with patch.object(trace_log, "TRACE_DIR", tmp_path):
        trace_log.append_event(VALID_TRACE_ID, {"type": "reasoning", "turn": 0, "text": "推理中"})
        trace_log.append_event(VALID_TRACE_ID, {"type": "tool_call_started", "turn": 0, "tool_name": "get_financials"})

        events = trace_log.read_trace(VALID_TRACE_ID)

    assert [e["type"] for e in events] == ["reasoning", "tool_call_started"]
    assert events[0]["text"] == "推理中"
    assert "logged_at" in events[0]  # 每条记录都带时间戳，不是只有原始事件字段


def test_read_trace_preserves_write_order(tmp_path):
    with patch.object(trace_log, "TRACE_DIR", tmp_path):
        for i in range(5):
            trace_log.append_event(VALID_TRACE_ID, {"type": "reasoning", "turn": i})
        events = trace_log.read_trace(VALID_TRACE_ID)

    assert [e["turn"] for e in events] == [0, 1, 2, 3, 4]


def test_read_trace_for_unknown_trace_id_returns_empty_list(tmp_path):
    with patch.object(trace_log, "TRACE_DIR", tmp_path):
        assert trace_log.read_trace(VALID_TRACE_ID) == []


def test_append_event_rejects_malformed_trace_id_without_raising(tmp_path):
    """真实的trace_id永远是task_registry生成的uuid4().hex，但这个函数不该
    信任调用方——路径穿越字符（比如"../../etc/passwd"）不该被拼进文件路径。
    非法trace_id应该静默丢弃，不写入任何文件，也不抛异常。"""
    with patch.object(trace_log, "TRACE_DIR", tmp_path):
        trace_log.append_event("../../etc/passwd", {"type": "reasoning"})

        assert list(tmp_path.iterdir()) == []


def test_read_trace_rejects_malformed_trace_id(tmp_path):
    with patch.object(trace_log, "TRACE_DIR", tmp_path):
        assert trace_log.read_trace("../../etc/passwd") == []


def test_append_event_swallows_disk_write_failures(tmp_path):
    """落盘失败（磁盘满/权限问题）是锦上添花功能的失败，不能让调用方感知到——
    这里模拟mkdir抛OSError，断言不会往外传播异常。"""
    with patch.object(trace_log, "TRACE_DIR", tmp_path / "nested"):
        with patch.object(trace_log.Path, "mkdir", side_effect=OSError("磁盘满了")):
            trace_log.append_event(VALID_TRACE_ID, {"type": "reasoning"})  # 不应该抛异常


def test_append_event_swallows_non_json_serializable_content(tmp_path):
    """真实验证过的漏洞：event字典里混进不可JSON序列化的内容时，json.dumps
    抛的是TypeError，不是OSError——之前只catch OSError时这条路径没被兜住，
    会真的往外传播。Best-of-N场景下这会让一次纯粹的可观测性写入失败，报废
    一个已经真实花了钱的候选（异常从capture一路冒泡到_run_candidate外层的
    except，被判定成整个候选失败）。"""

    class NotSerializable:
        pass

    with patch.object(trace_log, "TRACE_DIR", tmp_path):
        trace_log.append_event(VALID_TRACE_ID, {"type": "reasoning", "bad": NotSerializable()})  # 不应该抛异常

        assert trace_log.read_trace(VALID_TRACE_ID) == []  # 这一条没写成，但没有崩


def test_read_trace_skips_corrupted_lines(tmp_path):
    with patch.object(trace_log, "TRACE_DIR", tmp_path):
        trace_file = tmp_path / f"{VALID_TRACE_ID}.jsonl"
        tmp_path.mkdir(parents=True, exist_ok=True)
        trace_file.write_text('{"type": "reasoning", "turn": 0}\nnot valid json\n{"type": "reasoning", "turn": 1}\n')

        events = trace_log.read_trace(VALID_TRACE_ID)

    assert [e["turn"] for e in events] == [0, 1]
