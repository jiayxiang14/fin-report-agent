"""eval_tool_selection.py 的纯函数部分测试：精确率/召回率计算、自我核查触发率、
日志读写——不真的跑Agent Loop（那需要真实调用SEC EDGAR/Polygon/Alpha Vantage/
LLM，真花钱，属于脚本使用者手动执行的范畴）。另外单独校验标注数据本身的结构
完整性（ticker格式、引用的工具名都是真实存在的工具），防止手写JSON时的typo
悄悄让某条标注失效。
"""

import json

from app.models.agent import TranscriptEntry
from app.services.agent.tools import TOOL_SCHEMAS
from scripts.eval_tool_selection import (
    SELF_VERIFICATION_TOOL,
    _actual_tools_called,
    _append_run_log,
    _error_result,
    _estimate_alpha_vantage_calls,
    _load_eval_set,
    _print_table,
    _score_tool_selection,
    _self_verification_triggered,
)


def _entry(tool_name: str, is_error: bool = False) -> TranscriptEntry:
    return TranscriptEntry(turn=0, tool_name=tool_name, tool_input={}, tool_output_summary="{}", is_error=is_error)


def test_actual_tools_called_excludes_errored_and_self_verification_calls():
    transcript = [
        _entry("get_financials"),
        _entry("get_sector_position", is_error=True),  # 失败的调用不算"真的调过"
        _entry("verify_number"),  # 自我核查单独统计，不进这个集合
    ]

    assert _actual_tools_called(transcript) == {"get_financials"}


def test_self_verification_triggered_requires_a_successful_call():
    assert _self_verification_triggered([_entry("verify_number", is_error=True)]) is False
    assert _self_verification_triggered([_entry("verify_number")]) is True
    assert _self_verification_triggered([]) is False


def test_score_tool_selection_perfect_match():
    expected = {"get_financials", "get_sector_position"}
    score = _score_tool_selection(expected, expected)

    assert score["precision"] == 1.0
    assert score["recall"] == 1.0
    assert score["missing"] == []
    assert score["extra"] == []


def test_score_tool_selection_reports_precision_and_recall_separately():
    """多调了一个不该调的工具，同时漏调了一个该调的——精确率和召回率应该
    分别反映这两种不同性质的问题，不是合成一个笼统的分数。"""
    expected = {"get_financials", "get_thematic_flow"}
    actual = {"get_financials", "get_peer_comparison"}  # 漏了get_thematic_flow，多调了get_peer_comparison

    score = _score_tool_selection(expected, actual)

    assert score["precision"] == 0.5  # actual里2个，对了1个
    assert score["recall"] == 0.5  # expected里2个，对了1个
    assert score["missing"] == ["get_thematic_flow"]
    assert score["extra"] == ["get_peer_comparison"]


def test_score_tool_selection_precision_is_none_when_nothing_was_called():
    score = _score_tool_selection({"get_financials"}, set())
    assert score["precision"] is None
    assert score["recall"] == 0.0


def test_estimate_alpha_vantage_calls_only_counts_earnings_surprise_expectation():
    eval_set = [
        {"ticker": "A", "expected_tools": ["get_financials", "get_earnings_surprise"]},
        {"ticker": "B", "expected_tools": ["get_financials"]},
    ]

    assert _estimate_alpha_vantage_calls(eval_set, repeats=3) == 3  # 只有A带get_earnings_surprise，1个ticker*3次


def test_error_result_has_no_score_fields():
    result = _error_result("ZZZZ", RuntimeError("上游挂了"))
    assert result["precision"] is None
    assert result["recall"] is None
    assert result["error"] == "上游挂了"


def test_append_run_log_writes_one_jsonl_line_per_call(tmp_path, monkeypatch):
    import scripts.eval_tool_selection as mod

    log_path = tmp_path / "eval_tool_selection_runs.jsonl"
    monkeypatch.setattr(mod, "RUNS_LOG_PATH", log_path)

    _append_run_log([{"ticker": "AAPL", "mean_precision": 1.0}])
    _append_run_log([{"ticker": "AAPL", "mean_precision": 0.8}])

    lines = log_path.read_text().strip().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["results"][0]["mean_precision"] == 1.0


def test_print_table_handles_a_fully_failed_ticker_without_crashing(capsys):
    _print_table([{"ticker": "ZZZZ", "n": 1, "mean_precision": None, "mean_recall": None, "self_verification_rate": None}])
    assert "运行失败" in capsys.readouterr().out


def test_run_one_forwards_temperature_to_agent_loop(monkeypatch):
    import asyncio

    import scripts.eval_tool_selection as mod

    captured_temperature = None

    class _FakeRunResult:
        completed = True
        transcript = [_entry("get_financials")]

    async def fake_run_agent_loop(ticker, temperature=None):
        nonlocal captured_temperature
        captured_temperature = temperature
        return _FakeRunResult()

    monkeypatch.setattr(mod, "run_agent_loop", fake_run_agent_loop)

    result = asyncio.run(mod._run_one("AAPL", ["get_financials"], temperature=0.0))

    assert captured_temperature == 0.0
    assert result["precision"] == 1.0
    assert result["recall"] == 1.0


# --- 标注数据本身的结构完整性校验，不是脚本逻辑测试 ---

_VALID_TOOL_NAMES = {schema["name"] for schema in TOOL_SCHEMAS} - {SELF_VERIFICATION_TOOL}


def test_eval_set_loads_and_is_non_empty():
    eval_set = _load_eval_set()
    assert len(eval_set) >= 5  # 覆盖度太窄的评测集统计意义不大


def test_eval_set_every_entry_has_required_fields():
    for entry in _load_eval_set():
        assert entry["ticker"], entry
        assert entry["expected_tools"], f"{entry['ticker']} 没有标注任何expected_tools"
        assert entry["notes"], f"{entry['ticker']} 缺少notes说明"
        assert entry["last_verified"], f"{entry['ticker']} 缺少last_verified日期"


def test_eval_set_expected_tools_only_reference_real_tool_names():
    """防止手写JSON时把工具名拼错——拼错的工具名会让那条标注实际上永远
    算成'漏调'，因为Agent真实调用的工具名不可能匹配一个不存在的名字。"""
    for entry in _load_eval_set():
        unknown = set(entry["expected_tools"]) - _VALID_TOOL_NAMES
        assert not unknown, f"{entry['ticker']} 的expected_tools引用了不存在的工具：{unknown}"


def test_eval_set_does_not_include_verify_number_in_expected_tools():
    """verify_number是单独统计的自我核查触发率，不该混进expected_tools里，
    混进去会在_score_tool_selection里产生一个永远不会被_actual_tools_called
    匹配到的'幽灵期望值'（因为verify_number被显式排除在actual集合之外）。"""
    for entry in _load_eval_set():
        assert SELF_VERIFICATION_TOOL not in entry["expected_tools"], entry["ticker"]
