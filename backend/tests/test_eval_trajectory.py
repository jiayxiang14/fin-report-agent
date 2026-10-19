"""eval_trajectory.py 各环节判定的纯函数部分。不调用LLM和数据源。"""

import asyncio
import json

from app.models.agent import AgentRunResult, ReasoningNote, TranscriptEntry
from scripts import eval_trajectory as mod

REPORT = "<conclusion>c</conclusion><evidence>e</evidence><flags>f</flags>"


def _entry(tool, turn=0, is_error=False, **tool_input):
    tool_input = {"ticker": "AAPL", **tool_input}
    return TranscriptEntry(turn=turn, tool_name=tool, tool_input=tool_input, tool_output_summary="", is_error=is_error)


def _result(transcript, notes=None, **overrides):
    base = dict(
        ticker="AAPL", completed=True, stop_reason="end_turn", final_report=REPORT,
        reasoning_notes=notes or [ReasoningNote(turn=1, text=REPORT)], transcript=transcript, turns_used=3,
        self_verification_triggered=any(e.tool_name == "verify_number" and not e.is_error for e in transcript),
        traceable_numbers_matched=8, traceable_numbers_total=10,
    )
    base.update(overrides)
    return AgentRunResult(**base)


def test_param_validation_against_tool_schemas():
    assert mod.validate_params("get_financials", {"ticker": "AAPL"}) == []
    assert mod.validate_params("verify_number", {"ticker": "AAPL", "metric": "revenue", "claimed_value": 1.0, "period": "annual"}) == []
    problems = mod.validate_params("verify_number", {"ticker": "AAPL", "metric": "营收", "claimed_value": "1亿"})
    assert any("缺少必填参数 period" in p for p in problems)
    assert any("metric" in p and "不在允许值" in p for p in problems)
    assert any("claimed_value 应为数字" in p for p in problems)
    assert mod.validate_params("get_filing_text", {"ticker": "AAPL", "form": "20-F"})  # 20-F不在允许值里
    assert mod.validate_params("no_such_tool", {}) == ["未知工具 no_such_tool"]


def test_ticker_mismatch_and_duplicate_calls():
    transcript = [
        _entry("get_financials"),
        _entry("get_financials"),  # 成功后又用同样参数调一次
        _entry("get_peer_comparison", is_error=True),
        _entry("get_peer_comparison"),  # 失败后重试，不算重复
        _entry("get_financials", ticker="MSFT"),
    ]
    assert mod.duplicate_calls(transcript) == 1
    assert mod.ticker_mismatches(transcript, "AAPL") == ["get_financials(MSFT)"]


def test_verify_must_come_after_the_first_draft():
    """真实发现：两次运行都是先核实、后写简报——核的是"打算写的数字"，不是写下的数字。"""
    verify = _entry("verify_number", turn=2, metric="revenue", claimed_value=1.0, period="annual")
    after = _result([verify], notes=[ReasoningNote(turn=1, text=REPORT), ReasoningNote(turn=3, text=REPORT)])
    before = _result([verify], notes=[ReasoningNote(turn=1, text="先核实一下"), ReasoningNote(turn=3, text=REPORT)])
    assert mod.verify_after_draft(after) is True
    assert mod.verify_after_draft(before) is False
    assert mod.verify_after_draft(_result([_entry("get_financials")])) is None


def test_tool_failure_handling_distinguishes_recovered_disclosed_ignored():
    recovered = _result([_entry("get_financials", turn=0, is_error=True), _entry("get_financials", turn=1)])
    disclosed = _result([_entry("get_peer_comparison", is_error=True)], data_gap_disclosure_gate_resolved=True)
    ignored = _result([_entry("get_peer_comparison", is_error=True)], data_gap_disclosure_gate_resolved=False)
    assert mod.tool_failure_handling(recovered) == {"get_financials": "recovered"}
    assert mod.tool_failure_handling(disclosed) == {"get_peer_comparison": "disclosed"}
    assert mod.tool_failure_handling(ignored) == {"get_peer_comparison": "ignored"}
    # verify_number失败是核查请求本身的问题，不算工具失败
    assert mod.tool_failure_handling(_result([_entry("verify_number", is_error=True)])) == {}


def test_gate_outcomes_and_premature_stop():
    result = _result(
        [_entry("get_financials")],
        structure_gate_triggered=True, structure_gate_resolved=True,
        traceability_gate_triggered=True, traceability_gate_resolved=False,
        verification_mismatch_triggered=True, verification_mismatch_resolved=True,
    )
    gates = mod.gate_outcomes(result)
    assert gates["structure"] == "resolved"
    assert gates["traceability"] == "unresolved"
    assert gates["verification_mismatch"] == "resolved"
    assert gates["sentiment_consistency"] == "not_triggered"
    # 宣布完成但没做自我核查：信息不全就收尾
    assert mod.premature_stop(result) is True
    verified = _result([_entry("get_financials"), _entry("verify_number", metric="revenue", claimed_value=1.0, period="annual")])
    assert mod.premature_stop(verified) is False


def test_abnormal_input_must_disclose_and_not_invent_numbers():
    honest = _result([], final_report="<conclusion>未能获取 ZZZZ 的财务数据，该代码可能不存在</conclusion>", traceable_numbers_total=0)
    invented = _result([], final_report="<conclusion>未能获取部分数据，但营收同比增长20%</conclusion>")
    # 真实误判过：ZZZZ如实说明了拿不到数据，只是引用了板块数据里的数字，可追溯率低
    honest_with_sector_numbers = _result(
        [], final_report="<conclusion>无法获取 ZZZZ 的任何财务数据。科技板块 rs_ratio 100.20</conclusion>",
        traceable_numbers_matched=4, traceable_numbers_total=9,
    )
    assert mod.abnormal_input_handled(honest_with_sector_numbers) is True
    silent = _result([], final_report="<conclusion>前景良好</conclusion>", traceable_numbers_total=0)
    assert mod.abnormal_input_handled(honest) is True
    assert mod.abnormal_input_handled(invented) is False
    assert mod.abnormal_input_handled(silent) is False


def test_first_failing_stage_follows_the_chain_order():
    expected = {"AAPL": {"get_financials", "get_peer_comparison"}}
    ok = {"ticker": "AAPL", "scenario": "normal",
          "result": _result([_entry("get_financials"), _entry("get_peer_comparison"),
                             _entry("verify_number", turn=2, metric="revenue", claimed_value=416_161_000_000.0, period="annual")],
                            notes=[ReasoningNote(turn=1, text=REPORT)],
                            final_report=REPORT.replace("<evidence>e", "<evidence>营收4161.61亿美元")).model_dump()}
    assert mod.analyze_run(ok, expected)["first_failure"] is None

    missed_tool = {**ok, "result": _result([_entry("get_financials", metric="x")]).model_dump()}
    # 同时有工具漏调和参数问题时，按链路顺序先归到工具选择
    assert mod.analyze_run(missed_tool, expected)["first_failure"] == "工具选择"

    low_trace = {**ok, "result": {**ok["result"], "traceable_numbers_matched": 2, "traceable_numbers_total": 10}}
    assert mod.analyze_run(low_trace, expected)["first_failure"] == "结果解析"


def test_summary_aggregates_each_stage():
    expected = {"AAPL": {"get_financials"}}
    record = {"ticker": "AAPL", "scenario": "normal", "result": _result([_entry("get_financials")]).model_dump()}
    summary = mod.summarize([mod.analyze_run(record, expected)])
    assert summary["工具选择"]["recall_mean"] == 1.0
    assert summary["任务终止"]["completion_rate"] == 1.0
    assert summary["失败归因"] == {"任务终止": 1}  # 没做自我核查就收尾


def test_fault_injection_transient_fails_only_once_persistent_always():
    calls = []

    async def fake_tool(name, tool_input):
        calls.append(name)
        return "{}", False

    async def run():
        original = mod._original_execute_tool
        mod._original_execute_tool = fake_tool
        try:
            mod._fault.set({"kind": "transient_failure", "fired": False})
            first = await mod._execute_tool_with_faults("get_financials", {"ticker": "AAPL"})
            second = await mod._execute_tool_with_faults("get_financials", {"ticker": "AAPL"})
            mod._fault.set({"kind": "persistent_failure", "fired": False})
            peers = [await mod._execute_tool_with_faults("get_peer_comparison", {"ticker": "AAPL"}) for _ in range(2)]
        finally:
            mod._original_execute_tool = original
        return first, second, peers

    first, second, peers = asyncio.run(run())
    assert first[1] is True and second == ("{}", False)
    assert all(p[1] is True for p in peers)
    assert calls == ["get_financials"]  # 注入的失败不会真的去调数据源


def test_records_from_traces_keeps_only_real_normal_runs(tmp_path):
    real = {"type": "done", "result": _result([_entry("get_financials")], final_report="x" * 400).model_dump()}
    (tmp_path / "a.jsonl").write_text("\n".join(json.dumps(e) for e in [{"type": "llm_usage"}, real]))
    (tmp_path / "stub.jsonl").write_text(json.dumps({"type": "done", "result": {"final_report": "ok"}}))  # 测试桩
    (tmp_path / "bon.jsonl").write_text(json.dumps({"type": "llm_usage", "candidate_index": 0}))
    records = mod.records_from_traces(tmp_path)
    assert [r["source"] for r in records] == ["a.jsonl"]


def test_verification_not_covering_the_report_is_attributed_to_the_correction_stage():
    """方案B：不管先后顺序，管核实的对象是不是简报里真的写了的数字。"""
    expected = {"AAPL": {"get_financials"}}
    verify = _entry("verify_number", turn=1, metric="revenue", claimed_value=416_161_000_000.0, period="annual")
    covered = _result([_entry("get_financials"), verify], final_report=REPORT.replace("<evidence>e", "<evidence>营收4161.61亿美元"))
    uncovered = _result([_entry("get_financials"), verify], final_report=REPORT)
    run_ok = mod.analyze_run({"ticker": "AAPL", "scenario": "normal", "result": covered.model_dump()}, expected)
    run_bad = mod.analyze_run({"ticker": "AAPL", "scenario": "normal", "result": uncovered.model_dump()}, expected)
    assert run_ok["verification_covers_report"] is True and run_ok["first_failure"] is None
    assert run_bad["verification_covers_report"] is False and run_bad["first_failure"] == "纠错迭代"


def test_verified_values_include_actual_value_from_the_summary():
    entry = TranscriptEntry(turn=1, tool_name="verify_number", tool_input={"ticker": "AAPL", "metric": "revenue", "claimed_value": 1.0, "period": "annual"},
                            tool_output_summary='{"metric":"revenue","claimed_value":1.0,"actual_value":416161000000.0,"matches":false}', is_error=False)
    assert mod.verified_values(_result([entry])) == [1.0, 416161000000.0]


def test_abnormal_input_runs_are_judged_only_on_input_handling():
    """ZZZZ/SPY拿不到财务数据，可追溯率低、工具覆盖检查过不了都是必然的。"""
    honest = _result([_entry("get_financials", is_error=True)], final_report="<conclusion>无法获取ZZZZ的任何财务数据</conclusion>",
                     traceable_numbers_matched=1, traceable_numbers_total=9,
                     tool_coverage_gate_triggered=True, tool_coverage_gate_resolved=False)
    run = mod.analyze_run({"ticker": "ZZZZ", "scenario": "abnormal_input", "result": honest.model_dump()}, {})
    assert run["first_failure"] is None
