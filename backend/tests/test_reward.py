"""Best-of-N规则打分函数的单测：数字可追溯性/自我核查触发/三段式结构/长度，
以及LLM裁判打分的格式解析（含解析失败时的软降级）。全部用构造好的
AgentRunResult/工具原始输出直接测，不发真实网络请求。
"""

import asyncio
import json
from unittest.mock import patch

from app.models.agent import AgentRunResult, ReasoningNote, TranscriptEntry
from app.services.agent import reward
from app.services.agent.llm_client import LLMResponse


def _make_run_result(final_report: str | None, transcript: list[TranscriptEntry] | None = None) -> AgentRunResult:
    return AgentRunResult(
        ticker="AAPL",
        completed=True,
        stop_reason="end_turn",
        final_report=final_report,
        reasoning_notes=[],
        transcript=transcript or [],
        turns_used=1,
    )


def _verify_number_entry(is_error: bool = False) -> TranscriptEntry:
    return TranscriptEntry(
        turn=0,
        tool_name="verify_number",
        tool_input={"ticker": "AAPL", "metric": "revenues", "claimed_value": 1, "period": "annual"},
        tool_output_summary="{}",
        is_error=is_error,
    )


def test_traceability_full_marks_when_numbers_match_tool_output():
    # 故意不用"950亿"这种带中文单位倍数的写法——纯正则数字抽取不做单位换算，
    # 这是设计上承认的简化（见 reward.py 顶部注释），测试只验证"能抽取到的
    # 数字token确实会去核对"，不测单位换算
    report = "<conclusion>强劲</conclusion><evidence>营收同比增长12.5%，达到950,000,000美元</evidence><flags></flags>"
    run_result = _make_run_result(report)
    raw_outputs = [json.dumps({"revenue_growth_pct": 12.5, "revenue": 950000000})]

    score = reward.score_rule_based(run_result, raw_outputs)

    assert score.traceability_total == 2
    assert score.traceability_matched == 2
    assert score.traceability == reward.TRACEABILITY_MAX


def test_traceability_finds_numbers_embedded_in_filing_text_string_field():
    """真实复盘发现的漏判来源：get_filing_text返回的是一整段字符串塞在text
    字段里，之前_walk_json_numbers只收数值类型的叶子值，字符串里的数字完全
    不会被收进"已知数字"集合。这里验证修复后，简报引用的数字如果来源是财报
    原文叙述（而不是结构化字段），也能被正确追溯到。"""
    report = "<conclusion>强劲</conclusion><evidence>存货同比增长23.5%，管理层预计短期内NAND供应紧张会持续</evidence><flags></flags>"
    run_result = _make_run_result(report)
    raw_outputs = [
        json.dumps(
            {
                "ticker": "AMZN",
                "form": "10-Q",
                "text": "Inventory increased 23.5% year over year, driven by anticipated demand.",
            }
        )
    ]

    score = reward.score_rule_based(run_result, raw_outputs)

    assert score.traceability_matched == 1
    assert score.traceability == reward.TRACEABILITY_MAX


def test_traceability_partial_marks_when_a_number_cannot_be_traced():
    report = "<conclusion>强劲</conclusion><evidence>营收同比增长12.5%，净利润编造成了999亿美元</evidence><flags></flags>"
    run_result = _make_run_result(report)
    raw_outputs = [json.dumps({"revenue_growth_pct": 12.5})]

    score = reward.score_rule_based(run_result, raw_outputs)

    assert score.traceability_matched == 1
    assert score.traceability_total == 2
    assert score.traceability == reward.TRACEABILITY_MAX / 2


def test_traceability_full_marks_when_no_numeric_claims_present():
    report = "<conclusion>强劲</conclusion><evidence>管理层对前景表示乐观</evidence><flags></flags>"
    run_result = _make_run_result(report)

    score = reward.score_rule_based(run_result, [])

    assert score.traceability_total == 0
    assert score.traceability == reward.TRACEABILITY_MAX


def test_self_verification_scored_when_verify_number_called_successfully():
    report = "<conclusion>a</conclusion><evidence>b</evidence><flags>c</flags>"
    run_result = _make_run_result(report, transcript=[_verify_number_entry()])

    score = reward.score_rule_based(run_result, [])

    assert score.self_verification == reward.SELF_VERIFICATION_MAX


def test_self_verification_zero_when_never_called():
    report = "<conclusion>a</conclusion><evidence>b</evidence><flags>c</flags>"
    run_result = _make_run_result(report, transcript=[])

    score = reward.score_rule_based(run_result, [])

    assert score.self_verification == 0.0


def test_self_verification_zero_when_only_errored_call_exists():
    report = "<conclusion>a</conclusion><evidence>b</evidence><flags>c</flags>"
    run_result = _make_run_result(report, transcript=[_verify_number_entry(is_error=True)])

    score = reward.score_rule_based(run_result, [])

    assert score.self_verification == 0.0


def test_structure_full_marks_when_all_three_tags_present():
    report = "<conclusion>a</conclusion><evidence>b</evidence><flags>c</flags>"
    score = reward.score_rule_based(_make_run_result(report), [])
    assert score.structure == reward.STRUCTURE_MAX


def test_structure_partial_marks_when_a_tag_missing():
    report = "<conclusion>a</conclusion><evidence>b</evidence>"
    score = reward.score_rule_based(_make_run_result(report), [])
    assert round(score.structure, 2) == round(reward.STRUCTURE_MAX * 2 / 3, 2)


def test_structure_zero_when_no_final_report():
    score = reward.score_rule_based(_make_run_result(None), [])
    assert score.structure == 0.0


def test_length_full_marks_within_reasonable_band():
    report = "<conclusion>a</conclusion><evidence>" + "b" * 400 + "</evidence><flags>c</flags>"
    score = reward.score_rule_based(_make_run_result(report), [])
    assert score.length == reward.LENGTH_MAX


def test_length_half_marks_when_too_short():
    report = "<conclusion>短</conclusion>"
    score = reward.score_rule_based(_make_run_result(report), [])
    assert score.length == reward.LENGTH_MAX / 2


def test_length_zero_when_empty_report():
    score = reward.score_rule_based(_make_run_result(""), [])
    assert score.length == 0.0


def test_length_half_marks_when_too_long():
    report = "<conclusion>a</conclusion><evidence>" + "b" * 5000 + "</evidence><flags>c</flags>"
    score = reward.score_rule_based(_make_run_result(report), [])
    assert score.length == reward.LENGTH_MAX / 2


def test_llm_judge_parses_well_formed_response():
    class FakeLLM:
        async def create_message(self, system, messages, tools, temperature=None):
            return LLMResponse(
                stop_reason="end_turn",
                text="<score>8</score>\n<reason>逻辑连贯，数据支撑充分</reason>",
                tool_calls=[],
                raw=None,
            )

    with patch("app.services.agent.reward.get_llm_client", return_value=FakeLLM()):
        score, reason = asyncio.run(reward.score_llm_judge("<conclusion>a</conclusion>"))

    assert score == 80.0
    assert reason == "逻辑连贯，数据支撑充分"


def test_llm_judge_degrades_gracefully_when_response_not_tagged():
    class FakeLLM:
        async def create_message(self, system, messages, tools, temperature=None):
            return LLMResponse(stop_reason="end_turn", text="这份简报写得不错", tool_calls=[], raw=None)

    with patch("app.services.agent.reward.get_llm_client", return_value=FakeLLM()):
        score, reason = asyncio.run(reward.score_llm_judge("<conclusion>a</conclusion>"))

    assert score is None
    assert reason is None


def test_llm_judge_degrades_gracefully_when_call_raises():
    class FakeLLM:
        async def create_message(self, system, messages, tools, temperature=None):
            raise RuntimeError("上游挂了")

    with patch("app.services.agent.reward.get_llm_client", return_value=FakeLLM()):
        score, reason = asyncio.run(reward.score_llm_judge("<conclusion>a</conclusion>"))

    assert score is None
    assert reason is None


def test_llm_judge_skips_call_when_no_report():
    result = asyncio.run(reward.score_llm_judge(None))
    assert result == (None, None)


def test_trajectory_judge_parses_well_formed_response():
    class FakeLLM:
        async def create_message(self, system, messages, tools, temperature=None):
            return LLMResponse(
                stop_reason="end_turn",
                text="<score>7</score>\n<reason>信息收集充分，但没有验证异常数字</reason>",
                tool_calls=[],
                raw=None,
            )

    reasoning_notes = [ReasoningNote(turn=0, text="先查财务数据")]
    transcript = [
        TranscriptEntry(
            turn=0,
            tool_name="get_financials",
            tool_input={"ticker": "AAPL"},
            tool_output_summary="{}",
            is_error=False,
        )
    ]

    with patch("app.services.agent.reward.get_llm_client", return_value=FakeLLM()):
        score, reason = asyncio.run(reward.score_trajectory_judge(reasoning_notes, transcript))

    assert score == 70.0
    assert reason == "信息收集充分，但没有验证异常数字"


def test_trajectory_judge_skips_call_when_no_trajectory():
    result = asyncio.run(reward.score_trajectory_judge([], []))
    assert result == (None, None)


def test_trajectory_judge_degrades_gracefully_when_call_raises():
    class FakeLLM:
        async def create_message(self, system, messages, tools, temperature=None):
            raise RuntimeError("上游挂了")

    reasoning_notes = [ReasoningNote(turn=0, text="推理")]
    with patch("app.services.agent.reward.get_llm_client", return_value=FakeLLM()):
        score, reason = asyncio.run(reward.score_trajectory_judge(reasoning_notes, []))

    assert score is None
    assert reason is None


def test_render_trajectory_orders_steps_by_turn_and_interleaves_reasoning_with_tools():
    reasoning_notes = [ReasoningNote(turn=0, text="先查财务数据"), ReasoningNote(turn=1, text="结论已充分")]
    transcript = [
        TranscriptEntry(
            turn=0,
            tool_name="get_financials",
            tool_input={"ticker": "AAPL"},
            tool_output_summary="营收增长",
            is_error=False,
        )
    ]

    text = reward._render_trajectory(reasoning_notes, transcript)
    lines = text.splitlines()

    assert lines == [
        "[第0轮] 推理：先查财务数据",
        "[第0轮] 调用工具 get_financials（成功）：营收增长",
        "[第1轮] 推理：结论已充分",
    ]


def test_combine_scores_uses_rule_score_only_when_both_judges_missing():
    run_result = _make_run_result("<conclusion>a</conclusion><evidence>b</evidence><flags>c</flags>")
    rule_score = reward.score_rule_based(run_result, [])
    assert reward.combine_scores(rule_score, None, None) == rule_score.total


def test_combine_scores_blends_rule_and_both_judge_scores():
    run_result = _make_run_result("<conclusion>a</conclusion><evidence>b</evidence><flags>c</flags>")
    rule_score = reward.score_rule_based(run_result, [])
    combined = reward.combine_scores(rule_score, 90.0, 70.0)
    expected = round(
        rule_score.total * reward.RULE_WEIGHT + 90.0 * reward.OUTCOME_JUDGE_WEIGHT + 70.0 * reward.TRAJECTORY_JUDGE_WEIGHT,
        2,
    )
    assert combined == expected


def test_combine_scores_renormalizes_weights_when_only_trajectory_judge_missing():
    """结论裁判打成了分、过程裁判没打成——权重不该按"缺的那部分算0分"处理，
    应该在规则分和结论裁判分之间按比例重新分配，不能让总分被"没打成的裁判"
    悄悄拉低。"""
    run_result = _make_run_result("<conclusion>a</conclusion><evidence>b</evidence><flags>c</flags>")
    rule_score = reward.score_rule_based(run_result, [])
    combined = reward.combine_scores(rule_score, 90.0, None)
    total_weight = reward.RULE_WEIGHT + reward.OUTCOME_JUDGE_WEIGHT
    expected = round(
        (rule_score.total * reward.RULE_WEIGHT + 90.0 * reward.OUTCOME_JUDGE_WEIGHT) / total_weight, 2
    )
    assert combined == expected
