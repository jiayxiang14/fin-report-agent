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


def test_structure_full_marks_when_a_tag_is_missing_its_closing_tag():
    """模型偶尔漏写闭合标签（真实历史数据里实测约0.1%概率，比如写了<conclusion>
    却直接开始<evidence>而没有</conclusion>）——不该让整块内容被判定为"不存在"，
    退化成匹配到下一个已知标签为止。"""
    report = "<conclusion>a<evidence>b</evidence><flags>c</flags>"
    score = reward.score_rule_based(_make_run_result(report), [])
    assert score.structure == reward.STRUCTURE_MAX


def test_structure_recovers_content_even_when_two_consecutive_tags_are_unclosed():
    report = "<conclusion>a<evidence>b<flags>c</flags>"
    score = reward.score_rule_based(_make_run_result(report), [])
    assert score.structure == reward.STRUCTURE_MAX


def test_extract_tag_lenient_fallback_stops_at_sentiment_boundary():
    # flags漏关闭合标签时，兜底匹配不该把之后的<sentiment>内容也吞进去
    report = "<conclusion>a</conclusion><evidence>b</evidence><flags>c<sentiment>positive</sentiment>"
    assert reward._extract_tag(report, "flags") == "c"


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


def test_llm_judge_averages_scores_across_self_consistency_samples():
    """轻量自我一致性：裁判现在对同一个prompt并发采样JUDGE_SAMPLE_COUNT次取平均，
    不是只发一次请求。这里用一个调用计数器让每次返回不同的分数，验证最终结果
    确实是平均值，不是随便挑了某一次的结果。"""
    scores_by_call = iter([6, 7, 8, 9, 10])

    class FakeLLM:
        async def create_message(self, system, messages, tools, temperature=None):
            score = next(scores_by_call)
            return LLMResponse(
                stop_reason="end_turn", text=f"<reason>理由{score}</reason>\n<score>{score}</score>", tool_calls=[], raw=None
            )

    with patch("app.services.agent.reward.get_llm_client", return_value=FakeLLM()):
        score, reason = asyncio.run(reward.score_llm_judge("<conclusion>a</conclusion>"))

    assert reward.JUDGE_SAMPLE_COUNT == 5  # 断言依赖的采样次数没有被悄悄改掉
    assert score == 80.0  # (60+70+80+90+100)/5
    assert reason is not None


def test_llm_judge_ensemble_ignores_failed_samples_when_averaging():
    """3次采样里只要有一次解析失败（格式不对/调用报错），不该让整体退化成
    None——应该只用成功的那几次求平均，跟单次裁判"失败就整体降级"的哲学
    是两回事：这里"部分失败"不等于"全部失败"。"""
    call_count = {"n": 0}

    class FakeLLM:
        async def create_message(self, system, messages, tools, temperature=None):
            call_count["n"] += 1
            if call_count["n"] == 2:
                return LLMResponse(stop_reason="end_turn", text="没按格式回复", tool_calls=[], raw=None)
            return LLMResponse(
                stop_reason="end_turn", text="<reason>ok</reason>\n<score>8</score>", tool_calls=[], raw=None
            )

    with patch("app.services.agent.reward.get_llm_client", return_value=FakeLLM()):
        score, reason = asyncio.run(reward.score_llm_judge("<conclusion>a</conclusion>"))

    assert score == 80.0
    assert reason == "ok"


def test_llm_judge_ensemble_samples_use_nonzero_temperature():
    """自我一致性必须配非零温度，否则重复采样会得到几乎相同的单点结果，
    失去"降低方差"这个意义——验证确实把JUDGE_SAMPLE_TEMPERATURE传给了
    create_message，不是继续用之前单次裁判用的temperature=0.0。"""
    captured_temperatures: list[float | None] = []

    class FakeLLM:
        async def create_message(self, system, messages, tools, temperature=None):
            captured_temperatures.append(temperature)
            return LLMResponse(
                stop_reason="end_turn", text="<reason>ok</reason>\n<score>7</score>", tool_calls=[], raw=None
            )

    with patch("app.services.agent.reward.get_llm_client", return_value=FakeLLM()):
        asyncio.run(reward.score_llm_judge("<conclusion>a</conclusion>"))

    assert captured_temperatures == [reward.JUDGE_SAMPLE_TEMPERATURE] * reward.JUDGE_SAMPLE_COUNT


def test_judge_prompt_requires_reasoning_before_score():
    """G-Eval式"先推理后打分"：分数标签必须出现在理由标签之后，这样分数
    才是以模型自己刚写出的推理为条件生成的，不是先斩后奏的场面话。"""
    rendered = reward._JUDGE_PROMPT_TEMPLATE.format(report="x", anchors="")
    assert rendered.index("<reason>") < rendered.index("<score>")

    rendered_trajectory = reward._TRAJECTORY_JUDGE_PROMPT_TEMPLATE.format(trajectory="x")
    assert rendered_trajectory.index("<reason>") < rendered_trajectory.index("<score>")


def test_judge_score_anchors_are_sourced_from_the_human_labeled_eval_set():
    """锚点必须来自tests/fixtures/report_quality_eval_set.json里真实的人工标注
    结果（role=="anchor"的3条），不能是编造的/LLM生成的"示例"——那样起不到校准
    作用，等同于自己骗自己。这里核对锚点常量里的3段摘录，都能在标注文件里
    role=="anchor"的条目中找到对应的<conclusion>原文，防止手抄摘录时打字错误
    或者以后偷偷替换成非人工标注的内容而不被发现。"""
    import json
    import re
    from pathlib import Path

    eval_set_path = (
        Path(__file__).resolve().parent / "fixtures" / "report_quality_eval_set.json"
    )
    data = json.loads(eval_set_path.read_text())
    anchor_items = [item for item in data["items"] if item["role"] == "anchor"]

    assert len(anchor_items) == 3
    for item in anchor_items:
        match = re.search(r"<conclusion>([\s\S]*?)</conclusion>", item["final_report"])
        assert match is not None
        conclusion_text = match.group(1).strip()
        assert conclusion_text in reward._JUDGE_SCORE_ANCHORS


def test_judge_score_anchors_are_included_in_the_rendered_prompt():
    rendered = reward._JUDGE_PROMPT_TEMPLATE.format(report="x", anchors=reward._JUDGE_SCORE_ANCHORS)
    assert "参考示例" in rendered


def test_judge_uses_settings_judge_provider_when_configured(monkeypatch):
    """裁判可以配置成走跟生成不同的供应商（减少自评偏好嫌疑）——这里验证
    `settings.judge_llm_provider`确实被传给了`get_llm_client`，不是被忽略。"""
    monkeypatch.setattr("app.services.agent.reward.settings.judge_llm_provider", "claude")
    captured_provider = None

    class FakeLLM:
        async def create_message(self, system, messages, tools, temperature=None):
            return LLMResponse(
                stop_reason="end_turn", text="<score>7</score><reason>ok</reason>", tool_calls=[], raw=None
            )

    def fake_get_llm_client(provider=None):
        nonlocal captured_provider
        captured_provider = provider
        return FakeLLM()

    with patch("app.services.agent.reward.get_llm_client", new=fake_get_llm_client):
        asyncio.run(reward.score_llm_judge("<conclusion>a</conclusion>"))

    assert captured_provider == "claude"


def test_judge_uses_default_provider_when_not_configured(monkeypatch):
    monkeypatch.setattr("app.services.agent.reward.settings.judge_llm_provider", "")
    captured_provider = "not called"

    class FakeLLM:
        async def create_message(self, system, messages, tools, temperature=None):
            return LLMResponse(
                stop_reason="end_turn", text="<score>7</score><reason>ok</reason>", tool_calls=[], raw=None
            )

    def fake_get_llm_client(provider=None):
        nonlocal captured_provider
        captured_provider = provider
        return FakeLLM()

    with patch("app.services.agent.reward.get_llm_client", new=fake_get_llm_client):
        asyncio.run(reward.score_llm_judge("<conclusion>a</conclusion>"))

    assert captured_provider is None


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
