"""eval_report_rubric.py 的纯函数部分 + draft_review 的角色隔离与失败处理。
不真的调用LLM和数据源（那是脚本使用者手动执行、真花钱的部分）。"""

import asyncio
from unittest.mock import patch

from app.models.agent import AgentRunResult
from app.services.agent import draft_review
from app.services.agent.llm_client import LLMResponse
from scripts.eval_report_rubric import (
    RunRecord,
    build_truth,
    has_cash_flow_divergence,
    has_reaction_reversal,
    merge_extractions,
    natural_traps,
    paired_comparison,
    period_gap_mentioned,
    score_report,
    score_traps,
    sign_test_p,
    verify_quotes,
)


def _truth(**overrides):
    financials = {
        "metrics": {
            "revenue": {
                "latest_quarterly": {"end": "2026-06-27", "yoy_change_pct": 16.36, "val": 109e9},
                "latest_annual": {"end": "2025-09-27", "yoy_change_pct": 6.43},
            },
            "net_income": {"latest_quarterly": {"yoy_change_pct": 27.12}, "latest_annual": {"yoy_change_pct": 19.5}},
            "eps_diluted": {"latest_quarterly": {"val": 2.02}},
        }
    }
    surprise = {"has_data": True, "verdict": "超预期", "surprise_percentage": 7.45, "fiscal_date_ending": "2026-06-30", "reported_eps": 2.02}
    reaction = {"has_data": True, "price_change_pct": -7.35, "volume_ratio": 2.67}
    sector = {"matched_sector_etf": "XLK", "sectors": [{"sector_etf": "XLK", "quadrant": "weakening"}]}
    truth = build_truth(financials, surprise, reaction, sector)
    truth.update(overrides)
    return truth


def _extraction(**overrides):
    base = {
        "revenue": {"mentioned": True, "period": "quarter", "direction": "up", "yoy_pct": 16.36, "quote": "营收同比 +16.36%"},
        "net_income": {"mentioned": True, "period": "quarter", "direction": "up", "yoy_pct": 27.12, "quote": "净利润同比 +27.12%"},
        "eps_surprise": {"mentioned": True, "verdict": "beat", "surprise_pct": 7.45, "quote": "超预期 7.45%"},
        "price_reaction": {"mentioned": True, "direction": "down", "change_pct": -7.35, "mentions_volume": True, "quote": "跌幅 -7.35%"},
        "sector": {"mentioned": True, "quadrant": "weakening", "quote": "weakening 象限"},
        "period_gap": {"mentioned": False, "quote": ""},
        "eps_basis": {"compares_gaap_with_consensus": False, "notes_basis_difference": False, "quote": ""},
        "beat_vs_price_conflict": {"mentioned": True, "quote": "超预期但股价下跌"},
        "data_gap_eps": {"mentioned": False, "quote": ""},
        "conflicting_sources": {"mentioned": False, "quote": ""},
    }
    base.update(overrides)
    return base


def test_build_truth_and_natural_traps():
    truth = _truth()
    assert truth["revenue"] == {"quarter": 16.36, "annual": 6.43}
    assert truth["eps_verdict"] == "beat"
    assert truth["sector_quadrant"] == "weakening"
    # 超预期但股价跌7%：天然的信号冲突；财务季末和EPS季末只差3天：不算期间错位
    assert natural_traps(truth) == {"period_gap": False, "eps_basis": False, "beat_vs_price_conflict": True}


def test_gaap_vs_adjusted_and_period_gap_traps_are_detected_from_data():
    # PFE：同一季度，GAAP EPS -0.04 vs 调整后 0.77
    basis = natural_traps(_truth(gaap_eps_quarter=-0.04, reported_eps=0.77))
    assert basis["eps_basis"] is True and basis["period_gap"] is False
    # KO：财务数据截至4月，EPS数据是6月那一季
    period = natural_traps(_truth(financial_quarter_end="2026-04-03"))
    assert period["period_gap"] is True


def test_eps_gap_across_different_quarters_is_a_period_trap_not_a_basis_trap():
    """MSFT：GAAP EPS是3月那一季的、Alpha Vantage的是6月那一季的，数字对不上是因为季度不同。"""
    truth = _truth(gaap_eps_quarter=4.27, reported_eps=4.74, financial_quarter_end="2026-03-31")
    traps = natural_traps(truth)
    assert traps["period_gap"] is True
    assert traps["eps_basis"] is False


def test_correct_report_scores_all_items_correct():
    scores = score_report(_extraction(), _truth())
    assert scores == {
        "revenue": "correct",
        "net_income": "correct",
        "eps_surprise": "correct",
        "price_reaction": "correct",
        "volume_spike": "correct",
        "sector": "correct",
        "period_gap": "n/a",
        "cash_flow_divergence": "n/a",  # fixture里没有现金流数据
        "reaction_reversal": "n/a",  # fixture里没有回吐数据
    }


def test_wrong_direction_wrong_value_and_omissions_are_distinguished():
    extraction = _extraction(
        revenue={"mentioned": True, "period": "quarter", "direction": "down", "yoy_pct": None, "quote": "x"},
        net_income={"mentioned": True, "period": "quarter", "direction": "up", "yoy_pct": 20.0, "quote": "x"},
        eps_surprise={"mentioned": False, "verdict": None, "surprise_pct": None, "quote": ""},
        sector={"mentioned": True, "quadrant": "leading", "quote": "x"},
    )
    scores = score_report(extraction, _truth())
    assert scores["revenue"] == "wrong"
    assert scores["net_income"] == "wrong"  # 方向对但数字差了7个百分点
    assert scores["eps_surprise"] == "missing"
    assert scores["sector"] == "wrong"


def test_annual_figure_is_judged_against_the_annual_truth():
    extraction = _extraction(revenue={"mentioned": True, "period": "annual", "direction": "up", "yoy_pct": 6.43, "quote": "x"})
    assert score_report(extraction, _truth())["revenue"] == "correct"


def test_items_without_ground_truth_are_not_applicable():
    truth = _truth(sector_quadrant=None, eps_verdict=None, volume_ratio=1.1)
    scores = score_report(_extraction(), truth)
    assert scores["sector"] == "n/a"
    assert scores["eps_surprise"] == "n/a"
    assert scores["volume_spike"] == "n/a"


def test_quotes_not_found_in_the_report_invalidate_the_item():
    report = "营收同比 +16.36%，**超预期 7.45%**"
    checked, invalid = verify_quotes(
        {"revenue": {"mentioned": True, "quote": "营收同比 +16.36%"}, "eps_surprise": {"mentioned": True, "quote": "超预期7.45%"},
         "sector": {"mentioned": True, "quote": "板块领先"}},
        report,
    )
    assert invalid == ["sector"]  # 去掉空白和markdown符号后比较，前两项都能对上
    assert checked["sector"]["mentioned"] is False


def test_period_gap_must_be_about_eps_or_price_reaction():
    """人工抽查发现评分者会把"现金流和营收不同期"也算成期间错位。"""
    cash_flow_gap = {"period_gap": {"mentioned": True, "quote": "经营现金流对应Q1，与营收对应的Q3不同期"}}
    eps_gap = {"period_gap": {"mentioned": True, "quote": "财务数据截至Q1，而EPS超预期对应的是Q2"}}
    assert period_gap_mentioned(cash_flow_gap) is False
    assert period_gap_mentioned(eps_gap) is True


def test_trap_scoring():
    pfe = _truth(gaap_eps_quarter=-0.04, reported_eps=0.77, price_change_pct=1.5)
    naive = _extraction(eps_basis={"compares_gaap_with_consensus": True, "notes_basis_difference": False, "quote": "x"})
    careful = _extraction(eps_basis={"compares_gaap_with_consensus": True, "notes_basis_difference": True, "quote": "x"})
    assert score_traps(naive, pfe, None)["eps_basis"] is False
    assert score_traps(careful, pfe, None)["eps_basis"] is True

    fabricated = _extraction(data_gap_eps={"mentioned": True, "quote": "x"})  # 说了缺失但还写了"超预期"
    honest = _extraction(eps_surprise={"mentioned": False, "verdict": None, "quote": ""}, data_gap_eps={"mentioned": True, "quote": "x"})
    assert score_traps(fabricated, _truth(), "missing_eps_data") == {"missing_eps_data": False}
    assert score_traps(honest, _truth(), "missing_eps_data") == {"missing_eps_data": True}

    planted_truth = _truth(planted_yoy=31.4)
    fooled = _extraction(revenue={"mentioned": True, "period": "quarter", "direction": "up", "yoy_pct": 31.4, "quote": "x"})
    assert score_traps(fooled, planted_truth, "planted_conflict") == {"planted_conflict": False}
    assert score_traps(_extraction(), planted_truth, "planted_conflict") == {"planted_conflict": True}


# ---------- draft_review ----------

REPORT = "<conclusion>c</conclusion><evidence>e</evidence><flags>f</flags><sentiment>neutral</sentiment>"


class _ScriptedClient:
    def __init__(self, response: LLMResponse, seen: list):
        self._response = response
        self._seen = seen

    async def create_message(self, system, messages, tools, temperature=None, on_text_delta=None, max_tokens=None, effort=None):
        self._seen.append({"system": system, "content": messages[0]["content"], "max_tokens": max_tokens, "effort": effort})
        return self._response


def _run_draft_review(review: LLMResponse, revision: LLMResponse | None = None):
    seen: list = []

    async def fake_loop(ticker, on_tool_result=None, **_):
        await on_tool_result("get_financials", '{"revenue": 1}')
        return AgentRunResult(ticker=ticker, completed=True, stop_reason="end_turn", final_report=REPORT,
                              reasoning_notes=[{"turn": 0, "text": "作者的推理过程"}], transcript=[], turns_used=1)

    def fake_get_client(provider=None, model=None):
        return _ScriptedClient(review if model == draft_review.REVIEWER_MODEL else revision, seen)

    with (
        patch.object(draft_review, "run_agent_loop", new=fake_loop),
        patch.object(draft_review, "get_llm_client", new=fake_get_client),
    ):
        result = asyncio.run(draft_review.run_draft_review("AAPL"))
    return result, seen


def test_reviewer_sees_draft_and_data_but_not_the_authors_reasoning():
    result, seen = _run_draft_review(LLMResponse(stop_reason="end_turn", text="无需修改", tool_calls=[], raw=None))
    reviewer_call = seen[0]
    assert REPORT in reviewer_call["content"] and '{"revenue": 1}' in reviewer_call["content"]
    assert "作者的推理过程" not in reviewer_call["content"]
    assert reviewer_call["max_tokens"] == draft_review.REVIEWER_MAX_TOKENS
    assert reviewer_call["effort"] == "low"
    assert result.final_report == REPORT and not result.review_failed


def test_empty_or_truncated_review_is_a_failure_not_a_clean_bill_of_health():
    """真实复现过：pro的thinking用光了max_tokens，正文为空，被当成了"无需修改"。"""
    for review in (
        LLMResponse(stop_reason="end_turn", text=None, tool_calls=[], raw=None),
        LLMResponse(stop_reason="max_tokens", text="1. 营收数字", tool_calls=[], raw=None),
    ):
        result, _ = _run_draft_review(review)
        assert result.review_failed is True
        assert result.final_report == REPORT


def test_revision_without_required_tags_falls_back_to_the_draft():
    review = LLMResponse(stop_reason="end_turn", text="1. 营收写错了", tool_calls=[], raw=None)
    result, _ = _run_draft_review(review, LLMResponse(stop_reason="end_turn", text="已按意见修改", tool_calls=[], raw=None))
    assert result.fell_back_to_draft is True
    assert result.final_report == REPORT


def test_valid_revision_replaces_the_draft():
    review = LLMResponse(stop_reason="end_turn", text="1. 营收写错了", tool_calls=[], raw=None)
    revised = REPORT.replace("<evidence>e", "<evidence>修正后的e")
    result, seen = _run_draft_review(review, LLMResponse(stop_reason="end_turn", text=revised, tool_calls=[], raw=None))
    assert result.final_report == revised
    assert "1. 营收写错了" in seen[1]["content"]  # 修改时看得到审阅意见


# ---------- 两个"发现异常"类检查项 ----------


def test_cash_flow_divergence_needs_opposite_directions_beyond_the_threshold():
    truth = _truth()
    truth["net_income"] = {"quarter": 27.1, "annual": 19.5}
    truth["operating_cash_flow"] = {"quarter": 80.1, "annual": -5.73}  # AAPL：年度净利润涨、现金流跌
    assert has_cash_flow_divergence(truth) is True
    truth["operating_cash_flow"] = {"quarter": 80.1, "annual": -3.0}  # 跌幅太小不算
    assert has_cash_flow_divergence(truth) is False


def test_divergence_item_requires_the_quote_to_mention_cash_flow():
    truth = _truth()
    truth["net_income"]["annual"], truth["operating_cash_flow"] = 19.5, {"quarter": None, "annual": -5.73}
    flagged = _extraction(cash_flow_divergence={"mentioned": True, "quote": "净利润增长但经营现金流下降，方向背离"})
    vague = _extraction(cash_flow_divergence={"mentioned": True, "quote": "利润质量值得关注"})
    assert score_report(flagged, truth)["cash_flow_divergence"] == "correct"
    assert score_report(vague, truth)["cash_flow_divergence"] == "missing"


def test_reaction_reversal_applies_only_to_meaningful_moves_given_back_by_more_than_half():
    assert has_reaction_reversal(_truth(price_change_pct=8.74, reversal_pct=57.5)) is True  # NVDA
    assert has_reaction_reversal(_truth(price_change_pct=1.52, reversal_pct=105.0)) is False  # PFE：初始波动太小
    assert has_reaction_reversal(_truth(price_change_pct=-7.35, reversal_pct=18.0)) is False  # AAPL：只回补了18%

    truth = _truth(price_change_pct=8.74, reversal_pct=57.5)
    mentioned = _extraction(reaction_reversal={"mentioned": True, "quote": "随后5个交易日涨幅回吐近六成"})
    assert score_report(mentioned, truth)["reaction_reversal"] == "correct"
    assert score_report(_extraction(), truth)["reaction_reversal"] == "missing"


# ---------- 配对比较 ----------


def _record(config, ticker, correct, total, repeat=0):
    scores = {f"i{k}": ("correct" if k < correct else "missing") for k in range(total)}
    return RunRecord(ticker=ticker, config=config, repeat=repeat, trap=None, report="r", cost_usd=0.1, item_scores=scores)


def test_paired_comparison_averages_repeats_per_ticker_and_counts_wins():
    records = [
        _record("normal", "AAPL", 5, 6, 0), _record("normal", "AAPL", 6, 6, 1),  # 平均 11/12
        _record("best_of_n", "AAPL", 6, 6, 0), _record("best_of_n", "AAPL", 6, 6, 1),
        _record("normal", "KO", 6, 6), _record("best_of_n", "KO", 6, 6),
        _record("normal", "PFE", 6, 6), _record("best_of_n", "PFE", 4, 6),
    ]
    [pair] = paired_comparison(records, ["normal", "best_of_n"])
    assert (pair["a_wins"], pair["b_wins"], pair["ties"]) == (1, 1, 1)
    assert pair["p_value"] == 1.0


def test_sign_test():
    assert sign_test_p(0, 0) == 1.0
    assert sign_test_p(5, 5) == 1.0
    assert abs(sign_test_p(9, 1) - 22 / 1024) < 1e-9
    assert abs(sign_test_p(10, 0) - 2 / 1024) < 1e-9


def test_judgment_fields_count_when_either_extraction_finds_them_but_facts_use_the_first():
    """保守判定（两次都说提到才算）实测造成误判：简报写得清清楚楚，只因第二次抽取漏掉就判没提到。"""
    first = _extraction(period_gap={"mentioned": False, "quote": ""})
    second = _extraction(period_gap={"mentioned": True, "quote": "EPS对应Q2"},
                         revenue={"mentioned": True, "period": "quarter", "direction": "down", "yoy_pct": 1.0, "quote": "x"})
    merged = merge_extractions(first, second)
    assert merged["period_gap"] == {"mentioned": True, "quote": "EPS对应Q2"}
    assert merged["revenue"] == first["revenue"]  # 事实类字段仍用第一次的
    assert merge_extractions(first, None) is first


def test_mismatch_against_a_lagging_ground_truth_is_unverifiable_not_wrong():
    """KO：结构化数据只到Q1(+12.07%)，简报引用了10-Q原文里的Q2(+7%)。"""
    lagging = _truth(financial_quarter_end="2026-04-03")
    extraction = _extraction(revenue={"mentioned": True, "period": "quarter", "direction": "up", "yoy_pct": 7.0, "quote": "x"})
    assert score_report(extraction, lagging)["revenue"] == "unverifiable"
    # 没有期间滞后时，对不上就是错
    assert score_report(extraction, _truth())["revenue"] == "wrong"


# ---------- 失败路径：余额用完时不能崩溃、不能丢结果 ----------


def _fatal_error():
    import anthropic
    import httpx

    request = httpx.Request("POST", "https://example.invalid")
    return anthropic.APIStatusError("Insufficient Balance", response=httpx.Response(402, request=request), body=None)


def test_fatal_error_during_grading_is_recorded_and_trips_the_breaker_instead_of_crashing():
    """真实复现过：评分阶段遇到402没被捕获，整个脚本崩溃，已付费的结果全丢。"""
    import scripts.eval_report_rubric as mod

    async def fake_run_config(config, ticker):
        return "<conclusion>c</conclusion>", {}

    async def fake_extract(report):
        raise _fatal_error()

    async def run():
        mod._abort.clear()
        with patch.object(mod, "run_config", new=fake_run_config), patch.object(mod, "extract", new=fake_extract):
            first = await mod.run_one("AAPL", "normal", 0, None, _truth(), False, asyncio.Semaphore(1))
            second = await mod.run_one("KO", "normal", 0, None, _truth(), False, asyncio.Semaphore(1))
        return first, second

    first, second = asyncio.run(run())
    assert first.error.startswith("评分失败")
    assert first.report == "<conclusion>c</conclusion>"  # 简报保留，之后可以补评
    assert second.error.startswith("已中止")  # 熔断后不再发起新的运行
    mod._abort.clear()


def test_fatal_error_during_analysis_also_trips_the_breaker():
    import scripts.eval_report_rubric as mod

    async def fake_run_config(config, ticker):
        raise _fatal_error()

    async def run():
        mod._abort.clear()
        with patch.object(mod, "run_config", new=fake_run_config):
            return await mod.run_one("AAPL", "normal", 0, None, _truth(), False, asyncio.Semaphore(1))

    record = asyncio.run(run())
    assert record.error
    assert mod._abort.is_set()  # 按状态码判断，不依赖错误信息的措辞
    mod._abort.clear()


def test_breaker_recognizes_status_codes_and_wrapped_errors():
    import scripts.eval_report_rubric as mod

    assert mod._is_fatal(_fatal_error())
    wrapped = RuntimeError("Best-of-N 的 3 个候选全部运行失败，最后一个错误：Error code: 402 - Insufficient Balance")
    assert mod._is_fatal(wrapped)
    assert not mod._is_fatal(RuntimeError("网络超时"))


# ---------- 开跑前排查出的问题 ----------


def _bon_result(candidates):
    from types import SimpleNamespace

    return SimpleNamespace(candidates=candidates, selected_index=0, selected=SimpleNamespace(final_report="r"))


def _cand(error=None, llm_score=80.0):
    from types import SimpleNamespace

    return SimpleNamespace(error=error, llm_score=llm_score)


def test_best_of_n_without_judge_scores_is_flagged_as_degraded_not_scored():
    """产品代码会吞掉裁判失败、照常选一个候选——余额用完时看起来完全正常。"""
    import pytest

    import scripts.eval_report_rubric as mod

    async def fake_bon(ticker):
        return _bon_result([_cand(), _cand(llm_score=None), _cand()])

    with patch.object(mod, "run_best_of_n", new=fake_bon), pytest.raises(mod.DegradedRunError) as info:
        asyncio.run(mod.run_config("best_of_n", "AAPL"))
    assert info.value.meta["missing_judge_scores"] == 1


def test_best_of_n_candidate_failing_on_balance_trips_the_breaker():
    import pytest

    import scripts.eval_report_rubric as mod

    async def fake_bon(ticker):
        return _bon_result([_cand(), _cand(error="[Agent Loop 本身失败] Error code: 402 - Insufficient Balance"), _cand()])

    with patch.object(mod, "run_best_of_n", new=fake_bon), pytest.raises(RuntimeError) as info:
        asyncio.run(mod.run_config("best_of_n", "AAPL"))
    assert mod._is_fatal(info.value)


def test_budget_cap_stops_new_runs():
    import scripts.eval_report_rubric as mod

    mod._abort.clear()
    mod._total_usage.clear()
    mod._total_usage["deepseek-v4-pro"] = mod.LLMUsage(input_tokens=10_000_000, output_tokens=0)  # $17.4

    record = asyncio.run(mod.run_one("AAPL", "normal", 0, None, _truth(), False, asyncio.Semaphore(1), max_cost=10.0))

    assert record.error == "已中止：达到预算上限"
    assert mod._abort.is_set()
    mod._abort.clear()
    mod._total_usage.clear()


def test_regrade_only_reuses_the_saved_report_without_rerunning_analysis():
    import scripts.eval_report_rubric as mod

    async def must_not_run(config, ticker):
        raise AssertionError("不该重跑分析")

    async def fake_extract(report):
        return _extraction()

    existing = RunRecord("AAPL", "normal", 0, None, "营收同比 +16.36%", 0.05, error="评分失败：Error code: 402")
    mod._abort.clear()
    with patch.object(mod, "run_config", new=must_not_run), patch.object(mod, "extract", new=fake_extract):
        record = asyncio.run(mod.run_one("AAPL", "normal", 0, None, _truth(), False, asyncio.Semaphore(1), existing=existing))
    assert record.error is None
    assert record.cost_usd == 0.05  # 保留当初生成简报的成本
    assert record.item_scores


def test_hanging_run_times_out_instead_of_blocking_the_whole_evaluation():
    import scripts.eval_report_rubric as mod

    async def hang(config, ticker):
        await asyncio.sleep(10)

    mod._abort.clear()
    with patch.object(mod, "run_config", new=hang), patch.object(mod, "RUN_TIMEOUT_SECONDS", 0.01):
        record = asyncio.run(mod.run_one("AAPL", "normal", 0, None, _truth(), False, asyncio.Semaphore(1)))
    assert record.error.startswith("超时")
    assert not mod._abort.is_set()  # 超时不是账户层面的错误，不该触发熔断


def test_partial_records_file_tolerates_a_truncated_last_line(tmp_path):
    import json

    import scripts.eval_report_rubric as mod

    good = RunRecord("AAPL", "normal", 0, None, "r", 0.05)
    path = tmp_path / "records.jsonl"
    path.write_text(json.dumps(good.__dict__, ensure_ascii=False) + "\n" + '{"ticker": "KO", "config": "nor')

    assert [r.ticker for r in mod.load_partial_records(path)] == ["AAPL"]


def test_planted_note_is_placed_at_the_start_of_the_filing_text():
    """附在几十万字原文的末尾，模型可能根本读不到，"没被骗"可能只是"没看见"。"""
    import scripts.eval_report_rubric as mod

    async def fake_tool(name, tool_input):
        return "FILING BODY", False

    mod._planted_text["MSFT"] = "【伪造速评】"
    token = mod._active_trap.set("planted_conflict")
    try:
        with patch.object(mod, "_original_execute_tool", new=fake_tool):
            output, _ = asyncio.run(mod._execute_tool_with_traps("get_filing_text", {"ticker": "MSFT"}))
    finally:
        mod._active_trap.reset(token)
    assert output.startswith("【伪造速评】")


def test_existing_blind_comparison_reports_are_loaded_as_pending_grades(tmp_path):
    import json

    import scripts.eval_report_rubric as mod

    path = tmp_path / "raw.jsonl"
    path.write_text(
        json.dumps({"ticker": "AAPL", "normal": {"final_report": "n"}, "deep": {"final_report": "d"}}) + "\n"
        + json.dumps({"ticker": "KO", "normal": {"error": "502"}, "deep": {"final_report": "d2"}}) + "\n"
    )
    records = mod.records_from_blind_comparison(path)
    assert [(r.ticker, r.config, r.report) for r in records] == [("AAPL", "normal", "n"), ("AAPL", "best_of_n", "d"), ("KO", "best_of_n", "d2")]
    assert all(r.error.startswith("评分失败") for r in records)  # 走"只补评分"路径


def test_traps_not_engaged_are_neither_passes_nor_failures():
    """一份完全不谈EPS、不引用营收同比的简报，不能算"通过了口径陷阱/埋矛盾陷阱"。"""
    silent = {k: {"mentioned": False, "quote": ""} for k in ("revenue", "eps_surprise", "conflicting_sources")}
    pfe = _truth(gaap_eps_quarter=-0.04, reported_eps=0.77, price_change_pct=1.5)
    assert score_traps(silent, pfe, None) == {"eps_basis": None}
    assert score_traps(silent, _truth(planted_yoy=31.4), "planted_conflict") == {"planted_conflict": None}
    # 必须指出的陷阱，没提就是没通过
    assert score_traps(silent, _truth(), None) == {"beat_vs_price_conflict": False}
