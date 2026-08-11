"""Best-of-N编排逻辑的回归测试：mock掉 run_agent_loop 和奖励打分函数，验证
（1）选中的确实是总分最高的候选，（2）事件按candidate_index正确打标，
（3）候选/得分/选择结果被追加写进JSONL日志文件。不发真实LLM调用。
"""

import asyncio
import json
from unittest.mock import AsyncMock, patch

import pytest

from app.models.agent import AgentRunResult
from app.models.best_of_n import RuleScoreBreakdown
from app.services.agent import best_of_n


def _run_result(ticker: str, report: str) -> AgentRunResult:
    return AgentRunResult(
        ticker=ticker,
        completed=True,
        stop_reason="end_turn",
        final_report=report,
        reasoning_notes=[],
        transcript=[],
        turns_used=1,
    )


def _rule_score(total: float) -> RuleScoreBreakdown:
    return RuleScoreBreakdown(
        traceability=total,
        traceability_matched=1,
        traceability_total=1,
        self_verification=0.0,
        structure=0.0,
        length=0.0,
        total=total,
    )


@pytest.fixture
def isolated_run_log(tmp_path, monkeypatch):
    log_path = tmp_path / "best_of_n_runs.jsonl"
    monkeypatch.setattr(best_of_n, "RUNS_LOG_PATH", log_path)
    return log_path


def test_selects_candidate_with_highest_total_score(isolated_run_log):
    run_results = [_run_result("AAPL", f"候选{i}") for i in range(3)]
    rule_scores = [_rule_score(40.0), _rule_score(90.0), _rule_score(60.0)]

    with (
        patch("app.services.agent.best_of_n.run_agent_loop", new=AsyncMock(side_effect=run_results)),
        patch("app.services.agent.best_of_n.reward.score_rule_based", side_effect=rule_scores),
        patch("app.services.agent.best_of_n.reward.score_llm_judge", new=AsyncMock(return_value=(None, None))),
        patch("app.services.agent.best_of_n.reward.score_trajectory_judge", new=AsyncMock(return_value=(None, None))),
        patch(
            "app.services.agent.best_of_n.reward.combine_scores",
            side_effect=lambda rule, outcome_llm, trajectory_llm: rule.total,
        ),
    ):
        result = asyncio.run(best_of_n.run_best_of_n("AAPL"))

    assert result.selected_index == 1
    assert result.selected.final_report == "候选1"
    assert [c.total_score for c in result.candidates] == [40.0, 90.0, 60.0]


def test_candidates_receive_distinct_temperatures():
    run_results = [_run_result("AAPL", f"候选{i}") for i in range(3)]
    captured_temperatures = []

    async def fake_run_agent_loop(ticker, temperature=None, on_event=None, on_tool_result=None, reflexion_check=None):
        captured_temperatures.append(temperature)
        return run_results[len(captured_temperatures) - 1]

    with (
        patch("app.services.agent.best_of_n.run_agent_loop", new=fake_run_agent_loop),
        patch(
            "app.services.agent.best_of_n.reward.score_rule_based",
            return_value=_rule_score(50.0),
        ),
        patch("app.services.agent.best_of_n.reward.score_llm_judge", new=AsyncMock(return_value=(None, None))),
        patch("app.services.agent.best_of_n.reward.score_trajectory_judge", new=AsyncMock(return_value=(None, None))),
        patch("app.services.agent.best_of_n.reward.combine_scores", return_value=50.0),
    ):
        asyncio.run(best_of_n.run_best_of_n("AAPL"))

    assert captured_temperatures == list(best_of_n.CANDIDATE_TEMPERATURES)


def test_events_are_tagged_with_candidate_index_and_scored_event_emitted():
    run_results = [_run_result("AAPL", f"候选{i}") for i in range(3)]
    events: list[dict] = []

    async def on_event(event: dict) -> None:
        events.append(event)

    async def fake_run_agent_loop(ticker, temperature=None, on_event=None, on_tool_result=None, reflexion_check=None):
        call_index = fake_run_agent_loop.calls
        fake_run_agent_loop.calls += 1
        await on_event({"type": "reasoning", "turn": 0, "text": "推理中"})
        return run_results[call_index]

    fake_run_agent_loop.calls = 0

    with (
        patch("app.services.agent.best_of_n.run_agent_loop", new=fake_run_agent_loop),
        patch(
            "app.services.agent.best_of_n.reward.score_rule_based",
            return_value=_rule_score(50.0),
        ),
        patch("app.services.agent.best_of_n.reward.score_llm_judge", new=AsyncMock(return_value=(70.0, "还行"))),
        patch(
            "app.services.agent.best_of_n.reward.score_trajectory_judge",
            new=AsyncMock(return_value=(65.0, "信息收集比较充分")),
        ),
        patch("app.services.agent.best_of_n.reward.combine_scores", return_value=58.0),
    ):
        asyncio.run(best_of_n.run_best_of_n("AAPL", on_event=on_event))

    reasoning_events = [e for e in events if e["type"] == "reasoning"]
    assert [e["candidate_index"] for e in reasoning_events] == [0, 1, 2]

    scored_events = [e for e in events if e["type"] == "candidate_scored"]
    assert [e["candidate_index"] for e in scored_events] == [0, 1, 2]
    assert scored_events[0]["total_score"] == 58.0
    assert scored_events[0]["llm_score"] == 70.0
    assert scored_events[0]["llm_reason"] == "还行"
    assert scored_events[0]["trajectory_score"] == 65.0
    assert scored_events[0]["trajectory_reason"] == "信息收集比较充分"


def test_a_failed_candidate_does_not_discard_already_completed_ones(isolated_run_log):
    """真实复盘发现的问题：候选1(temp=0.6)运行时上游报402(账户余额不足)，之前的代码
    会让整个run_best_of_n直接抛异常，连已经跑完、真花了钱的候选0一起被扔掉。现在
    候选1失败应该被跳过，候选0和候选2正常参与打分和选择。"""
    run_result_0 = _run_result("AAPL", "候选0")
    run_result_2 = _run_result("AAPL", "候选2")

    async def fake_run_agent_loop(ticker, temperature=None, on_event=None, on_tool_result=None, reflexion_check=None):
        call = fake_run_agent_loop.calls
        fake_run_agent_loop.calls += 1
        if call == 1:
            raise RuntimeError("Error code: 402 - Insufficient Balance")
        return run_result_0 if call == 0 else run_result_2

    fake_run_agent_loop.calls = 0

    with (
        patch("app.services.agent.best_of_n.run_agent_loop", new=fake_run_agent_loop),
        patch(
            "app.services.agent.best_of_n.reward.score_rule_based",
            side_effect=[_rule_score(40.0), _rule_score(90.0)],
        ),
        patch("app.services.agent.best_of_n.reward.score_llm_judge", new=AsyncMock(return_value=(None, None))),
        patch("app.services.agent.best_of_n.reward.score_trajectory_judge", new=AsyncMock(return_value=(None, None))),
        patch(
            "app.services.agent.best_of_n.reward.combine_scores",
            side_effect=lambda rule, outcome_llm, trajectory_llm: rule.total,
        ),
    ):
        result = asyncio.run(best_of_n.run_best_of_n("AAPL"))

    assert len(result.candidates) == 3
    assert result.candidates[1].error is not None
    assert "402" in result.candidates[1].error
    assert result.candidates[1].total_score is None
    assert result.candidates[1].rule_score is None
    # 候选0(40分)和候选2(90分)都成功，选中的应该是分数更高的候选2，
    # 不是"因为候选1失败就整体报错"或者"错误地选中候选0"
    assert result.selected_index == 2
    assert result.selected.final_report == "候选2"


def test_candidate_failed_event_emitted_and_other_candidates_still_run():
    events: list[dict] = []

    async def on_event(event: dict) -> None:
        events.append(event)

    run_results = {0: _run_result("AAPL", "候选0"), 2: _run_result("AAPL", "候选2")}

    async def fake_run_agent_loop(ticker, temperature=None, on_event=None, on_tool_result=None, reflexion_check=None):
        call = fake_run_agent_loop.calls
        fake_run_agent_loop.calls += 1
        if call == 1:
            raise RuntimeError("网络错误")
        return run_results[call]

    fake_run_agent_loop.calls = 0

    with (
        patch("app.services.agent.best_of_n.run_agent_loop", new=fake_run_agent_loop),
        patch("app.services.agent.best_of_n.reward.score_rule_based", return_value=_rule_score(50.0)),
        patch("app.services.agent.best_of_n.reward.score_llm_judge", new=AsyncMock(return_value=(None, None))),
        patch("app.services.agent.best_of_n.reward.score_trajectory_judge", new=AsyncMock(return_value=(None, None))),
        patch("app.services.agent.best_of_n.reward.combine_scores", return_value=50.0),
    ):
        asyncio.run(best_of_n.run_best_of_n("AAPL", on_event=on_event))

    failed_events = [e for e in events if e["type"] == "candidate_failed"]
    assert len(failed_events) == 1
    assert failed_events[0]["candidate_index"] == 1
    assert failed_events[0]["error"] == "网络错误"

    scored_events = [e for e in events if e["type"] == "candidate_scored"]
    assert [e["candidate_index"] for e in scored_events] == [0, 2]


def test_raises_best_of_n_error_when_all_candidates_fail():
    async def always_fails(ticker, temperature=None, on_event=None, on_tool_result=None, reflexion_check=None):
        raise RuntimeError("Error code: 402 - Insufficient Balance")

    with patch("app.services.agent.best_of_n.run_agent_loop", new=always_fails):
        with pytest.raises(best_of_n.BestOfNError, match="402"):
            asyncio.run(best_of_n.run_best_of_n("AAPL"))


def test_run_log_is_appended_with_ticker_and_selected_index(isolated_run_log):
    run_results = [_run_result("AAPL", f"候选{i}") for i in range(3)]
    rule_scores = [_rule_score(40.0), _rule_score(90.0), _rule_score(60.0)]

    with (
        patch("app.services.agent.best_of_n.run_agent_loop", new=AsyncMock(side_effect=run_results)),
        patch("app.services.agent.best_of_n.reward.score_rule_based", side_effect=rule_scores),
        patch("app.services.agent.best_of_n.reward.score_llm_judge", new=AsyncMock(return_value=(None, None))),
        patch("app.services.agent.best_of_n.reward.score_trajectory_judge", new=AsyncMock(return_value=(None, None))),
        patch(
            "app.services.agent.best_of_n.reward.combine_scores",
            side_effect=lambda rule, outcome_llm, trajectory_llm: rule.total,
        ),
    ):
        asyncio.run(best_of_n.run_best_of_n("AAPL"))

    lines = isolated_run_log.read_text().strip().splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["ticker"] == "AAPL"
    assert record["selected_index"] == 1
    assert len(record["candidates"]) == 3


def test_decide_reflexion_returns_none_when_score_meets_threshold():
    assert best_of_n._decide_reflexion(best_of_n.REFLEXION_SCORE_THRESHOLD, "过程扎实") is None


def test_decide_reflexion_returns_none_when_trajectory_judge_did_not_score():
    assert best_of_n._decide_reflexion(None, None) is None


def test_decide_reflexion_returns_critique_text_when_below_threshold():
    below_threshold = best_of_n.REFLEXION_SCORE_THRESHOLD - 1
    critique = best_of_n._decide_reflexion(below_threshold, "没有验证异常数字")
    assert critique is not None
    assert "没有验证异常数字" in critique


def test_make_reflexion_check_records_score_and_reason_into_cache():
    cache: dict = {}
    check = best_of_n._make_reflexion_check(cache)

    with patch(
        "app.services.agent.best_of_n.reward.score_trajectory_judge",
        new=AsyncMock(return_value=(55.0, "信息收集不充分")),
    ):
        critique = asyncio.run(check([], []))

    assert cache == {"computed": True, "score": 55.0, "reason": "信息收集不充分"}
    assert critique is not None
    assert "信息收集不充分" in critique


def test_run_best_of_n_wires_a_reflexion_check_into_run_agent_loop(isolated_run_log):
    run_results = [_run_result("AAPL", f"候选{i}") for i in range(3)]
    captured_kwargs = []

    async def fake_run_agent_loop(ticker, **kwargs):
        captured_kwargs.append(kwargs)
        return run_results[len(captured_kwargs) - 1]

    with (
        patch("app.services.agent.best_of_n.run_agent_loop", new=fake_run_agent_loop),
        patch("app.services.agent.best_of_n.reward.score_rule_based", return_value=_rule_score(50.0)),
        patch("app.services.agent.best_of_n.reward.score_llm_judge", new=AsyncMock(return_value=(None, None))),
        patch("app.services.agent.best_of_n.reward.score_trajectory_judge", new=AsyncMock(return_value=(None, None))),
        patch("app.services.agent.best_of_n.reward.combine_scores", return_value=50.0),
    ):
        asyncio.run(best_of_n.run_best_of_n("AAPL"))

    # 现在每个候选拿到的是各自独立的闭包实例（各自绑定自己的缓存字典），不再是
    # 同一个模块级函数，所以不能用身份比较——改成验证每一个都是可调用的，并且
    # 真的会去调用过程裁判（表现符合reflexion_check的职责）
    assert len(captured_kwargs) == 3
    for kwargs in captured_kwargs:
        check = kwargs.get("reflexion_check")
        assert callable(check)
        with patch(
            "app.services.agent.best_of_n.reward.score_trajectory_judge",
            new=AsyncMock(return_value=(None, None)),
        ):
            asyncio.run(check([], []))


class _FakeLLMClientForCandidate:
    """跟test_reflexion_nudge.py是同一个假客户端模式，这里用真实的run_agent_loop
    （不mock）去跑_run_candidate，才能验证reflexion_check闭包真的会在Loop内部
    被调用、trajectory_cache真的被填过——如果像其它测试那样直接mock掉
    run_agent_loop，闭包根本不会被执行，测不出"避免重复调用"这件事本身。
    """

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = 0

    async def create_message(self, system, messages, tools, temperature=None):
        response = self._responses[self.calls]
        self.calls += 1
        return response

    def append_assistant_turn(self, messages, response):
        return [*messages, {"role": "assistant", "content": "fake"}]

    def append_tool_results(self, messages, tool_calls, results):
        return [*messages, {"role": "user", "content": "fake"}]


def _verified_end_turn(text: str) -> list:
    """先核实一个数字满足自我核查底线，再end_turn——避免自我核查兜底截胡了
    第一次end_turn机会，干扰对reflexion这条链路的观察。"""
    from app.services.agent.llm_client import LLMResponse, ToolCall

    return [
        LLMResponse(
            stop_reason="tool_use",
            text=None,
            tool_calls=[
                ToolCall(
                    id="1",
                    name="verify_number",
                    input={"ticker": "AAPL", "metric": "revenue", "claimed_value": 1.0, "period": "annual"},
                )
            ],
            raw=None,
        ),
        LLMResponse(stop_reason="end_turn", text=f"<conclusion>{text}</conclusion>", tool_calls=[], raw=None),
    ]


def test_run_candidate_reuses_cached_trajectory_score_when_reflexion_not_triggered():
    """过程裁判分数够高、不触发整改时，_run_candidate不该在run_agent_loop返回后
    再原样问一遍过程裁判——之前这里是重复调用两次同一个问题。"""
    responses = _verified_end_turn("结论")
    fake_llm = _FakeLLMClientForCandidate(responses)
    call_count = 0

    async def fake_score_trajectory_judge(reasoning_notes, transcript):
        nonlocal call_count
        call_count += 1
        return best_of_n.REFLEXION_SCORE_THRESHOLD, "过程扎实"  # 够格，不触发整改

    with (
        patch("app.services.agent.loop.get_llm_client", return_value=fake_llm),
        patch("app.services.agent.loop.execute_tool", new=AsyncMock(return_value=("{}", False))),
        patch("app.services.agent.best_of_n.reward.score_trajectory_judge", new=fake_score_trajectory_judge),
        patch("app.services.agent.best_of_n.reward.score_llm_judge", new=AsyncMock(return_value=(None, None))),
    ):
        summary, run_result = asyncio.run(best_of_n._run_candidate("AAPL", 0, 0.3, on_event=None))

    assert call_count == 1  # 只在loop内部检查时问过一次，收尾后没有再重复问
    assert run_result.reflexion_triggered is False
    assert summary.trajectory_score == best_of_n.REFLEXION_SCORE_THRESHOLD
    assert summary.trajectory_reason == "过程扎实"


def test_run_candidate_recomputes_trajectory_score_when_reflexion_triggered():
    """触发过整改的话，收尾时的transcript/reasoning_notes已经跟检查那一刻不一样
    了（多了一轮整改），必须重新问一遍过程裁判才能拿到准确的最终打分。"""
    from app.services.agent.llm_client import LLMResponse

    responses = [
        *_verified_end_turn("结论"),
        LLMResponse(stop_reason="end_turn", text="<conclusion>结论（已整改）</conclusion>", tool_calls=[], raw=None),
    ]
    fake_llm = _FakeLLMClientForCandidate(responses)
    scores = [(best_of_n.REFLEXION_SCORE_THRESHOLD - 1, "信息不足"), (90.0, "已改进")]
    call_count = 0

    async def fake_score_trajectory_judge(reasoning_notes, transcript):
        nonlocal call_count
        score, reason = scores[call_count]
        call_count += 1
        return score, reason

    with (
        patch("app.services.agent.loop.get_llm_client", return_value=fake_llm),
        patch("app.services.agent.loop.execute_tool", new=AsyncMock(return_value=("{}", False))),
        patch("app.services.agent.best_of_n.reward.score_trajectory_judge", new=fake_score_trajectory_judge),
        patch("app.services.agent.best_of_n.reward.score_llm_judge", new=AsyncMock(return_value=(None, None))),
    ):
        summary, run_result = asyncio.run(best_of_n._run_candidate("AAPL", 0, 0.3, on_event=None))

    assert call_count == 2  # loop内部检查一次(不及格触发整改) + 收尾后重新打分一次
    assert run_result.reflexion_triggered is True
    assert summary.trajectory_score == 90.0
    assert summary.trajectory_reason == "已改进"


def test_reflexion_triggered_propagates_to_candidate_summary_and_scored_event(isolated_run_log):
    reflexion_run_result = AgentRunResult(
        ticker="AAPL",
        completed=True,
        stop_reason="end_turn",
        final_report="候选0（已整改）",
        reasoning_notes=[],
        transcript=[],
        turns_used=2,
        reflexion_triggered=True,
    )
    other_run_results = [_run_result("AAPL", f"候选{i}") for i in (1, 2)]
    events: list[dict] = []

    async def on_event(event: dict) -> None:
        events.append(event)

    with (
        patch(
            "app.services.agent.best_of_n.run_agent_loop",
            new=AsyncMock(side_effect=[reflexion_run_result, *other_run_results]),
        ),
        patch("app.services.agent.best_of_n.reward.score_rule_based", return_value=_rule_score(50.0)),
        patch("app.services.agent.best_of_n.reward.score_llm_judge", new=AsyncMock(return_value=(None, None))),
        patch("app.services.agent.best_of_n.reward.score_trajectory_judge", new=AsyncMock(return_value=(None, None))),
        patch("app.services.agent.best_of_n.reward.combine_scores", return_value=50.0),
    ):
        result = asyncio.run(best_of_n.run_best_of_n("AAPL", on_event=on_event))

    assert result.candidates[0].reflexion_triggered is True
    assert result.candidates[1].reflexion_triggered is False

    scored_events = [e for e in events if e["type"] == "candidate_scored"]
    assert scored_events[0]["reflexion_triggered"] is True
    assert scored_events[1]["reflexion_triggered"] is False
