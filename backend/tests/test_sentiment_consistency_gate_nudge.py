"""sentiment与市场反应一致性gate：<sentiment>是模型对自己<conclusion>的主观
归纳，代码没法判断它"对不对"——但能检测一种具体、真实存在的矛盾：sentiment
标注positive，但get_price_reaction显示财报后股价明显下跌（或反过来），这是
两个独立信号打架，值得让模型自己面对、给出解释，不是让代码替它下判断，也
不该被装没看见。

fixture先安排get_financials+verify_number满足排在这道gate之前的几道gate，
report文本不含数字型断言（避免触发可追溯率gate），这个文件只关心sentiment
跟价格反应方向是否明显矛盾这一件事。
"""

import asyncio
import json
from unittest.mock import AsyncMock, patch

from app.services.agent.llm_client import LLMResponse, ToolCall
from app.services.agent.loop import run_agent_loop


class FakeLLMClient:
    def __init__(self, responses: list[LLMResponse]):
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


def _tool_use_turn(*, include_price_reaction: bool) -> LLMResponse:
    tool_calls = [
        ToolCall(id="0", name="get_financials", input={"ticker": "AAPL"}),
        ToolCall(
            id="1",
            name="verify_number",
            input={"ticker": "AAPL", "metric": "revenue", "claimed_value": 1.0, "period": "annual"},
        ),
    ]
    if include_price_reaction:
        tool_calls.append(ToolCall(id="2", name="get_price_reaction", input={"ticker": "AAPL"}))
    return LLMResponse(stop_reason="tool_use", text=None, tool_calls=tool_calls, raw=None)


def _report(conclusion: str, sentiment: str) -> str:
    return (
        f"<conclusion>{conclusion}</conclusion><evidence>业绩表现符合预期</evidence>"
        f"<flags>f</flags><sentiment>{sentiment}</sentiment>"
    )


def _make_execute_tool(price_change_pct: float | None):
    async def _fake(name: str, tool_input: dict) -> tuple[str, bool]:
        if name == "verify_number":
            return (json.dumps({"matches": True}), False)
        if name == "get_price_reaction":
            return (
                json.dumps({"ticker": "AAPL", "has_data": True, "price_change_pct": price_change_pct}),
                False,
            )
        return ("{}", False)

    return _fake


def test_nudge_when_positive_sentiment_contradicts_negative_price_reaction():
    """模型被拦下来之后，只改了<conclusion>的措辞（"已说明落差"），但
    <sentiment>依然是positive、且没有重新调用get_price_reaction——
    sentiment_consistency_gate_resolved是拿最终文本和raw_tool_outputs（不受
    messages影响的真实工具输出）重新核对一次矛盾，不会被"改了收尾措辞"这种
    表面动作骗过去，应该如实反映"矛盾其实还在"。"""
    responses = [
        _tool_use_turn(include_price_reaction=True),
        LLMResponse(stop_reason="end_turn", text=_report("结论", "positive"), tool_calls=[], raw=None),
        LLMResponse(stop_reason="end_turn", text=_report("结论（已说明落差）", "positive"), tool_calls=[], raw=None),
    ]
    fake = FakeLLMClient(responses)
    with (
        patch("app.services.agent.loop.get_llm_client", return_value=fake),
        patch("app.services.agent.loop.execute_tool", new=AsyncMock(side_effect=_make_execute_tool(-5.0))),
    ):
        result = asyncio.run(run_agent_loop("AAPL"))

    assert fake.calls == 3
    assert result.completed is True
    assert result.sentiment_consistency_gate_triggered is True
    assert result.sentiment_consistency_gate_resolved is False
    assert result.final_report == _report("结论（已说明落差）", "positive")


def test_resolved_is_true_when_revised_sentiment_no_longer_contradicts():
    """跟上一个测试对照：模型被拦下来之后真的把sentiment改成了跟价格反应
    方向一致的值，resolved应该如实反映"矛盾确实处理掉了"。"""
    responses = [
        _tool_use_turn(include_price_reaction=True),
        LLMResponse(stop_reason="end_turn", text=_report("结论", "positive"), tool_calls=[], raw=None),
        LLMResponse(stop_reason="end_turn", text=_report("结论（已重新评估）", "neutral"), tool_calls=[], raw=None),
    ]
    fake = FakeLLMClient(responses)
    with (
        patch("app.services.agent.loop.get_llm_client", return_value=fake),
        patch("app.services.agent.loop.execute_tool", new=AsyncMock(side_effect=_make_execute_tool(-5.0))),
    ):
        result = asyncio.run(run_agent_loop("AAPL"))

    assert result.sentiment_consistency_gate_triggered is True
    assert result.sentiment_consistency_gate_resolved is True


def test_nudge_when_negative_sentiment_contradicts_positive_price_reaction():
    responses = [
        _tool_use_turn(include_price_reaction=True),
        LLMResponse(stop_reason="end_turn", text=_report("结论", "negative"), tool_calls=[], raw=None),
        LLMResponse(stop_reason="end_turn", text=_report("结论（已说明落差）", "negative"), tool_calls=[], raw=None),
    ]
    fake = FakeLLMClient(responses)
    with (
        patch("app.services.agent.loop.get_llm_client", return_value=fake),
        patch("app.services.agent.loop.execute_tool", new=AsyncMock(side_effect=_make_execute_tool(5.0))),
    ):
        result = asyncio.run(run_agent_loop("AAPL"))

    assert fake.calls == 3
    assert result.sentiment_consistency_gate_triggered is True
    assert result.sentiment_consistency_gate_resolved is False


def test_no_nudge_when_sentiment_matches_price_reaction_direction():
    responses = [
        _tool_use_turn(include_price_reaction=True),
        LLMResponse(stop_reason="end_turn", text=_report("结论", "positive"), tool_calls=[], raw=None),
    ]
    fake = FakeLLMClient(responses)
    with (
        patch("app.services.agent.loop.get_llm_client", return_value=fake),
        patch("app.services.agent.loop.execute_tool", new=AsyncMock(side_effect=_make_execute_tool(5.0))),
    ):
        result = asyncio.run(run_agent_loop("AAPL"))

    assert fake.calls == 2
    assert result.sentiment_consistency_gate_triggered is False
    assert result.sentiment_consistency_gate_resolved is True  # 从没出过问题，vacuous true


def test_no_nudge_when_price_change_within_threshold():
    """跌幅没超过阈值(3%)，属于正常波动，不该被当成矛盾。"""
    responses = [
        _tool_use_turn(include_price_reaction=True),
        LLMResponse(stop_reason="end_turn", text=_report("结论", "positive"), tool_calls=[], raw=None),
    ]
    fake = FakeLLMClient(responses)
    with (
        patch("app.services.agent.loop.get_llm_client", return_value=fake),
        patch("app.services.agent.loop.execute_tool", new=AsyncMock(side_effect=_make_execute_tool(-1.0))),
    ):
        result = asyncio.run(run_agent_loop("AAPL"))

    assert fake.calls == 2
    assert result.sentiment_consistency_gate_triggered is False
    assert result.sentiment_consistency_gate_resolved is True


def test_no_nudge_when_no_price_reaction_data_available():
    """这次运行没有调用过get_price_reaction，没有独立数据源可交叉核对，
    不该强行拦截。"""
    responses = [
        _tool_use_turn(include_price_reaction=False),
        LLMResponse(stop_reason="end_turn", text=_report("结论", "positive"), tool_calls=[], raw=None),
    ]
    fake = FakeLLMClient(responses)
    with (
        patch("app.services.agent.loop.get_llm_client", return_value=fake),
        patch("app.services.agent.loop.execute_tool", new=AsyncMock(side_effect=_make_execute_tool(-5.0))),
    ):
        result = asyncio.run(run_agent_loop("AAPL"))

    assert fake.calls == 2
    assert result.sentiment_consistency_gate_triggered is False
    assert result.sentiment_consistency_gate_resolved is True


def test_nudge_does_not_retrigger_if_contradiction_persists():
    """拦了一次之后模型依然没处理这个矛盾，不该无限重复拦截——只拦一次，
    max_turns兜底。"""
    responses = [
        _tool_use_turn(include_price_reaction=True),
        LLMResponse(stop_reason="end_turn", text=_report("结论", "positive"), tool_calls=[], raw=None),
        LLMResponse(stop_reason="end_turn", text=_report("结论（仍未处理）", "positive"), tool_calls=[], raw=None),
    ]
    fake = FakeLLMClient(responses)
    with (
        patch("app.services.agent.loop.get_llm_client", return_value=fake),
        patch("app.services.agent.loop.execute_tool", new=AsyncMock(side_effect=_make_execute_tool(-5.0))),
    ):
        result = asyncio.run(run_agent_loop("AAPL"))

    assert fake.calls == 3  # 没有第四次调用
    assert result.completed is True
    assert result.sentiment_consistency_gate_triggered is True
    assert result.sentiment_consistency_gate_resolved is False
