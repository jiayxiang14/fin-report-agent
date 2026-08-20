"""工具使用底线gate：下结论前至少要成功调用过一次get_financials——之前完全
没有代码层面的强制，模型理论上可以一个工具都不查就直接下结论。只设这一个
底线，不强制其它工具的调用顺序或是否调用（CLAUDE.md核心原则2明确"不能写死
固定的工具调用顺序"），get_financials是唯一"简报里几乎不可能不引用、且几乎
不存在'刻意跳过'的合理理由"的工具。

fixture先安排verify_number调用满足排在这道gate之后的自我核查gate，report
文本都带全三个标签避免被结构合规gate截胡——这个文件只关心"查没查过基础
数据"这一件事。
"""

import asyncio
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


def _verify_number_call() -> ToolCall:
    return ToolCall(
        id="1",
        name="verify_number",
        input={"ticker": "AAPL", "metric": "revenue", "claimed_value": 1.0, "period": "annual"},
    )


def _financials_call() -> ToolCall:
    return ToolCall(id="0", name="get_financials", input={"ticker": "AAPL"})


def _full_report(conclusion: str) -> str:
    return f"<conclusion>{conclusion}</conclusion><evidence>e</evidence><flags>f</flags>"


def test_nudge_injected_when_get_financials_never_called():
    """模型被拦下来之后，只在文字里"嘴上说"已经补查了（"结论（已补查）"），
    但从始至终没有真的发起过get_financials这次工具调用——tool_coverage_gate
    _resolved 是从 transcript（真实工具调用记录，不是文本）重新核对的，不会
    被这种嘴上说说的文案骗过去，应该如实反映"其实没有真的补查"。"""
    responses = [
        LLMResponse(stop_reason="tool_use", text=None, tool_calls=[_verify_number_call()], raw=None),
        LLMResponse(stop_reason="end_turn", text=_full_report("结论"), tool_calls=[], raw=None),
        LLMResponse(stop_reason="end_turn", text=_full_report("结论（已补查）"), tool_calls=[], raw=None),
    ]
    fake = FakeLLMClient(responses)
    with (
        patch("app.services.agent.loop.get_llm_client", return_value=fake),
        patch("app.services.agent.loop.execute_tool", new=AsyncMock(return_value=("{}", False))),
    ):
        result = asyncio.run(run_agent_loop("AAPL"))

    assert fake.calls == 3
    assert result.completed is True
    assert result.tool_coverage_gate_triggered is True
    assert result.tool_coverage_gate_resolved is False
    assert result.final_report == _full_report("结论（已补查）")


def test_resolved_is_true_when_model_actually_calls_get_financials_after_nudge():
    """跟上一个测试对照：模型被拦下来之后真的发起了get_financials调用（不是
    只在文字里说说），resolved应该如实反映"确实补查了"。"""
    responses = [
        LLMResponse(stop_reason="tool_use", text=None, tool_calls=[_verify_number_call()], raw=None),
        LLMResponse(stop_reason="end_turn", text=_full_report("结论"), tool_calls=[], raw=None),
        LLMResponse(stop_reason="tool_use", text=None, tool_calls=[_financials_call()], raw=None),
        LLMResponse(stop_reason="end_turn", text=_full_report("结论（已补查）"), tool_calls=[], raw=None),
    ]
    fake = FakeLLMClient(responses)
    with (
        patch("app.services.agent.loop.get_llm_client", return_value=fake),
        patch("app.services.agent.loop.execute_tool", new=AsyncMock(return_value=("{}", False))),
    ):
        result = asyncio.run(run_agent_loop("AAPL"))

    assert result.tool_coverage_gate_triggered is True
    assert result.tool_coverage_gate_resolved is True


def test_no_nudge_when_get_financials_called():
    responses = [
        LLMResponse(
            stop_reason="tool_use", text=None, tool_calls=[_financials_call(), _verify_number_call()], raw=None
        ),
        LLMResponse(stop_reason="end_turn", text=_full_report("结论"), tool_calls=[], raw=None),
    ]
    fake = FakeLLMClient(responses)
    with (
        patch("app.services.agent.loop.get_llm_client", return_value=fake),
        patch("app.services.agent.loop.execute_tool", new=AsyncMock(return_value=("{}", False))),
    ):
        result = asyncio.run(run_agent_loop("AAPL"))

    assert fake.calls == 2
    assert result.tool_coverage_gate_triggered is False
    assert result.tool_coverage_gate_resolved is True  # 从没出过问题，vacuous true


def test_nudge_does_not_retrigger_if_still_missing_after_nudge():
    """拦了一次之后模型依然没有调用get_financials，不该无限重复拦截——只拦
    一次，max_turns兜底。"""
    responses = [
        LLMResponse(stop_reason="tool_use", text=None, tool_calls=[_verify_number_call()], raw=None),
        LLMResponse(stop_reason="end_turn", text=_full_report("结论"), tool_calls=[], raw=None),
        LLMResponse(stop_reason="end_turn", text=_full_report("结论（仍未查）"), tool_calls=[], raw=None),
    ]
    fake = FakeLLMClient(responses)
    with (
        patch("app.services.agent.loop.get_llm_client", return_value=fake),
        patch("app.services.agent.loop.execute_tool", new=AsyncMock(return_value=("{}", False))),
    ):
        result = asyncio.run(run_agent_loop("AAPL"))

    assert fake.calls == 3  # 没有第四次调用
    assert result.completed is True
    assert result.tool_coverage_gate_triggered is True
    assert result.tool_coverage_gate_resolved is False
