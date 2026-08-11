"""自我核查兜底：之前完全靠 system prompt 要求"必须核查"，代码层面没有任何强制——
模型不调用 verify_number 也能正常 end_turn。现在 loop.py 在模型想结束、但全程
没调用过 verify_number 时，插入一条提示强制再走一轮，并且把这个信号通过
AgentRunResult.self_verification_triggered 暴露出去，不再只有 Best-of-N 内部
打分能看到。
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


def _tool_call_result(name: str, is_error: bool = False) -> tuple[str, bool]:
    return ("{}", is_error)


def test_nudge_injected_when_ending_without_ever_calling_verify_number():
    responses = [
        LLMResponse(stop_reason="end_turn", text="<conclusion>结论</conclusion>", tool_calls=[], raw=None),
        LLMResponse(stop_reason="end_turn", text="<conclusion>结论（修订）</conclusion>", tool_calls=[], raw=None),
    ]
    fake = FakeLLMClient(responses)
    with patch("app.services.agent.loop.get_llm_client", return_value=fake):
        result = asyncio.run(run_agent_loop("AAPL"))

    # 模型第一次想结束时被拦了一次，第二轮才是真正的收尾——发生了2次create_message调用
    assert fake.calls == 2
    assert result.completed is True
    assert result.self_verification_triggered is False
    assert result.final_report == "<conclusion>结论（修订）</conclusion>"


def test_no_nudge_when_verify_number_already_called():
    responses = [
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
        LLMResponse(stop_reason="end_turn", text="<conclusion>结论</conclusion>", tool_calls=[], raw=None),
    ]
    fake = FakeLLMClient(responses)
    with (
        patch("app.services.agent.loop.get_llm_client", return_value=fake),
        patch("app.services.agent.loop.execute_tool", new=AsyncMock(return_value=_tool_call_result("verify_number"))),
    ):
        result = asyncio.run(run_agent_loop("AAPL"))

    # 已经成功调用过 verify_number，不应该再被拦——只有2次create_message调用
    # （1次工具调用轮 + 1次收尾），而不是3次
    assert fake.calls == 2
    assert result.self_verification_triggered is True


def test_nudge_does_not_retrigger_if_model_still_skips_verification():
    """模型被拦了一次之后，如果依然不调用 verify_number 就再次end_turn，不应该
    无限重复插入提示——只拦一次，第二次直接放行，避免死循环吃满max_turns。"""
    responses = [
        LLMResponse(stop_reason="end_turn", text="<conclusion>结论</conclusion>", tool_calls=[], raw=None),
        LLMResponse(stop_reason="end_turn", text="<conclusion>结论</conclusion>", tool_calls=[], raw=None),
    ]
    fake = FakeLLMClient(responses)
    with patch("app.services.agent.loop.get_llm_client", return_value=fake):
        result = asyncio.run(run_agent_loop("AAPL"))

    assert fake.calls == 2  # 没有第三次调用，说明没有再次触发nudge
    assert result.completed is True
    assert result.self_verification_triggered is False


def test_no_nudge_when_stop_reason_is_not_end_turn():
    """refusal/max_tokens这类非正常结束，不该被自我核查兜底拦下来强行续写——
    拦下来也没有意义，模型本来就没打算正常给出结论。"""
    responses = [LLMResponse(stop_reason="max_tokens", text="部分内容", tool_calls=[], raw=None)]
    fake = FakeLLMClient(responses)
    with patch("app.services.agent.loop.get_llm_client", return_value=fake):
        result = asyncio.run(run_agent_loop("AAPL"))

    assert fake.calls == 1
    assert result.completed is False
    assert result.self_verification_triggered is False


def test_self_verification_triggered_reflects_transcript_on_max_turns_exceeded():
    """即使因为max_turns耗尽而未完成，self_verification_triggered也应该如实
    反映transcript里有没有verify_number记录，不是硬编码False。"""
    responses = [
        LLMResponse(
            stop_reason="tool_use",
            text=None,
            tool_calls=[
                ToolCall(
                    id=str(i),
                    name="verify_number",
                    input={"ticker": "AAPL", "metric": "revenue", "claimed_value": 1.0, "period": "annual"},
                )
            ],
            raw=None,
        )
        for i in range(8)
    ]
    fake = FakeLLMClient(responses)
    with (
        patch("app.services.agent.loop.get_llm_client", return_value=fake),
        patch("app.services.agent.loop.execute_tool", new=AsyncMock(return_value=_tool_call_result("verify_number"))),
    ):
        result = asyncio.run(run_agent_loop("AAPL", max_turns=8))

    assert result.stop_reason == "max_turns_exceeded"
    assert result.self_verification_triggered is True
