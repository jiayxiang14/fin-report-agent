"""三段式结构合规gate：<conclusion>本身已经是所有gate的触发前提，真正可能
缺失的是<evidence>/<flags>——之前完全没人管，模型只要在文字里塞了个
<conclusion>标签就能直接end_turn，前端会因为找不到<evidence>/<flags>而渲染
不完整。这里补一道纯正则、没有语义模糊空间的拦截：标签缺失就不放行，要求
重新完整输出一遍。

fixture先安排get_financials+verify_number满足排在结构合规gate之后的那几道
gate（工具使用底线/自我核查），这个文件只关心结构标签这一件事。
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


def _financials_and_verify_turn() -> LLMResponse:
    return LLMResponse(
        stop_reason="tool_use",
        text=None,
        tool_calls=[
            ToolCall(id="0", name="get_financials", input={"ticker": "AAPL"}),
            ToolCall(
                id="1",
                name="verify_number",
                input={"ticker": "AAPL", "metric": "revenue", "claimed_value": 1.0, "period": "annual"},
            ),
        ],
        raw=None,
    )


def test_nudge_injected_when_evidence_tag_missing():
    responses = [
        _financials_and_verify_turn(),
        LLMResponse(stop_reason="end_turn", text="<conclusion>结论</conclusion><flags>f</flags>", tool_calls=[], raw=None),
        LLMResponse(
            stop_reason="end_turn",
            text="<conclusion>结论（补全）</conclusion><evidence>e</evidence><flags>f</flags>",
            tool_calls=[],
            raw=None,
        ),
    ]
    fake = FakeLLMClient(responses)
    with (
        patch("app.services.agent.loop.get_llm_client", return_value=fake),
        patch("app.services.agent.loop.execute_tool", new=AsyncMock(return_value=("{}", False))),
    ):
        result = asyncio.run(run_agent_loop("AAPL"))

    assert fake.calls == 3
    assert result.completed is True
    assert result.structure_gate_triggered is True
    assert result.structure_gate_resolved is True
    assert result.final_report == "<conclusion>结论（补全）</conclusion><evidence>e</evidence><flags>f</flags>"


def test_nudge_injected_when_flags_tag_missing():
    responses = [
        _financials_and_verify_turn(),
        LLMResponse(
            stop_reason="end_turn", text="<conclusion>结论</conclusion><evidence>e</evidence>", tool_calls=[], raw=None
        ),
        LLMResponse(
            stop_reason="end_turn",
            text="<conclusion>结论（补全）</conclusion><evidence>e</evidence><flags>f</flags>",
            tool_calls=[],
            raw=None,
        ),
    ]
    fake = FakeLLMClient(responses)
    with (
        patch("app.services.agent.loop.get_llm_client", return_value=fake),
        patch("app.services.agent.loop.execute_tool", new=AsyncMock(return_value=("{}", False))),
    ):
        result = asyncio.run(run_agent_loop("AAPL"))

    assert fake.calls == 3
    assert result.structure_gate_triggered is True
    assert result.structure_gate_resolved is True


def test_no_nudge_when_all_tags_present_even_if_flags_is_empty():
    """<flags></flags>空标签本身是合法表述（"检查过、没有异常"），不该被当成
    缺失去拦截。"""
    responses = [
        _financials_and_verify_turn(),
        LLMResponse(
            stop_reason="end_turn",
            text="<conclusion>结论</conclusion><evidence>e</evidence><flags></flags>",
            tool_calls=[],
            raw=None,
        ),
    ]
    fake = FakeLLMClient(responses)
    with (
        patch("app.services.agent.loop.get_llm_client", return_value=fake),
        patch("app.services.agent.loop.execute_tool", new=AsyncMock(return_value=("{}", False))),
    ):
        result = asyncio.run(run_agent_loop("AAPL"))

    assert fake.calls == 2
    assert result.structure_gate_triggered is False
    assert result.structure_gate_resolved is True  # 从没出过问题，vacuous true


def test_nudge_does_not_retrigger_if_tags_still_missing_after_nudge():
    """拦了一次之后模型依然没补全标签，不该无限重复拦截——只拦一次，
    max_turns兜底。这里模型的第二次回应本身依然带着<conclusion>标签（只是
    还缺evidence），resolved应该如实反映"仍未解决"。"""
    responses = [
        _financials_and_verify_turn(),
        LLMResponse(stop_reason="end_turn", text="<conclusion>结论</conclusion><flags>f</flags>", tool_calls=[], raw=None),
        LLMResponse(
            stop_reason="end_turn", text="<conclusion>结论（仍缺）</conclusion><flags>f</flags>", tool_calls=[], raw=None
        ),
    ]
    fake = FakeLLMClient(responses)
    with (
        patch("app.services.agent.loop.get_llm_client", return_value=fake),
        patch("app.services.agent.loop.execute_tool", new=AsyncMock(return_value=("{}", False))),
    ):
        result = asyncio.run(run_agent_loop("AAPL"))

    assert fake.calls == 3  # 没有第四次调用
    assert result.completed is True
    assert result.structure_gate_triggered is True
    assert result.structure_gate_resolved is False


def test_resolved_is_false_when_model_ignores_both_the_batch_nudge_and_the_second_chance():
    """真实复现过的bug：gate拦下来要求"重新完整输出一遍"，模型没照做，只回
    一句不带任何标签的收尾话（比如"已修正，以上是最终版本"）。这种情况下
    `_resolve_final_report`的回退机制会往前找最近一条带<conclusion>标签的
    草稿顶上去——但那份草稿正是触发这道gate的、本来就缺标签的原始草稿。
    现在批量nudge之后还有一次"完全没配合"的二次机会（IGNORED_FORMAT_NUDGE），
    这里验证如果模型把两次机会都用一句收尾话敷衍过去，resolved依然能如实
    反映"没解决"，不会因为多给了一次机会就放松了对最终结果的核实。"""
    responses = [
        _financials_and_verify_turn(),
        LLMResponse(stop_reason="end_turn", text="<conclusion>结论</conclusion><evidence>e</evidence>", tool_calls=[], raw=None),
        LLMResponse(stop_reason="end_turn", text="已修正，以上是最终版本", tool_calls=[], raw=None),
        LLMResponse(stop_reason="end_turn", text="真的已经修正了", tool_calls=[], raw=None),
    ]
    fake = FakeLLMClient(responses)
    with (
        patch("app.services.agent.loop.get_llm_client", return_value=fake),
        patch("app.services.agent.loop.execute_tool", new=AsyncMock(return_value=("{}", False))),
    ):
        result = asyncio.run(run_agent_loop("AAPL"))

    assert fake.calls == 4
    assert result.completed is True
    assert result.structure_gate_triggered is True
    assert result.structure_gate_resolved is False
    # final_report被回退机制捞回的正是那份缺flags的旧草稿——这就是bug的真实表现，
    # 即便加了二次机会也无法强制模型合规，只能如实报告最终状态
    assert result.final_report == "<conclusion>结论</conclusion><evidence>e</evidence>"
    assert "<flags>" not in result.final_report


def test_second_chance_succeeds_when_model_finally_complies():
    """跟上一个测试对照：模型第一次批量nudge没配合（收尾话不带标签），但
    第二次机会（IGNORED_FORMAT_NUDGE）真的照做了，重新完整输出了带标签的
    简报——这时候应该正常收尾，resolved如实反映"确实解决了"。"""
    responses = [
        _financials_and_verify_turn(),
        LLMResponse(stop_reason="end_turn", text="<conclusion>结论</conclusion><evidence>e</evidence>", tool_calls=[], raw=None),
        LLMResponse(stop_reason="end_turn", text="已修正，以上是最终版本", tool_calls=[], raw=None),
        LLMResponse(
            stop_reason="end_turn",
            text="<conclusion>结论（真的补全了）</conclusion><evidence>e</evidence><flags>f</flags>",
            tool_calls=[],
            raw=None,
        ),
    ]
    fake = FakeLLMClient(responses)
    with (
        patch("app.services.agent.loop.get_llm_client", return_value=fake),
        patch("app.services.agent.loop.execute_tool", new=AsyncMock(return_value=("{}", False))),
    ):
        result = asyncio.run(run_agent_loop("AAPL"))

    assert fake.calls == 4
    assert result.completed is True
    assert result.structure_gate_triggered is True
    assert result.structure_gate_resolved is True
    assert "<flags>f</flags>" in result.final_report
