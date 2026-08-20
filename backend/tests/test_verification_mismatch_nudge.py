"""自我核查结果不一致兜底：`_has_self_verification` 只检查"调用过 verify_number
且没技术性报错"，不检查调用结果本身——verify_number 会返回真正客观、代码算出来
的 matches 字段，之前代码完全不读它，模型哪怕亲眼看到 matches=false 也能照样
end_turn，self_verification_triggered 还照样标记成 True。"核查过了"变成了摆设，
真正拍板"这份报告能不能定稿"的还是模型那一刻的自我判断，不是一个真会拒绝放行
的客观 gate。这里补上第二道拦截：模型最近一次成功的 verify_number 调用如果
matches=false 且没被后续调用推翻，就不放行，强制再走一轮去修正或如实标注。

fixture里第一轮都带上get_financials调用、report文本都带全三个标签——不这样
做的话会先被结构合规gate/工具使用底线gate拦下来，这个文件只关心核查结果
不一致这一件事。
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


def _verify_number_call(call_id: str = "1") -> ToolCall:
    return ToolCall(
        id=call_id,
        name="verify_number",
        input={"ticker": "AAPL", "metric": "revenue", "claimed_value": 1.0, "period": "annual"},
    )


def _financials_call(call_id: str = "0") -> ToolCall:
    return ToolCall(id=call_id, name="get_financials", input={"ticker": "AAPL"})


def _verify_result(matches: bool | None) -> tuple[str, bool]:
    # period是VerifyNumberResponse的必填字段（真实响应里永远有值），mock也要
    # 带上，不然loop.py现在按(metric, period)分组时会因为period缺失把这次
    # 结果整条丢弃，测试意图就落空了
    return (json.dumps({"ticker": "AAPL", "metric": "revenue", "period": "annual", "matches": matches}), False)


def _full_report(conclusion: str) -> str:
    return f"<conclusion>{conclusion}</conclusion><evidence>e</evidence><flags>f</flags>"


def _execute_tool_returning(matches: bool | None):
    async def _fake(name: str, tool_input: dict) -> tuple[str, bool]:
        if name == "verify_number":
            return _verify_result(matches)
        return ("{}", False)

    return _fake


def test_nudge_injected_when_last_verification_did_not_match():
    """模型被拦下来之后，只在文字里"嘴上说"已经修正了（"结论（已修正）"），
    但从始至终没有真的重新调用过verify_number——verification_mismatch_resolved
    是从verify_number_outcomes（真实工具调用返回的matches字段，不是文本）
    重新核对的，不会被这种嘴上说说的文案骗过去，应该如实反映"其实没有真的
    重新核实过"。"""
    responses = [
        LLMResponse(
            stop_reason="tool_use", text=None, tool_calls=[_financials_call(), _verify_number_call()], raw=None
        ),
        LLMResponse(stop_reason="end_turn", text=_full_report("结论"), tool_calls=[], raw=None),
        LLMResponse(stop_reason="end_turn", text=_full_report("结论（已修正）"), tool_calls=[], raw=None),
    ]
    fake = FakeLLMClient(responses)
    with (
        patch("app.services.agent.loop.get_llm_client", return_value=fake),
        patch("app.services.agent.loop.execute_tool", new=AsyncMock(side_effect=_execute_tool_returning(False))),
    ):
        result = asyncio.run(run_agent_loop("AAPL"))

    # 工具调用轮 + 想结束但被拦下 + 真正收尾，一共3次create_message调用
    assert fake.calls == 3
    assert result.completed is True
    assert result.self_verification_triggered is True
    assert result.verification_mismatch_triggered is True
    assert result.verification_mismatch_resolved is False
    assert result.final_report == _full_report("结论（已修正）")


def test_resolved_is_true_when_model_actually_reverifies_after_nudge():
    """跟上一个测试对照：模型被拦下来之后真的重新调用了verify_number（不是
    只在文字里说说），这次matches=true，resolved应该如实反映"确实核实通过"。"""
    responses = [
        LLMResponse(
            stop_reason="tool_use", text=None, tool_calls=[_financials_call(), _verify_number_call("1")], raw=None
        ),
        LLMResponse(stop_reason="end_turn", text=_full_report("结论"), tool_calls=[], raw=None),
        LLMResponse(stop_reason="tool_use", text=None, tool_calls=[_verify_number_call("2")], raw=None),
        LLMResponse(stop_reason="end_turn", text=_full_report("结论（已修正）"), tool_calls=[], raw=None),
    ]
    fake = FakeLLMClient(responses)
    verify_call_count = 0

    async def _sequenced_execute_tool(name: str, tool_input: dict) -> tuple[str, bool]:
        nonlocal verify_call_count
        if name == "verify_number":
            verify_call_count += 1
            return _verify_result(False if verify_call_count == 1 else True)
        return ("{}", False)

    with (
        patch("app.services.agent.loop.get_llm_client", return_value=fake),
        patch("app.services.agent.loop.execute_tool", new=AsyncMock(side_effect=_sequenced_execute_tool)),
    ):
        result = asyncio.run(run_agent_loop("AAPL"))

    assert result.verification_mismatch_triggered is True
    assert result.verification_mismatch_resolved is True


def test_no_nudge_when_verification_matches():
    responses = [
        LLMResponse(
            stop_reason="tool_use", text=None, tool_calls=[_financials_call(), _verify_number_call()], raw=None
        ),
        LLMResponse(stop_reason="end_turn", text=_full_report("结论"), tool_calls=[], raw=None),
    ]
    fake = FakeLLMClient(responses)
    with (
        patch("app.services.agent.loop.get_llm_client", return_value=fake),
        patch("app.services.agent.loop.execute_tool", new=AsyncMock(side_effect=_execute_tool_returning(True))),
    ):
        result = asyncio.run(run_agent_loop("AAPL"))

    assert fake.calls == 2
    assert result.verification_mismatch_triggered is False
    assert result.verification_mismatch_resolved is True  # 从没出过问题，vacuous true


def test_no_nudge_when_matches_is_none():
    """matches=None代表"指标不存在/数据缺失"，不是"不匹配"，不该触发这道拦截。"""
    responses = [
        LLMResponse(
            stop_reason="tool_use", text=None, tool_calls=[_financials_call(), _verify_number_call()], raw=None
        ),
        LLMResponse(stop_reason="end_turn", text=_full_report("结论"), tool_calls=[], raw=None),
    ]
    fake = FakeLLMClient(responses)
    with (
        patch("app.services.agent.loop.get_llm_client", return_value=fake),
        patch("app.services.agent.loop.execute_tool", new=AsyncMock(side_effect=_execute_tool_returning(None))),
    ):
        result = asyncio.run(run_agent_loop("AAPL"))

    assert fake.calls == 2
    assert result.verification_mismatch_triggered is False
    assert result.verification_mismatch_resolved is True


def test_later_matching_verification_clears_an_earlier_mismatch():
    """模型先核实一个数字没对上，改完之后又重新核实过并且这次对上了——最近一次
    的结果才算数，不该被更早那次失败的核查拖着不放。"""
    responses = [
        LLMResponse(
            stop_reason="tool_use", text=None, tool_calls=[_financials_call(), _verify_number_call("1")], raw=None
        ),
        LLMResponse(stop_reason="tool_use", text=None, tool_calls=[_verify_number_call("2")], raw=None),
        LLMResponse(stop_reason="end_turn", text=_full_report("结论"), tool_calls=[], raw=None),
    ]
    fake = FakeLLMClient(responses)
    verify_call_count = 0

    async def _sequenced_execute_tool(name: str, tool_input: dict) -> tuple[str, bool]:
        nonlocal verify_call_count
        if name == "verify_number":
            verify_call_count += 1
            return _verify_result(False if verify_call_count == 1 else True)
        return ("{}", False)

    with (
        patch("app.services.agent.loop.get_llm_client", return_value=fake),
        patch("app.services.agent.loop.execute_tool", new=AsyncMock(side_effect=_sequenced_execute_tool)),
    ):
        result = asyncio.run(run_agent_loop("AAPL"))

    assert fake.calls == 3
    assert result.verification_mismatch_triggered is False
    assert result.verification_mismatch_resolved is True


def test_mismatch_on_one_metric_is_not_masked_by_a_different_metric_matching():
    """真实复现过的bug：模型核实营收发现不对（没去改），紧接着核实净利润，
    这次对上了——旧逻辑只看verify_number_outcomes整个列表的最后一个元素，
    净利润这次不相关的成功核查会把营收那个明确有问题、却从没被处理的
    mismatch"顺手"洗白掉。营收和净利润是两个独立的(metric, period)，各自
    的最新结果都要单独判断，不能互相掩盖。"""
    revenue_call = ToolCall(
        id="1", name="verify_number", input={"ticker": "AAPL", "metric": "revenue", "claimed_value": 1.0, "period": "annual"}
    )
    net_income_call = ToolCall(
        id="2",
        name="verify_number",
        input={"ticker": "AAPL", "metric": "net_income", "claimed_value": 2.0, "period": "annual"},
    )
    responses = [
        LLMResponse(
            stop_reason="tool_use", text=None, tool_calls=[_financials_call(), revenue_call, net_income_call], raw=None
        ),
        LLMResponse(stop_reason="end_turn", text=_full_report("结论"), tool_calls=[], raw=None),
        LLMResponse(stop_reason="end_turn", text=_full_report("结论（已修正）"), tool_calls=[], raw=None),
    ]
    fake = FakeLLMClient(responses)

    async def _fake_execute_tool(name: str, tool_input: dict) -> tuple[str, bool]:
        if name == "verify_number":
            matches = False if tool_input["metric"] == "revenue" else True
            return (
                json.dumps({"ticker": "AAPL", "metric": tool_input["metric"], "period": tool_input["period"], "matches": matches}),
                False,
            )
        return ("{}", False)

    with (
        patch("app.services.agent.loop.get_llm_client", return_value=fake),
        patch("app.services.agent.loop.execute_tool", new=AsyncMock(side_effect=_fake_execute_tool)),
    ):
        result = asyncio.run(run_agent_loop("AAPL"))

    assert result.verification_mismatch_triggered is True
    assert result.verification_mismatch_resolved is False


def test_nudge_does_not_retrigger_if_model_still_ignores_mismatch():
    """拦了一次之后模型依然不处理就直接end_turn，不该无限重复拦截——只拦一次，
    第二次直接放行，max_turns兜底。"""
    responses = [
        LLMResponse(
            stop_reason="tool_use", text=None, tool_calls=[_financials_call(), _verify_number_call()], raw=None
        ),
        LLMResponse(stop_reason="end_turn", text=_full_report("结论"), tool_calls=[], raw=None),
        LLMResponse(stop_reason="end_turn", text=_full_report("结论"), tool_calls=[], raw=None),
    ]
    fake = FakeLLMClient(responses)
    with (
        patch("app.services.agent.loop.get_llm_client", return_value=fake),
        patch("app.services.agent.loop.execute_tool", new=AsyncMock(side_effect=_execute_tool_returning(False))),
    ):
        result = asyncio.run(run_agent_loop("AAPL"))

    assert fake.calls == 3  # 没有第四次调用，说明没有再次触发
    assert result.completed is True
    assert result.verification_mismatch_triggered is True
    assert result.verification_mismatch_resolved is False
