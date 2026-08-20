"""可追溯率gate：verify_number那道gate只管模型自己主动选去核实的那一个数字，
简报里其他没被挑中核实的数字断言完全没人管——这里补一道覆盖面更广的检查：
用traceability.py算出来的"全部数字断言里有多少能在本次工具输出里找到依据"这个
客观比例，明显偏低时代码层面拦下来，不是任由模型自己觉得"这份报告写得不错"
就直接定稿。跟verify_number的gate、自我核查的gate是同一套"发现问题→插入提示
→只拦一次"机制。
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


def _financials_call(call_id: str = "2") -> ToolCall:
    return ToolCall(id=call_id, name="get_financials", input={"ticker": "AAPL"})


async def _fake_execute_tool(name: str, tool_input: dict) -> tuple[str, bool]:
    if name == "verify_number":
        return (json.dumps({"ticker": "AAPL", "metric": "revenue", "matches": True}), False)
    return (json.dumps({"revenue": 950000000}), False)


def test_nudge_injected_when_traceability_ratio_is_low():
    """模型被拦下来之后，只改了<conclusion>的措辞（"已补充说明"），但
    <evidence>里那些验证不了的数字断言原封不动——traceability_gate_resolved
    是拿最终文本重新算一次可追溯率，不会被"改了收尾措辞"这种表面动作骗过去，
    应该如实反映"其实没有真的补充依据"。"""
    bad_evidence = (
        "<evidence>营收增长12.3%，净利润5.6亿，毛利率34%，资本支出2.1亿</evidence><flags></flags>"
    )
    responses = [
        LLMResponse(stop_reason="tool_use", text=None, tool_calls=[_verify_number_call(), _financials_call()], raw=None),
        LLMResponse(stop_reason="end_turn", text=f"<conclusion>结论</conclusion>{bad_evidence}", tool_calls=[], raw=None),
        LLMResponse(
            stop_reason="end_turn",
            text=f"<conclusion>结论（已补充说明）</conclusion>{bad_evidence}",
            tool_calls=[],
            raw=None,
        ),
    ]
    fake = FakeLLMClient(responses)
    with (
        patch("app.services.agent.loop.get_llm_client", return_value=fake),
        patch("app.services.agent.loop.execute_tool", new=AsyncMock(side_effect=_fake_execute_tool)),
    ):
        result = asyncio.run(run_agent_loop("AAPL"))

    # 工具调用轮 + 想结束但可追溯率太低被拦下 + 真正收尾，一共3次create_message调用
    assert fake.calls == 3
    assert result.completed is True
    assert result.traceability_gate_triggered is True
    assert result.traceability_gate_resolved is False
    assert result.verification_mismatch_triggered is False


def test_resolved_is_true_when_revised_evidence_is_actually_traceable():
    """跟上一个测试对照：模型被拦下来之后真的把验证不了的数字换成了能在
    工具输出里找到依据的数字，resolved应该如实反映"确实改善了"。"""
    bad_evidence = (
        "<evidence>营收增长12.3%，净利润5.6亿，毛利率34%，资本支出2.1亿</evidence><flags></flags>"
    )
    good_evidence = "<evidence>营收950000000美元</evidence><flags></flags>"
    responses = [
        LLMResponse(stop_reason="tool_use", text=None, tool_calls=[_verify_number_call(), _financials_call()], raw=None),
        LLMResponse(stop_reason="end_turn", text=f"<conclusion>结论</conclusion>{bad_evidence}", tool_calls=[], raw=None),
        LLMResponse(
            stop_reason="end_turn", text=f"<conclusion>结论（已修正）</conclusion>{good_evidence}", tool_calls=[], raw=None
        ),
    ]
    fake = FakeLLMClient(responses)
    with (
        patch("app.services.agent.loop.get_llm_client", return_value=fake),
        patch("app.services.agent.loop.execute_tool", new=AsyncMock(side_effect=_fake_execute_tool)),
    ):
        result = asyncio.run(run_agent_loop("AAPL"))

    assert result.traceability_gate_triggered is True
    assert result.traceability_gate_resolved is True


def test_no_nudge_when_traceability_ratio_is_healthy():
    good_evidence = "<evidence>营收950000000美元</evidence><flags></flags>"
    responses = [
        LLMResponse(
            stop_reason="tool_use",
            text=None,
            tool_calls=[_verify_number_call(), _financials_call()],
            raw=None,
        ),
        LLMResponse(stop_reason="end_turn", text=f"<conclusion>结论</conclusion>{good_evidence}", tool_calls=[], raw=None),
    ]
    fake = FakeLLMClient(responses)
    with (
        patch("app.services.agent.loop.get_llm_client", return_value=fake),
        patch("app.services.agent.loop.execute_tool", new=AsyncMock(side_effect=_fake_execute_tool)),
    ):
        result = asyncio.run(run_agent_loop("AAPL"))

    assert fake.calls == 2
    assert result.traceability_gate_triggered is False
    assert result.traceability_gate_resolved is True  # 从没出过问题，vacuous true


def test_no_nudge_when_report_has_no_numeric_claims():
    """total==0（没有可核对的数字主张）不该被当成"有问题"去拦——跟
    traceability.py本身"total==0是无法判断不是有问题"的定位一致。"""
    responses = [
        LLMResponse(stop_reason="tool_use", text=None, tool_calls=[_verify_number_call(), _financials_call()], raw=None),
        LLMResponse(
            stop_reason="end_turn",
            text="<conclusion>结论</conclusion><evidence>管理层态度乐观，未提及具体数字</evidence><flags></flags>",
            tool_calls=[],
            raw=None,
        ),
    ]
    fake = FakeLLMClient(responses)
    with (
        patch("app.services.agent.loop.get_llm_client", return_value=fake),
        patch("app.services.agent.loop.execute_tool", new=AsyncMock(side_effect=_fake_execute_tool)),
    ):
        result = asyncio.run(run_agent_loop("AAPL"))

    assert fake.calls == 2
    assert result.traceability_gate_triggered is False
    assert result.traceability_gate_resolved is True


def test_nudge_does_not_retrigger_if_ratio_still_low_after_nudge():
    """拦了一次之后模型给出的修订版可追溯率依然很低，不该无限重复拦截——
    只拦一次，max_turns兜底。"""
    bad_evidence = (
        "<evidence>营收增长12.3%，净利润5.6亿，毛利率34%，资本支出2.1亿</evidence><flags></flags>"
    )
    responses = [
        LLMResponse(stop_reason="tool_use", text=None, tool_calls=[_verify_number_call(), _financials_call()], raw=None),
        LLMResponse(stop_reason="end_turn", text=f"<conclusion>结论</conclusion>{bad_evidence}", tool_calls=[], raw=None),
        LLMResponse(stop_reason="end_turn", text=f"<conclusion>结论（仍未改善）</conclusion>{bad_evidence}", tool_calls=[], raw=None),
    ]
    fake = FakeLLMClient(responses)
    with (
        patch("app.services.agent.loop.get_llm_client", return_value=fake),
        patch("app.services.agent.loop.execute_tool", new=AsyncMock(side_effect=_fake_execute_tool)),
    ):
        result = asyncio.run(run_agent_loop("AAPL"))

    assert fake.calls == 3  # 没有第四次调用，说明没有再次触发
    assert result.completed is True
    assert result.traceability_gate_triggered is True
    assert result.traceability_gate_resolved is False
