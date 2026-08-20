"""Reflexion兜底：run_agent_loop新增的reflexion_check回调机制。跟自我核查兜底
是同一套"发现问题→插入提示→再走一轮，只拦一次"的模式，区别是这里的"发现问题"
判断完全交给调用方传入的回调决定，loop.py本身不关心判断逻辑是什么（不直接
依赖reward.py），只测试机制本身：回调返回字符串就该注入并续一轮，返回None
就该正常收尾，只触发一次不管回调后续怎么判断。

大部分用例先安排一轮成功的get_financials+verify_number调用，满足结构合规/
工具使用底线/自我核查这几道排在reflexion前面的gate——不然模型第一次end_turn
时会先被这些拦下来（那是另外几套独立的拦截，分别见test_structure_gate_nudge.py/
test_tool_coverage_gate_nudge.py/test_self_verification_nudge.py），会干扰
"这里到底测的是reflexion"这件事。report文本也都带全三个标签、不含数字型断言，
避免触发可追溯率gate。
"""

import asyncio
from unittest.mock import AsyncMock, patch

from app.services.agent.llm_client import LLMResponse, ToolCall
from app.services.agent.loop import run_agent_loop


class FakeLLMClient:
    def __init__(self, responses: list[LLMResponse]):
        self._responses = list(responses)
        self.calls = 0
        self.received_messages: list[list[dict]] = []

    async def create_message(self, system, messages, tools, temperature=None):
        self.received_messages.append(messages)
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


def _financials_turn() -> LLMResponse:
    return LLMResponse(
        stop_reason="tool_use",
        text=None,
        tool_calls=[ToolCall(id="0", name="get_financials", input={"ticker": "AAPL"})],
        raw=None,
    )


def _full_report(conclusion: str) -> str:
    return f"<conclusion>{conclusion}</conclusion><evidence>e</evidence><flags>f</flags>"


def test_reflexion_injects_critique_and_forces_another_turn_when_check_returns_text():
    responses = [
        _financials_and_verify_turn(),
        LLMResponse(stop_reason="end_turn", text=_full_report("结论"), tool_calls=[], raw=None),
        LLMResponse(stop_reason="end_turn", text=_full_report("结论（已整改）"), tool_calls=[], raw=None),
    ]
    fake = FakeLLMClient(responses)

    async def reflexion_check(reasoning_notes, transcript):
        return "过程裁判说信息收集不充分，请补充核实。"

    with (
        patch("app.services.agent.loop.get_llm_client", return_value=fake),
        patch("app.services.agent.loop.execute_tool", new=AsyncMock(return_value=("{}", False))),
    ):
        result = asyncio.run(run_agent_loop("AAPL", reflexion_check=reflexion_check))

    assert fake.calls == 3
    assert result.self_verification_triggered is True
    assert result.reflexion_triggered is True
    assert result.final_report == _full_report("结论（已整改）")
    # 批评文字确实被塞进了第3轮的对话历史里
    assert any("过程裁判说信息收集不充分" in str(msg) for msg in fake.received_messages[2])


def test_no_reflexion_when_check_returns_none():
    responses = [
        _financials_and_verify_turn(),
        LLMResponse(stop_reason="end_turn", text=_full_report("结论"), tool_calls=[], raw=None),
    ]
    fake = FakeLLMClient(responses)

    async def reflexion_check(reasoning_notes, transcript):
        return None

    with (
        patch("app.services.agent.loop.get_llm_client", return_value=fake),
        patch("app.services.agent.loop.execute_tool", new=AsyncMock(return_value=("{}", False))),
    ):
        result = asyncio.run(run_agent_loop("AAPL", reflexion_check=reflexion_check))

    assert fake.calls == 2
    assert result.reflexion_triggered is False
    assert result.completed is True


def test_reflexion_check_not_invoked_when_not_provided():
    """默认行为不变：不传reflexion_check（普通单次分析的调用方）时，机制完全
    不生效，跟这个功能加进来之前行为一致。"""
    responses = [
        _financials_and_verify_turn(),
        LLMResponse(stop_reason="end_turn", text=_full_report("结论"), tool_calls=[], raw=None),
    ]
    fake = FakeLLMClient(responses)

    with (
        patch("app.services.agent.loop.get_llm_client", return_value=fake),
        patch("app.services.agent.loop.execute_tool", new=AsyncMock(return_value=("{}", False))),
    ):
        result = asyncio.run(run_agent_loop("AAPL"))

    assert fake.calls == 2
    assert result.reflexion_triggered is False


def test_reflexion_only_fires_once_even_if_check_keeps_returning_critique():
    """模型被整改一次之后，就算过程裁判还是不满意，也不该无限循环下去——
    只拦一次，第二次直接放行，用户最终拿到的是"整改过一次"的版本。"""
    responses = [
        _financials_and_verify_turn(),
        LLMResponse(stop_reason="end_turn", text=_full_report("结论"), tool_calls=[], raw=None),
        LLMResponse(stop_reason="end_turn", text=_full_report("结论（仍不完美）"), tool_calls=[], raw=None),
    ]
    fake = FakeLLMClient(responses)
    check_calls = 0

    async def reflexion_check(reasoning_notes, transcript):
        nonlocal check_calls
        check_calls += 1
        return "还是不够好"

    with (
        patch("app.services.agent.loop.get_llm_client", return_value=fake),
        patch("app.services.agent.loop.execute_tool", new=AsyncMock(return_value=("{}", False))),
    ):
        result = asyncio.run(run_agent_loop("AAPL", reflexion_check=reflexion_check))

    assert fake.calls == 3  # 没有第4次调用
    assert check_calls == 1  # reflexion_check本身也只被问过一次
    assert result.reflexion_triggered is True
    assert result.completed is True


def test_self_verification_nudge_takes_priority_over_reflexion_in_the_same_round():
    """同一轮里如果自我核查都还没过关，不该先去问过程裁判——先把底线要求
    （核实过数字）解决掉，reflexion_check在这一轮根本不会被调用。这里刻意只
    调用get_financials、不调用verify_number，让结构合规/工具使用底线这两道
    排在自我核查前面的gate过关，但自我核查本身过不了关。"""
    responses = [
        _financials_turn(),
        LLMResponse(stop_reason="end_turn", text=_full_report("结论"), tool_calls=[], raw=None),
        LLMResponse(stop_reason="end_turn", text=_full_report("结论（已核查）"), tool_calls=[], raw=None),
    ]
    fake = FakeLLMClient(responses)
    reflexion_check = AsyncMock(return_value=None)

    with (
        patch("app.services.agent.loop.get_llm_client", return_value=fake),
        patch("app.services.agent.loop.execute_tool", new=AsyncMock(return_value=("{}", False))),
    ):
        result = asyncio.run(run_agent_loop("AAPL", reflexion_check=reflexion_check))

    # 第一次end_turn触发的是自我核查兜底，不是reflexion；自我核查只拦一次，
    # 第二次end_turn时（依然没调用过verify_number）自我核查不会再拦，
    # 才轮到reflexion_check第一次被调用
    reflexion_check.assert_called_once()
    assert result.self_verification_triggered is False  # 全程都没真的调用verify_number
    assert result.reflexion_triggered is False


def test_reflexion_triggered_reflects_state_on_max_turns_exceeded():
    """max_turns=3：第0轮先查financials+核实一个数字（满足自我核查底线），
    第1轮end_turn时结构/工具覆盖/自我核查都已经过关、轮到reflexion_check
    介入并返回批评——插入提示后continue，但预算已经耗尽，没有第3轮真正把
    修正版写出来，落到max_turns_exceeded分支。reflexion_triggered应该如实
    反映"确实触发过"，不因为最终没能在预算内收尾就被抹成False。"""
    responses = [
        _financials_and_verify_turn(),
        LLMResponse(stop_reason="end_turn", text=_full_report("结论"), tool_calls=[], raw=None),
    ]
    fake = FakeLLMClient(responses)

    async def reflexion_check(reasoning_notes, transcript):
        return "请补充信息"

    with (
        patch("app.services.agent.loop.get_llm_client", return_value=fake),
        patch("app.services.agent.loop.execute_tool", new=AsyncMock(return_value=("{}", False))),
    ):
        result = asyncio.run(run_agent_loop("AAPL", max_turns=2, reflexion_check=reflexion_check))

    assert result.stop_reason == "max_turns_exceeded"
    assert result.self_verification_triggered is True
    assert result.reflexion_triggered is True  # 第1轮已经真实触发过，即便后面没能在预算内收尾
