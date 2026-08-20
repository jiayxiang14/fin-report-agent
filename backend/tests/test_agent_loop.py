"""回归测试：Stage 3 复盘发现的两个逻辑问题——
(1) 中间轮次的文字被静默丢弃，没有进 reasoning_notes
(2) completed 字段只看"是不是tool_use"，refusal/max_tokens 也被错误标记成完成

用一个可控的假 LLM 客户端跑 Loop，不需要真实调用 DeepSeek/Claude。
"""

import asyncio
from unittest.mock import AsyncMock, patch

from app.services.agent.llm_client import LLMResponse, ToolCall
from app.services.agent.loop import run_agent_loop


class FakeLLMClient:
    def __init__(self, responses: list[LLMResponse]):
        self._responses = list(responses)
        self.calls = 0
        self.received_temperatures: list[float | None] = []

    async def create_message(self, system, messages, tools, temperature=None):
        self.received_temperatures.append(temperature)
        response = self._responses[self.calls]
        self.calls += 1
        return response

    def append_assistant_turn(self, messages, response):
        return [*messages, {"role": "assistant", "content": "fake"}]

    def append_tool_results(self, messages, tool_calls, results):
        return [*messages, {"role": "user", "content": "fake"}]


def test_intermediate_turn_text_is_captured_as_reasoning_note():
    responses = [
        LLMResponse(
            stop_reason="tool_use",
            text="先查一下财务数据",
            tool_calls=[ToolCall(id="1", name="get_financials", input={"ticker": "AAPL"})],
            raw=None,
        ),
        # 第一次 end_turn 时还没调用过 verify_number，会被自我核查兜底拦一次
        # （见 test_self_verification_nudge.py），所以还需要一轮收尾的 end_turn。
        # evidence/flags标签必须都在，不然会先被结构合规gate拦下来
        # （见test_structure_gate_nudge.py），需要更多轮次，这里不测那个
        LLMResponse(
            stop_reason="end_turn",
            text="<conclusion>...</conclusion><evidence>...</evidence><flags>...</flags>",
            tool_calls=[],
            raw=None,
        ),
        LLMResponse(
            stop_reason="end_turn",
            text="<conclusion>...</conclusion><evidence>...</evidence><flags>...</flags>",
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

    assert any(n.text == "先查一下财务数据" for n in result.reasoning_notes)
    assert result.completed is True


def test_refusal_stop_reason_is_not_marked_completed():
    responses = [LLMResponse(stop_reason="refusal", text=None, tool_calls=[], raw=None)]
    fake = FakeLLMClient(responses)
    with patch("app.services.agent.loop.get_llm_client", return_value=fake):
        result = asyncio.run(run_agent_loop("AAPL"))
    assert result.completed is False
    assert result.stop_reason == "refusal"


def test_max_tokens_stop_reason_is_not_marked_completed():
    responses = [LLMResponse(stop_reason="max_tokens", text="部分内容...", tool_calls=[], raw=None)]
    fake = FakeLLMClient(responses)
    with patch("app.services.agent.loop.get_llm_client", return_value=fake):
        result = asyncio.run(run_agent_loop("AAPL"))
    assert result.completed is False
    assert result.final_report == "部分内容..."  # 截断的部分内容仍然透出，但completed如实标False


def test_end_turn_is_marked_completed():
    responses = [LLMResponse(stop_reason="end_turn", text="最终简报", tool_calls=[], raw=None)]
    fake = FakeLLMClient(responses)
    with patch("app.services.agent.loop.get_llm_client", return_value=fake):
        result = asyncio.run(run_agent_loop("AAPL"))
    assert result.completed is True
    assert result.stop_reason == "end_turn"


def test_on_event_fires_in_expected_order_for_reasoning_and_tool_calls():
    """Stage 4新增：on_event 回调是SSE流式展示的数据来源，验证它按
    reasoning -> tool_call_started -> tool_call_finished 的顺序触发，
    且 on_event=None（默认值）时完全不影响行为——这是前面几个测试早就
    验证过的。"""
    responses = [
        LLMResponse(
            stop_reason="tool_use",
            text="先查一下财务数据",
            tool_calls=[ToolCall(id="1", name="get_financials", input={"ticker": "AAPL"})],
            raw=None,
        ),
        # 第一次 end_turn 时还没调用过 verify_number，会被自我核查兜底拦一次
        # （见 test_self_verification_nudge.py），所以还需要一轮收尾的 end_turn。
        # evidence/flags标签必须都在，不然会先被结构合规gate拦下来
        LLMResponse(
            stop_reason="end_turn",
            text="<conclusion>...</conclusion><evidence>...</evidence><flags>...</flags>",
            tool_calls=[],
            raw=None,
        ),
        LLMResponse(
            stop_reason="end_turn",
            text="<conclusion>...</conclusion><evidence>...</evidence><flags>...</flags>",
            tool_calls=[],
            raw=None,
        ),
    ]
    fake = FakeLLMClient(responses)
    events: list[dict] = []

    async def on_event(event: dict) -> None:
        events.append(event)

    with (
        patch("app.services.agent.loop.get_llm_client", return_value=fake),
        patch("app.services.agent.loop.execute_tool", new=AsyncMock(return_value=("{}", False))),
    ):
        asyncio.run(run_agent_loop("AAPL", on_event=on_event))

    # 第二轮 end_turn 触发自我核查兜底（还没调用过verify_number），插入nudge后
    # 再走一轮才真正结束，每一轮只要有文字都会触发一次 reasoning 事件，一共5个
    event_types = [e["type"] for e in events]
    assert event_types == ["reasoning", "tool_call_started", "tool_call_finished", "reasoning", "reasoning"]
    assert events[0]["text"] == "先查一下财务数据"
    assert events[1]["tool_name"] == "get_financials"
    assert events[2]["tool_name"] == "get_financials"
    assert events[2]["is_error"] is False
    assert events[3]["text"] == "<conclusion>...</conclusion><evidence>...</evidence><flags>...</flags>"


def test_final_report_falls_back_to_earlier_tagged_draft_when_last_turn_has_no_tags():
    """真实复盘发现的问题：模型有时会在verify_number核实通过之后，最后一轮只写一句
    "核实通过，以上就是最终版本"这样的收尾话去指代前一轮已经写过的完整三段式简报，
    而不是重新完整输出一遍带标签的内容。之前的代码只认最后一轮的文字，会把这句
    收尾话当成final_report，导致前端因为找不到<conclusion>标签而展示"未能按标准
    格式解析"。"""
    responses = [
        LLMResponse(
            stop_reason="tool_use",
            text="<conclusion>亚马逊财报强劲</conclusion><evidence>...</evidence><flags>...</flags>",
            tool_calls=[
                ToolCall(
                    id="1",
                    name="verify_number",
                    input={
                        "ticker": "AMZN",
                        "metric": "operating_income",
                        "claimed_value": 27461000000,
                        "period": "quarterly",
                    },
                )
            ],
            raw=None,
        ),
        LLMResponse(
            stop_reason="end_turn",
            text="核实通过：Q2 2026运营利润$274.61亿与实际数据完全一致。\n\n以上就是本次AMZN投研简报的最终版本。",
            tool_calls=[],
            raw=None,
        ),
    ]
    fake = FakeLLMClient(responses)
    with (
        patch("app.services.agent.loop.get_llm_client", return_value=fake),
        patch("app.services.agent.loop.execute_tool", new=AsyncMock(return_value=("{}", False))),
    ):
        result = asyncio.run(run_agent_loop("AMZN"))

    assert result.completed is True
    assert "<conclusion>亚马逊财报强劲</conclusion>" in result.final_report
    # 收尾话本身依然完整保留在 reasoning_notes 里，只是不作为 final_report
    assert any("以上就是本次AMZN投研简报的最终版本" in n.text for n in result.reasoning_notes)


def test_temperature_is_forwarded_to_llm_client():
    """Best-of-N需要靠不同temperature制造候选间的差异，这里验证 run_agent_loop
    确实把它原样透传给了每一轮的 create_message 调用，且默认(None)不受影响。"""
    responses = [LLMResponse(stop_reason="end_turn", text="最终简报", tool_calls=[], raw=None)]
    fake = FakeLLMClient(responses)
    with patch("app.services.agent.loop.get_llm_client", return_value=fake):
        asyncio.run(run_agent_loop("AAPL", temperature=0.7))
    assert fake.received_temperatures == [0.7]


def test_on_tool_result_receives_full_untruncated_output():
    """Best-of-N的规则打分需要工具的完整原始输出去核对数字，不能用transcript里
    截断到300字符的摘要——这里验证 on_tool_result 拿到的是未截断的完整内容。"""
    long_output = "x" * 500
    responses = [
        LLMResponse(
            stop_reason="tool_use",
            text=None,
            tool_calls=[ToolCall(id="1", name="get_financials", input={"ticker": "AAPL"})],
            raw=None,
        ),
        LLMResponse(stop_reason="end_turn", text="最终简报", tool_calls=[], raw=None),
    ]
    fake = FakeLLMClient(responses)
    captured: list[tuple[str, str]] = []

    async def on_tool_result(tool_name: str, content: str) -> None:
        captured.append((tool_name, content))

    with (
        patch("app.services.agent.loop.get_llm_client", return_value=fake),
        patch("app.services.agent.loop.execute_tool", new=AsyncMock(return_value=(long_output, False))),
    ):
        asyncio.run(run_agent_loop("AAPL", on_tool_result=on_tool_result))

    assert captured == [("get_financials", long_output)]


def test_on_tool_result_not_called_when_tool_errors():
    responses = [
        LLMResponse(
            stop_reason="tool_use",
            text=None,
            tool_calls=[ToolCall(id="1", name="get_financials", input={"ticker": "AAPL"})],
            raw=None,
        ),
        LLMResponse(stop_reason="end_turn", text="最终简报", tool_calls=[], raw=None),
    ]
    fake = FakeLLMClient(responses)
    captured: list[tuple[str, str]] = []

    async def on_tool_result(tool_name: str, content: str) -> None:
        captured.append((tool_name, content))

    with (
        patch("app.services.agent.loop.get_llm_client", return_value=fake),
        patch("app.services.agent.loop.execute_tool", new=AsyncMock(return_value=("出错了", True))),
    ):
        asyncio.run(run_agent_loop("AAPL", on_tool_result=on_tool_result))

    assert captured == []


def test_multiple_tool_calls_in_the_same_turn_run_concurrently():
    """模型在同一轮里一次性打包了3个工具调用——现在应该并行执行，不是排队
    一个个跑。用会真实耗时的fake execute_tool验证：如果是排队执行，3个各
    耗时0.05秒的调用总共至少要0.15秒；并行执行的话，总耗时应该接近单次
    调用的耗时，不会随调用数量线性增长。"""
    responses = [
        LLMResponse(
            stop_reason="tool_use",
            text=None,
            tool_calls=[
                ToolCall(id="1", name="get_financials", input={"ticker": "AAPL"}),
                ToolCall(id="2", name="get_sector_position", input={"ticker": "AAPL"}),
                ToolCall(id="3", name="get_peer_comparison", input={"ticker": "AAPL"}),
            ],
            raw=None,
        ),
        LLMResponse(stop_reason="end_turn", text="最终简报", tool_calls=[], raw=None),
    ]
    fake = FakeLLMClient(responses)

    async def slow_execute_tool(name, tool_input):
        await asyncio.sleep(0.05)
        return "{}", False

    async def run():
        with (
            patch("app.services.agent.loop.get_llm_client", return_value=fake),
            patch("app.services.agent.loop.execute_tool", new=slow_execute_tool),
        ):
            start = asyncio.get_event_loop().time()
            result = await run_agent_loop("AAPL")
            elapsed = asyncio.get_event_loop().time() - start
        return result, elapsed

    result, elapsed = asyncio.run(run())

    assert elapsed < 0.12  # 排队执行至少要0.15秒，留了些余量但远小于串行耗时
    assert len(result.transcript) == 3
    assert {entry.tool_name for entry in result.transcript} == {
        "get_financials",
        "get_sector_position",
        "get_peer_comparison",
    }


def test_max_turns_exceeded_marks_not_completed():
    responses = [
        LLMResponse(
            stop_reason="tool_use",
            text=None,
            tool_calls=[ToolCall(id=str(i), name="get_financials", input={"ticker": "AAPL"})],
            raw=None,
        )
        for i in range(3)
    ]
    fake = FakeLLMClient(responses)
    with (
        patch("app.services.agent.loop.get_llm_client", return_value=fake),
        patch("app.services.agent.loop.execute_tool", new=AsyncMock(return_value=("{}", False))),
    ):
        result = asyncio.run(run_agent_loop("AAPL", max_turns=3))
    assert result.completed is False
    assert result.stop_reason == "max_turns_exceeded"
    assert result.turns_used == 3


def test_traceable_numbers_reflect_final_report_against_tool_outputs():
    """普通单次分析路径（没有传 on_tool_result）现在也应该自己算出数字可
    追溯性信号——之前这个校验只在 Best-of-N 内部生效，普通分析完全没有。"""
    _final_response = LLMResponse(
        stop_reason="end_turn",
        text="<conclusion>强劲</conclusion><evidence>营收达到950000000美元</evidence><flags></flags>",
        tool_calls=[],
        raw=None,
    )
    responses = [
        LLMResponse(
            stop_reason="tool_use",
            text=None,
            tool_calls=[ToolCall(id="1", name="get_financials", input={"ticker": "AAPL"})],
            raw=None,
        ),
        # 第一次 end_turn 时还没调用过 verify_number，会被自我核查兜底拦一次
        # （见 test_self_verification_nudge.py），所以还需要一轮收尾的 end_turn
        _final_response,
        _final_response,
    ]
    fake = FakeLLMClient(responses)
    tool_output = '{"revenue": 950000000}'
    with (
        patch("app.services.agent.loop.get_llm_client", return_value=fake),
        patch("app.services.agent.loop.execute_tool", new=AsyncMock(return_value=(tool_output, False))),
    ):
        result = asyncio.run(run_agent_loop("AAPL"))

    assert result.traceable_numbers_matched == 1
    assert result.traceable_numbers_total == 1


def test_traceable_numbers_default_to_zero_when_no_numeric_claims():
    # 故意不带 <conclusion> 标签——带的话会触发自我核查兜底要求再走一轮，
    # 这里只关心"没有数字主张时 total 是 0"这一件事，不需要凑第二轮响应
    responses = [LLMResponse(stop_reason="end_turn", text="最终简报，无数字", tool_calls=[], raw=None)]
    fake = FakeLLMClient(responses)
    with patch("app.services.agent.loop.get_llm_client", return_value=fake):
        result = asyncio.run(run_agent_loop("AAPL"))

    assert result.traceable_numbers_matched == 0
    assert result.traceable_numbers_total == 0
