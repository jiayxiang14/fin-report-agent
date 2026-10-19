"""核实覆盖gate：核实过的数字至少要有一个写进了最终简报。

全链路评测发现：24次运行里Agent全部是先核实"打算写的数字"、下一轮才写简报，
原来的检查只看"有没有调用过verify_number"——核实的数字跟写进去的不一定是同一个。
这里不规定先后顺序，只要求核实的对象是简报里真的写了的数字。
"""

import asyncio
import json
from unittest.mock import patch

from app.services.agent.llm_client import LLMResponse, ToolCall
from app.services.agent.loop import _verification_covers_report, run_agent_loop
from app.services.agent.traceability import report_mentions_value


def test_report_mentions_value_handles_chinese_magnitudes_and_separators():
    assert report_mentions_value("FY2025营收20.24亿美元", 2_023_994_000)
    assert report_mentions_value("营收 $2,023,994,000", 2_023_994_000)
    assert not report_mentions_value("营收约30亿美元", 2_023_994_000)


def test_coverage_requires_at_least_one_verified_value_in_the_report():
    assert _verification_covers_report("净利润 131.07亿美元", [13_107_000_000.0, 13_107_000_000.0])
    assert not _verification_covers_report("营收同比增长16%", [13_107_000_000.0])
    # 没做任何核实时不在这里拦（那是"没做自我核查"那条检查的事）
    assert _verification_covers_report("任何内容", [])


class FakeLLMClient:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = 0
        self.nudges: list[str] = []

    async def create_message(self, system, messages, tools, temperature=None, on_text_delta=None):
        last = messages[-1]
        if last["role"] == "user" and isinstance(last["content"], str) and "问题需要处理" in last["content"]:
            self.nudges.append(last["content"])
        response = self._responses[self.calls]
        self.calls += 1
        return response

    def append_assistant_turn(self, messages, response):
        return [*messages, {"role": "assistant", "content": "fake"}]

    def append_tool_results(self, messages, tool_calls, results):
        return [*messages, {"role": "user", "content": "fake"}]


async def _fake_execute_tool(name, tool_input):
    if name == "verify_number":
        value = tool_input["claimed_value"]
        return json.dumps({"metric": tool_input["metric"], "period": "annual", "claimed_value": value, "actual_value": value, "matches": True}), False
    return "{}", False


def _verify_turn(value):
    return LLMResponse(
        stop_reason="tool_use",
        text=None,
        tool_calls=[
            ToolCall(id=f"f{value}", name="get_financials", input={"ticker": "AAPL"}),
            ToolCall(id=f"v{value}", name="verify_number", input={"ticker": "AAPL", "metric": "revenue", "claimed_value": value, "period": "annual"}),
        ],
        raw=None,
    )


def _report(evidence):
    return f"<conclusion>c</conclusion><evidence>{evidence}</evidence><flags>无</flags><sentiment>neutral</sentiment>"


def _run(responses):
    fake = FakeLLMClient(responses)
    with patch("app.services.agent.loop.get_llm_client", return_value=fake), patch("app.services.agent.loop.execute_tool", new=_fake_execute_tool):
        result = asyncio.run(run_agent_loop("AAPL"))
    return result, fake


def test_verifying_a_number_that_never_appears_in_the_report_is_sent_back_once():
    result, fake = _run(
        [
            _verify_turn(416_161_000_000),  # 核实的是4161.61亿
            LLMResponse(stop_reason="end_turn", text=_report("营收同比增长显著"), tool_calls=[], raw=None),
            _verify_turn(416_161_000_000),
            LLMResponse(stop_reason="end_turn", text=_report("FY2025营收4161.61亿美元"), tool_calls=[], raw=None),
        ]
    )
    assert len(fake.nudges) == 1
    assert "没有一个出现在这份简报里" in fake.nudges[0]
    assert result.verification_coverage_gate_triggered is True
    assert result.verification_coverage_gate_resolved is True


def test_report_that_contains_the_verified_number_is_not_sent_back():
    result, fake = _run(
        [
            _verify_turn(416_161_000_000),
            LLMResponse(stop_reason="end_turn", text=_report("FY2025营收4161.61亿美元"), tool_calls=[], raw=None),
        ]
    )
    assert fake.nudges == []
    assert result.verification_coverage_gate_triggered is False
    assert result.verification_coverage_gate_resolved is True


def test_magnitude_unit_separated_by_a_space_is_still_recognized():
    """真实误判：简报写的是"营收 1094.17 亿美元"（数字和"亿"之间有空格），
    被当成1094.17，核实覆盖检查因此误判打回。"""
    assert report_mentions_value("营收 1094.17 亿美元，同比 +16.4%", 109_417_000_000)
    assert report_mentions_value("净利润 297.89 亿美元", 29_789_000_000)
    # 隔太远的不算：数字后面跟的是别的内容
    assert not report_mentions_value("营收 1094.17 美元的单价，亿万用户", 109_417_000_000)


def test_english_magnitude_suffixes_are_recognized():
    """真实缺口：AMZN简报写的是"营收$200.606B"，原来只认中文的亿/万。"""
    assert report_mentions_value("营收$200.606B，同比+19.62%", 200_606_000_000)
    assert report_mentions_value("净利润 62.65 billion 美元", 62_650_000_000)
    assert report_mentions_value("市值 $3.1T", 3_100_000_000_000)
    assert report_mentions_value("自由现金流 $850M", 850_000_000)
    # 报表类型这类写法不能被误认成单位
    assert not report_mentions_value("10-K 年报", 10_000_000_000)
