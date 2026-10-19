"""数据缺失披露gate：工具调用失败、之后又没重新成功拿到数据时，<flags>里必须
写明缺了哪个数据源——借鉴claude-quickstarts daily-brief的"读不到的数据源要
报告为'无法读取'，不能当作'没新消息'"。

fixture先安排get_financials+verify_number满足排在前面的几道gate，report文本
不含数字型断言（避免触发可追溯率gate），这个文件只关心"数据缺失有没有被
写进flags"这一件事。
"""

import asyncio
import json
from unittest.mock import patch

from app.models.agent import TranscriptEntry
from app.services.agent.llm_client import LLMResponse, ToolCall
from app.services.agent.loop import _undisclosed_data_gaps, _unrecovered_failed_tools, run_agent_loop


def _entry(tool_name: str, is_error: bool) -> TranscriptEntry:
    return TranscriptEntry(turn=0, tool_name=tool_name, tool_input={}, tool_output_summary="", is_error=is_error)


def _report(flags: str) -> str:
    return f"<conclusion>c</conclusion><evidence>e</evidence><flags>{flags}</flags><sentiment>neutral</sentiment>"


# ---------- 纯函数 ----------


def test_only_failures_never_recovered_later_count_as_missing_data():
    transcript = [
        _entry("get_earnings_surprise", True),
        _entry("get_filing_text", True),
        _entry("get_filing_text", False),  # 换参数重试成功了，不算缺失
        _entry("verify_number", True),  # 核查请求本身的问题，不算缺数据
        _entry("get_financials", False),
    ]
    assert _unrecovered_failed_tools(transcript) == ["get_earnings_surprise"]


def test_disclosure_needs_a_gap_phrase_and_the_tool_keyword_in_the_same_sentence():
    transcript = [_entry("get_earnings_surprise", True)]

    assert _undisclosed_data_gaps(_report("本次未能获取EPS一致预期数据，无法判断是否超预期。"), transcript) == []
    assert _undisclosed_data_gaps(_report("- EPS 预期数据：获取失败"), transcript) == []
    # 提到了EPS但没说缺失
    assert _undisclosed_data_gaps(_report("EPS表现稳健。"), transcript) == ["get_earnings_surprise"]
    # 说了缺失，但说的是别的数据；缺失和EPS分在两句话里，不算写明
    assert _undisclosed_data_gaps(_report("同行数据缺失。EPS表现稳健。"), transcript) == ["get_earnings_surprise"]
    # 只在<evidence>里提不算，必须写进<flags>
    report = "<conclusion>c</conclusion><evidence>未能获取EPS预期数据</evidence><flags>无</flags>"
    assert _undisclosed_data_gaps(report, transcript) == ["get_earnings_surprise"]


def test_each_failed_tool_must_be_disclosed_separately():
    transcript = [_entry("get_earnings_surprise", True), _entry("get_peer_comparison", True)]
    report = _report("未能获取EPS预期数据\n同行对比结果正常")
    assert _undisclosed_data_gaps(report, transcript) == ["get_peer_comparison"]


def test_no_failed_tools_means_nothing_to_disclose():
    assert _undisclosed_data_gaps(_report(""), [_entry("get_financials", False)]) == []


# ---------- Agent Loop 集成 ----------


class FakeLLMClient:
    def __init__(self, responses: list[LLMResponse]):
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


async def _fake_execute_tool(name: str, tool_input: dict) -> tuple[str, bool]:
    if name == "get_earnings_surprise":
        return "Alpha Vantage 今日额度已用完", True
    if name == "verify_number":
        return json.dumps({"metric": "revenue", "period": "annual", "matches": True}), False
    return "{}", False


def _tool_turn() -> LLMResponse:
    return LLMResponse(
        stop_reason="tool_use",
        text=None,
        tool_calls=[
            ToolCall(id="0", name="get_financials", input={"ticker": "AAPL"}),
            ToolCall(id="1", name="get_earnings_surprise", input={"ticker": "AAPL"}),
            ToolCall(id="2", name="verify_number", input={"ticker": "AAPL", "metric": "revenue", "claimed_value": 1.0, "period": "annual"}),
        ],
        raw=None,
    )


def _run(responses):
    fake = FakeLLMClient(responses)
    with (
        patch("app.services.agent.loop.get_llm_client", return_value=fake),
        patch("app.services.agent.loop.execute_tool", new=_fake_execute_tool),
    ):
        result = asyncio.run(run_agent_loop("AAPL"))
    return result, fake


def test_report_that_hides_a_failed_tool_is_sent_back_once_and_resolved_after_disclosure():
    result, fake = _run(
        [
            _tool_turn(),
            LLMResponse(stop_reason="end_turn", text=_report("未发现明显异常"), tool_calls=[], raw=None),
            LLMResponse(stop_reason="end_turn", text=_report("本次未能获取EPS一致预期数据，超预期判断缺少依据"), tool_calls=[], raw=None),
        ]
    )

    assert len(fake.nudges) == 1
    assert "EPS超预期幅度（get_earnings_surprise）" in fake.nudges[0]
    assert "不要断言这项数据本身不存在" in fake.nudges[0]
    assert result.data_gap_disclosure_gate_triggered is True
    assert result.data_gap_disclosure_gate_resolved is True
    assert result.completed is True


def test_resolved_is_false_when_the_model_still_does_not_disclose():
    result, fake = _run(
        [
            _tool_turn(),
            LLMResponse(stop_reason="end_turn", text=_report("未发现明显异常"), tool_calls=[], raw=None),
            LLMResponse(stop_reason="end_turn", text=_report("依然未发现明显异常"), tool_calls=[], raw=None),
        ]
    )

    assert result.data_gap_disclosure_gate_triggered is True
    assert result.data_gap_disclosure_gate_resolved is False  # 只拦一次，如实记录没解决


def test_report_that_already_discloses_the_gap_is_not_sent_back():
    result, fake = _run(
        [
            _tool_turn(),
            LLMResponse(stop_reason="end_turn", text=_report("EPS预期数据获取失败，无法判断是否超预期"), tool_calls=[], raw=None),
        ]
    )

    assert fake.nudges == []
    assert result.data_gap_disclosure_gate_triggered is False
    assert result.data_gap_disclosure_gate_resolved is True
