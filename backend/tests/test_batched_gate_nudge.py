"""批量gate检测：之前6道确定性gate（结构/工具覆盖/自我核查/核查一致性/
可追溯率/sentiment一致性）是逐个检测、逐个拦截的——测到第一个有问题的就
`continue`，下一轮才测下一个，最坏情况6道gate各拦一次=最多6轮，直接挤占
MAX_TURNS预算（真实观测到过一次max_turns_exceeded）。现在改成一次性测完
6项、把所有真正有问题的收集起来合成一条消息、只`continue`一次。

这个文件专门验证"批量"这个核心行为本身：多个问题同时存在时，是不是真的
只发一次nudge（而不是逐个拦截），以及批量nudge发出去之后模型完全不配合时
的二次机会（`IGNORED_FORMAT_NUDGE`）能不能生效。单个gate各自的
触发/resolved逻辑已经在`test_structure_gate_nudge.py`等5个专属文件里
覆盖过，这里不重复。
"""

import asyncio
import json
from unittest.mock import AsyncMock, patch

from app.services.agent.llm_client import LLMResponse, ToolCall
from app.services.agent.loop import _build_combined_gate_nudge, run_agent_loop


def test_build_combined_gate_nudge_numbers_every_issue():
    nudge = _build_combined_gate_nudge(["问题A", "问题B", "问题C"])

    assert "1. 问题A" in nudge
    assert "2. 问题B" in nudge
    assert "3. 问题C" in nudge
    assert "3个问题" in nudge
    assert "<conclusion>" in nudge and "<evidence>" in nudge and "<flags>" in nudge


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


async def _fake_execute_tool(name: str, tool_input: dict) -> tuple[str, bool]:
    if name == "verify_number":
        return (json.dumps({"matches": True}), False)
    return (json.dumps({"revenue": 950000000}), False)


def test_multiple_simultaneous_issues_are_collapsed_into_a_single_nudge():
    """模型第一次end_turn就同时踩中4个问题：从没调用过任何工具（触发工具
    覆盖+自我核查两项）、简报缺<flags>标签（结构）、引用了一个查不到依据的
    数字（可追溯率）。如果是逐个拦截，这至少要4轮nudge才能收尾；批量设计
    下应该只用1次nudge就把4个问题全部列出来，模型下一轮补齐工具调用，
    再下一轮正常收尾——一共3次create_message调用，不是4+。"""
    responses = [
        LLMResponse(
            stop_reason="end_turn",
            text="<conclusion>结论</conclusion><evidence>营收增长12.3%</evidence>",
            tool_calls=[],
            raw=None,
        ),
        LLMResponse(
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
        ),
        LLMResponse(
            stop_reason="end_turn",
            text="<conclusion>结论（已处理）</conclusion><evidence>营收950000000美元</evidence><flags>f</flags>",
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

    # 只有3次create_message调用：首次草稿 + 工具调用轮 + 收尾，不是4道gate各拦一次
    assert fake.calls == 3
    assert result.completed is True
    assert result.structure_gate_triggered is True
    assert result.structure_gate_resolved is True
    assert result.tool_coverage_gate_triggered is True
    assert result.tool_coverage_gate_resolved is True
    assert result.traceability_gate_triggered is True
    assert result.traceability_gate_resolved is True

    # 批量nudge（received_messages[1]，发给"工具调用轮"那次请求的历史）里
    # 应该同时列出全部4个问题，证明确实是一次性说清楚，不是逐个提示
    combined_nudge = fake.received_messages[1][-1]["content"]
    assert "verify_number" in combined_nudge  # 自我核查
    assert "get_financials" in combined_nudge  # 工具覆盖
    assert "flags" in combined_nudge  # 结构
    assert "12.3" in combined_nudge  # 可追溯率


def test_second_chance_nudge_only_fires_once_after_batch():
    """批量nudge之后，模型两次都用不带标签的收尾话敷衍——二次机会只给一次，
    第二次依然不配合就不再拦截，直接落到resolved=False如实报告，不会无限
    循环。"""
    responses = [
        LLMResponse(
            stop_reason="end_turn", text="<conclusion>结论</conclusion><evidence>e</evidence>", tool_calls=[], raw=None
        ),  # 缺flags，触发批量nudge
        LLMResponse(stop_reason="end_turn", text="已处理", tool_calls=[], raw=None),  # 完全没配合，触发二次机会
        LLMResponse(stop_reason="end_turn", text="真的已处理", tool_calls=[], raw=None),  # 依然没配合
    ]
    fake = FakeLLMClient(responses)
    with (
        patch("app.services.agent.loop.get_llm_client", return_value=fake),
        patch("app.services.agent.loop.execute_tool", new=AsyncMock(side_effect=_fake_execute_tool)),
    ):
        result = asyncio.run(run_agent_loop("AAPL"))

    assert fake.calls == 3  # 没有第四次调用，说明二次机会没有重复触发
    assert result.completed is True
    assert result.structure_gate_resolved is False


def test_no_nudge_at_all_when_first_draft_is_already_clean():
    """健康路径：第一次草稿就已经完整合规，不该触发任何批量nudge或二次机会。"""
    responses = [
        LLMResponse(
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
        ),
        LLMResponse(
            stop_reason="end_turn",
            text="<conclusion>结论</conclusion><evidence>营收950000000美元</evidence><flags>f</flags>",
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
    assert result.structure_gate_triggered is False
    assert result.tool_coverage_gate_triggered is False
    assert result.traceability_gate_triggered is False
    assert result.sentiment_consistency_gate_triggered is False
