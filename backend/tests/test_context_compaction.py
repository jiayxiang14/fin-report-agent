"""上下文压缩：`get_filing_text`这类工具的完整原始输出（实测一份10-K剥离
HTML后约39万字符/约10万token）会原样进messages历史，之后每一轮
create_message都把整个历史重新发一遍——查完财报原文后，剩下的每一轮都在
重复携带这一大块。`_compact_seen_tool_results`在追加"这一轮新的"工具结果
之前，把已经被模型看过至少一轮的大体积旧tool_result替换成占位说明。

只做体积门槛判断（多大的内容该处理），不判断"哪段内容重要该留"——跟
CLAUDE.md"财报文本不做语义清洗规则"是两回事：模型第一次看到工具结果时
永远是完整原文，压缩只发生在它已经看过、后续轮次不需要再重复携带的部分。

三处后续加固（针对"只按体积判断会误伤结构化数据"这类真实风险）：
1. `_COMPACTION_EXEMPT_TOOLS`白名单——财务指标/板块位置这类结构化数据源
   即使体积意外变大也绝不压缩，不是靠"体积天然小"这个巧合
2. `enable_compaction`开关——给eval脚本做可重复的压缩前后质量对照，不用
   每次手改代码
3. `_append_compaction_log`审计日志——持久化记录"这次运行第几轮压缩了
   哪个工具的结果"，运行结束后依然能查
"""

import asyncio
import json
import unittest.mock as mock

from app.services.agent import loop as loop_module
from app.services.agent.llm_client import LLMResponse, ToolCall
from app.services.agent.loop import _append_compaction_log, _compact_seen_tool_results, run_agent_loop

_OVER_THRESHOLD = "x" * (loop_module.CONTEXT_COMPACTION_THRESHOLD_CHARS + 1)
_UNDER_THRESHOLD = "y" * (loop_module.CONTEXT_COMPACTION_THRESHOLD_CHARS - 1)


def _tool_result_message(tool_use_id: str, content: str, *, cache_control: bool = False) -> dict:
    block: dict = {"type": "tool_result", "tool_use_id": tool_use_id, "content": content, "is_error": False}
    if cache_control:
        block["cache_control"] = {"type": "ephemeral"}
    return {"role": "user", "content": [block]}


def test_large_tool_result_gets_compacted():
    messages = [
        {"role": "user", "content": "分析 AAPL"},
        {"role": "assistant", "content": "..."},
        _tool_result_message("1", _OVER_THRESHOLD),
    ]

    compacted, records = _compact_seen_tool_results(messages, {"1": "get_filing_text"})

    result_block = compacted[2]["content"][0]
    assert result_block["content"] != _OVER_THRESHOLD
    assert str(len(_OVER_THRESHOLD)) in result_block["content"]
    assert len(result_block["content"]) < len(_OVER_THRESHOLD)
    assert records == [{"tool_use_id": "1", "tool_name": "get_filing_text", "original_length": len(_OVER_THRESHOLD)}]


def test_small_tool_result_is_left_untouched():
    messages = [_tool_result_message("1", _UNDER_THRESHOLD)]

    compacted, records = _compact_seen_tool_results(messages, {"1": "get_filing_text"})

    assert compacted[0]["content"][0]["content"] == _UNDER_THRESHOLD
    assert records == []


def test_assistant_and_plain_user_messages_are_left_untouched():
    messages = [
        {"role": "user", "content": "分析 AAPL"},
        {"role": "assistant", "content": "一些推理文字"},
    ]

    compacted, records = _compact_seen_tool_results(messages, {})

    assert compacted == messages
    assert records == []


def test_cache_control_is_stripped_from_compacted_block():
    """压缩后的占位文字很短，留着cache_control字段没有意义，反而可能造成
    误导（看起来像是在缓存一个大块内容）。"""
    messages = [_tool_result_message("1", _OVER_THRESHOLD, cache_control=True)]

    compacted, _ = _compact_seen_tool_results(messages, {"1": "get_filing_text"})

    assert "cache_control" not in compacted[0]["content"][0]


def test_multiple_blocks_in_same_message_compacted_independently():
    """同一轮里模型并行调用了好几个工具，其中只有一个返回值很大——只压缩
    那一个，其它保持原样。"""
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "1", "content": _OVER_THRESHOLD, "is_error": False},
                {"type": "tool_result", "tool_use_id": "2", "content": _UNDER_THRESHOLD, "is_error": False},
            ],
        }
    ]

    compacted, records = _compact_seen_tool_results(
        messages, {"1": "get_filing_text", "2": "get_filing_text"}
    )

    blocks = {b["tool_use_id"]: b["content"] for b in compacted[0]["content"]}
    assert blocks["1"] != _OVER_THRESHOLD
    assert blocks["2"] == _UNDER_THRESHOLD
    assert [r["tool_use_id"] for r in records] == ["1"]


def test_exempt_tool_is_never_compacted_even_when_oversized():
    """结构化数据源（财务指标/板块位置等）是简报数字断言的权威依据，即使
    哪天体积意外变大也不该被压缩——现在天然体积小只是巧合，白名单才是真正
    的保护。"""
    messages = [_tool_result_message("1", _OVER_THRESHOLD)]

    compacted, records = _compact_seen_tool_results(messages, {"1": "get_financials"})

    assert compacted[0]["content"][0]["content"] == _OVER_THRESHOLD
    assert records == []


def test_unknown_tool_use_id_falls_back_to_compacting():
    """tool_call_names里查不到对应的工具名（理论上不该发生，但作为fail-safe
    行为要明确）时，不能默认当成"豁免工具"放过——找不到身份信息时按最保守的
    方式处理，也就是仍然按体积判断是否压缩，不能因为查不到名字就意外获得
    豁免。"""
    messages = [_tool_result_message("unknown-id", _OVER_THRESHOLD)]

    compacted, records = _compact_seen_tool_results(messages, {})

    assert compacted[0]["content"][0]["content"] != _OVER_THRESHOLD
    assert records[0]["tool_name"] is None


def test_compaction_log_writes_expected_fields(tmp_path, monkeypatch):
    log_path = tmp_path / "context_compaction.jsonl"
    monkeypatch.setattr(loop_module, "CONTEXT_COMPACTION_LOG_PATH", log_path)

    _append_compaction_log(
        "AMD", 2, [{"tool_use_id": "1", "tool_name": "get_filing_text", "original_length": 394877}]
    )

    lines = log_path.read_text().strip().splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["ticker"] == "AMD"
    assert record["turn"] == 2
    assert record["tool_name"] == "get_filing_text"
    assert record["original_length"] == 394877
    assert "timestamp" in record


def test_compaction_log_no_op_when_no_records(tmp_path, monkeypatch):
    log_path = tmp_path / "context_compaction.jsonl"
    monkeypatch.setattr(loop_module, "CONTEXT_COMPACTION_LOG_PATH", log_path)

    _append_compaction_log("AMD", 2, [])

    assert not log_path.exists()


class _FaithfulFakeLLMClient:
    """跟真实 AnthropicCompatibleClient 的 append_tool_results/
    append_assistant_turn 结构一致（tool_result 是带 type/tool_use_id/
    content 字段的 block 列表），不是其它测试文件里那种把 content 直接
    塞成字符串 "fake" 的简化版——压缩逻辑要处理的正是这个真实结构，用
    简化版假客户端测不出来这条链路真的接上了。
    """

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
        return [*messages, {"role": "assistant", "content": "fake-assistant"}]

    def append_tool_results(self, messages, tool_calls, results):
        content = [
            {
                "type": "tool_result",
                "tool_use_id": r.tool_call_id,
                "content": r.content,
                "is_error": r.is_error,
            }
            for r in results
        ]
        return [*messages, {"role": "user", "content": content}]


def _find_tool_result_content(messages: list[dict], tool_use_id: str) -> str | None:
    for message in messages:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if isinstance(block, dict) and block.get("tool_use_id") == tool_use_id:
                return block["content"]
    return None


def _three_turn_responses(huge_tool_name: str) -> list[LLMResponse]:
    return [
        LLMResponse(
            stop_reason="tool_use",
            text=None,
            tool_calls=[ToolCall(id="huge", name=huge_tool_name, input={"ticker": "AAPL"})],
            raw=None,
        ),
        LLMResponse(
            stop_reason="tool_use",
            text=None,
            tool_calls=[ToolCall(id="verify", name="verify_number", input={"ticker": "AAPL"})],
            raw=None,
        ),
        LLMResponse(stop_reason="end_turn", text="就到这里，不含结论标签", tool_calls=[], raw=None),
    ]


def test_huge_filing_text_result_is_compacted_from_the_second_subsequent_request_onward():
    """端到端验证：第一次把大结果发给模型时必须是完整原文（模型要能真正
    读到内容），从下一次请求开始就该被压缩掉，不再重复携带。"""
    huge_filing_text = "真实财报原文内容" * 1000  # 远超压缩阈值
    responses = _three_turn_responses("get_filing_text")
    fake = _FaithfulFakeLLMClient(responses)

    async def fake_execute_tool(name, tool_input):
        if name == "get_filing_text":
            return (huge_filing_text, False)
        return ("{}", False)

    with (
        mock.patch("app.services.agent.loop.get_llm_client", return_value=fake),
        mock.patch("app.services.agent.loop.execute_tool", new=fake_execute_tool),
    ):
        asyncio.run(run_agent_loop("AAPL"))

    assert fake.calls == 3
    # 第2次请求（received_messages[1]）：模型第一次看到这个工具结果，必须是完整原文
    assert _find_tool_result_content(fake.received_messages[1], "huge") == huge_filing_text
    # 第3次请求（received_messages[2]）：已经被压缩，不再是完整原文
    third_request_content = _find_tool_result_content(fake.received_messages[2], "huge")
    assert third_request_content is not None
    assert third_request_content != huge_filing_text
    assert len(third_request_content) < len(huge_filing_text)


def test_small_tool_results_never_get_compacted_across_the_whole_run():
    """普通大小的工具结果（比如get_financials的JSON）不该被压缩逻辑误伤，
    全程原样保留。"""
    responses = [
        LLMResponse(
            stop_reason="tool_use",
            text=None,
            tool_calls=[ToolCall(id="fin", name="get_financials", input={"ticker": "AAPL"})],
            raw=None,
        ),
        LLMResponse(
            stop_reason="tool_use",
            text=None,
            tool_calls=[ToolCall(id="verify", name="verify_number", input={"ticker": "AAPL"})],
            raw=None,
        ),
        LLMResponse(stop_reason="end_turn", text="就到这里，不含结论标签", tool_calls=[], raw=None),
    ]
    fake = _FaithfulFakeLLMClient(responses)

    async def fake_execute_tool(name, tool_input):
        return ('{"revenue": 950000000}', False)

    with (
        mock.patch("app.services.agent.loop.get_llm_client", return_value=fake),
        mock.patch("app.services.agent.loop.execute_tool", new=fake_execute_tool),
    ):
        asyncio.run(run_agent_loop("AAPL"))

    third_request_content = _find_tool_result_content(fake.received_messages[2], "fin")
    assert third_request_content == '{"revenue": 950000000}'


def test_exempt_tool_large_result_survives_compaction_end_to_end():
    """即使get_financials哪天真的返回了一份体积很大的数据，也不该在后续
    轮次里被压缩掉——白名单保护要在真实Loop运行里生效，不只是单测里生效。"""
    huge_financials_payload = json.dumps({"revenue": 950000000, "note": "x" * 6000})
    responses = _three_turn_responses("get_financials")
    fake = _FaithfulFakeLLMClient(responses)

    async def fake_execute_tool(name, tool_input):
        if name == "get_financials":
            return (huge_financials_payload, False)
        return ("{}", False)

    with (
        mock.patch("app.services.agent.loop.get_llm_client", return_value=fake),
        mock.patch("app.services.agent.loop.execute_tool", new=fake_execute_tool),
    ):
        asyncio.run(run_agent_loop("AAPL"))

    third_request_content = _find_tool_result_content(fake.received_messages[2], "huge")
    assert third_request_content == huge_financials_payload


def test_enable_compaction_false_disables_compaction_entirely():
    """给eval脚本的--no-compaction开关用：显式关闭时，即使是非豁免的大结果
    也完全不压缩，行为等价于压缩机制上线之前。"""
    huge_filing_text = "真实财报原文内容" * 1000
    responses = _three_turn_responses("get_filing_text")
    fake = _FaithfulFakeLLMClient(responses)

    async def fake_execute_tool(name, tool_input):
        if name == "get_filing_text":
            return (huge_filing_text, False)
        return ("{}", False)

    with (
        mock.patch("app.services.agent.loop.get_llm_client", return_value=fake),
        mock.patch("app.services.agent.loop.execute_tool", new=fake_execute_tool),
    ):
        asyncio.run(run_agent_loop("AAPL", enable_compaction=False))

    third_request_content = _find_tool_result_content(fake.received_messages[2], "huge")
    assert third_request_content == huge_filing_text


def test_compaction_writes_audit_log_entries(tmp_path, monkeypatch):
    """压缩审计日志要在真实Loop运行里被正确写入——不只是_append_compaction_log
    这个函数本身单测过，还要验证run_agent_loop确实在正确的时机、带着正确的
    ticker/turn调用了它。"""
    log_path = tmp_path / "context_compaction.jsonl"
    monkeypatch.setattr(loop_module, "CONTEXT_COMPACTION_LOG_PATH", log_path)

    huge_filing_text = "真实财报原文内容" * 1000
    responses = _three_turn_responses("get_filing_text")
    fake = _FaithfulFakeLLMClient(responses)

    async def fake_execute_tool(name, tool_input):
        if name == "get_filing_text":
            return (huge_filing_text, False)
        return ("{}", False)

    with (
        mock.patch("app.services.agent.loop.get_llm_client", return_value=fake),
        mock.patch("app.services.agent.loop.execute_tool", new=fake_execute_tool),
    ):
        asyncio.run(run_agent_loop("AMD"))

    lines = log_path.read_text().strip().splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["ticker"] == "AMD"
    assert record["tool_name"] == "get_filing_text"
    assert record["original_length"] == len(huge_filing_text)
