"""LLM 调用层的超时配置 + 连接重试。

之前完全没传 timeout（等于用SDK默认的10分钟），也没有任何重试——anthropic SDK
自带的 max_retries 只认HTTP状态码（408/409/429/5xx），网络层直接断连/超时
（APITimeoutError/APIConnectionError）不在那套重试范围内，会直接抛给调用方，
让已经跑了好几轮、真花了钱的 Agent Loop 因为一次网络抖动整体作废。
"""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import anthropic
import pytest

from app.core.config import settings
from app.services.agent.llm_client import (
    CLAUDE_BASE_URL,
    CONNECTION_RETRY_ATTEMPTS,
    DEEPSEEK_BASE_URL,
    REQUEST_TIMEOUT_SECONDS,
    AnthropicCompatibleClient,
    get_llm_client,
)


def _fake_message(text: str = "ok") -> MagicMock:
    message = MagicMock()
    message.content = [MagicMock(type="text", text=text)]
    message.stop_reason = "end_turn"
    message.usage = MagicMock(
        input_tokens=1,
        output_tokens=1,
        cache_creation_input_tokens=0,
        cache_read_input_tokens=0,
    )
    return message


class _FakeStream:
    """`client.messages.stream(**kwargs)`真实返回的是一个async context manager，
    进去之后可以`async for`拿增量事件，退出前调用`get_final_message()`拿完整
    响应——这里只模拟这层协议本身，不模拟真的逐token输出（那是
    test_agent_loop.py里`on_text_delta`那条测试该验证的事，这个文件只关心
    重试/超时这些传输层行为，不关心有没有真的产出delta）。"""

    def __init__(self, final_message: MagicMock):
        self._final_message = final_message

    async def __aenter__(self) -> "_FakeStream":
        return self

    async def __aexit__(self, *exc_info: object) -> bool:
        return False

    async def __aiter__(self):
        return
        yield  # pragma: no cover - 让这个方法成为异步生成器，但没有事件要产出

    async def get_final_message(self) -> MagicMock:
        return self._final_message


def _fake_stream(text: str = "ok") -> _FakeStream:
    return _FakeStream(_fake_message(text))


def _timeout_error() -> anthropic.APITimeoutError:
    return anthropic.APITimeoutError(request=MagicMock())


def _connection_error() -> anthropic.APIConnectionError:
    return anthropic.APIConnectionError(request=MagicMock())


def test_client_is_constructed_with_explicit_timeout():
    with patch("app.services.agent.llm_client.anthropic.AsyncAnthropic") as mock_ctor:
        AnthropicCompatibleClient(
            base_url="https://example.com", api_key="key", model="m", use_cache_control=False
        )
    assert mock_ctor.call_args.kwargs["timeout"] == REQUEST_TIMEOUT_SECONDS


def test_retries_and_recovers_from_transient_connection_timeout():
    client = AnthropicCompatibleClient(
        base_url="https://example.com", api_key="key", model="m", use_cache_control=False
    )
    client._client.messages.stream = MagicMock(side_effect=[_timeout_error(), _fake_stream("恢复成功")])

    with patch("app.services.agent.llm_client.asyncio.sleep", new=AsyncMock()):
        response = asyncio.run(client.create_message(system="s", messages=[], tools=[]))

    assert response.text == "恢复成功"
    assert client._client.messages.stream.call_count == 2


def test_gives_up_after_exhausting_connection_retry_budget():
    client = AnthropicCompatibleClient(
        base_url="https://example.com", api_key="key", model="m", use_cache_control=False
    )
    # 预算是 CONNECTION_RETRY_ATTEMPTS 次重试，也就是总共 CONNECTION_RETRY_ATTEMPTS+1
    # 次尝试，全部失败才应该真正把异常抛给调用方
    client._client.messages.stream = MagicMock(
        side_effect=[_connection_error() for _ in range(CONNECTION_RETRY_ATTEMPTS + 1)]
    )

    with patch("app.services.agent.llm_client.asyncio.sleep", new=AsyncMock()):
        with pytest.raises(anthropic.APIConnectionError):
            asyncio.run(client.create_message(system="s", messages=[], tools=[]))

    assert client._client.messages.stream.call_count == CONNECTION_RETRY_ATTEMPTS + 1


def test_does_not_retry_non_connection_errors():
    """比如请求格式错误这类重试了也不会成功的错误，不该白白多等——直接抛出去，
    不占用连接重试预算。"""
    client = AnthropicCompatibleClient(
        base_url="https://example.com", api_key="key", model="m", use_cache_control=False
    )
    bad_request = anthropic.BadRequestError(
        message="bad request", response=MagicMock(status_code=400), body=None
    )
    client._client.messages.stream = MagicMock(side_effect=bad_request)

    with pytest.raises(anthropic.BadRequestError):
        asyncio.run(client.create_message(system="s", messages=[], tools=[]))

    assert client._client.messages.stream.call_count == 1


def test_get_llm_client_defaults_to_settings_provider(monkeypatch):
    monkeypatch.setattr(settings, "llm_provider", "deepseek")
    monkeypatch.setattr(settings, "deepseek_api_key", "key")

    client = get_llm_client()

    assert client._client.base_url == DEEPSEEK_BASE_URL + "/"


def test_get_llm_client_explicit_provider_overrides_settings(monkeypatch):
    """Best-of-N 裁判可以走跟生成不同的供应商（减少自评偏好嫌疑）——这里验证
    显式传入的 provider 确实覆盖了 settings.llm_provider，不是被忽略。"""
    monkeypatch.setattr(settings, "llm_provider", "deepseek")
    monkeypatch.setattr(settings, "anthropic_api_key", "key")

    client = get_llm_client("claude")

    assert str(client._client.base_url) == CLAUDE_BASE_URL


def test_get_llm_client_rejects_unknown_provider():
    with pytest.raises(ValueError, match="不支持的 LLM_PROVIDER"):
        get_llm_client("bedrock")


def test_create_message_populates_model_and_usage_from_final_message():
    client = AnthropicCompatibleClient(
        base_url="https://example.com", api_key="key", model="m", use_cache_control=False
    )
    final = _fake_message("done")
    final.usage = MagicMock(
        input_tokens=42, output_tokens=7, cache_creation_input_tokens=3, cache_read_input_tokens=9
    )
    client._client.messages.stream = MagicMock(return_value=_FakeStream(final))

    response = asyncio.run(client.create_message(system="s", messages=[], tools=[]))

    assert response.model == "m"
    assert response.usage.input_tokens == 42
    assert response.usage.output_tokens == 7
    assert response.usage.cache_creation_input_tokens == 3
    assert response.usage.cache_read_input_tokens == 9


def test_create_message_forwards_text_deltas_to_callback():
    client = AnthropicCompatibleClient(
        base_url="https://example.com", api_key="key", model="m", use_cache_control=False
    )

    class _TextEventStream(_FakeStream):
        async def __aiter__(self):
            yield SimpleNamespace(type="text", text="Hel")
            yield SimpleNamespace(type="content_block_stop")  # 非text事件应该被忽略，不传给回调
            yield SimpleNamespace(type="text", text="lo")

    client._client.messages.stream = MagicMock(return_value=_TextEventStream(_fake_message("Hello")))
    deltas: list[str] = []

    async def on_text_delta(delta: str) -> None:
        deltas.append(delta)

    response = asyncio.run(
        client.create_message(system="s", messages=[], tools=[], on_text_delta=on_text_delta)
    )

    assert deltas == ["Hel", "lo"]
    assert response.text == "Hello"  # 回调只是旁路增量，最终结果仍然来自get_final_message()


def test_create_message_without_on_text_delta_still_works():
    """流式请求内部一直在开，只是没人读delta——on_text_delta=None（默认值）
    时不应该报错，行为要跟传了回调时的最终结果一致。"""
    client = AnthropicCompatibleClient(
        base_url="https://example.com", api_key="key", model="m", use_cache_control=False
    )

    class _TextEventStream(_FakeStream):
        async def __aiter__(self):
            yield SimpleNamespace(type="text", text="Hello")

    client._client.messages.stream = MagicMock(return_value=_TextEventStream(_fake_message("Hello")))

    response = asyncio.run(client.create_message(system="s", messages=[], tools=[]))

    assert response.text == "Hello"


def test_tool_results_marked_not_cacheable_skip_cache_control():
    """会被上下文压缩替换掉的大块结果不打cache_control：写入溢价付了也读不到。"""
    from app.services.agent.llm_client import ToolResult

    client = AnthropicCompatibleClient(base_url="https://example.com", api_key="key", model="m", use_cache_control=True)
    big = "x" * 6000
    messages = client.append_tool_results(
        [],
        [],
        [
            ToolResult(tool_call_id="keep", content=big),
            ToolResult(tool_call_id="compactable", content=big, cacheable=False),
            ToolResult(tool_call_id="small", content="{}"),
        ],
    )

    blocks = {b["tool_use_id"]: b for b in messages[-1]["content"]}
    assert blocks["keep"]["cache_control"] == {"type": "ephemeral"}
    assert "cache_control" not in blocks["compactable"]
    assert "cache_control" not in blocks["small"]


def test_effort_is_sent_only_when_explicitly_requested():
    """effort放在extra_body.output_config里；不传时请求体里不能出现这个字段。"""
    import asyncio
    from unittest.mock import AsyncMock

    client = AnthropicCompatibleClient(base_url="https://example.com", api_key="key", model="m", use_cache_control=False)
    captured = []

    async def fake_retry(kwargs, on_text_delta):
        captured.append(kwargs)
        raise RuntimeError("stop")

    client._create_with_connection_retry = AsyncMock(side_effect=fake_retry)
    for effort in (None, "low"):
        try:
            asyncio.run(client.create_message(system="s", messages=[], tools=[], effort=effort))
        except RuntimeError:
            pass
    assert "extra_body" not in captured[0]
    assert captured[1]["extra_body"] == {"output_config": {"effort": "low"}}
