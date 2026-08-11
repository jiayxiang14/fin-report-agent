"""LLM 调用层的超时配置 + 连接重试。

之前完全没传 timeout（等于用SDK默认的10分钟），也没有任何重试——anthropic SDK
自带的 max_retries 只认HTTP状态码（408/409/429/5xx），网络层直接断连/超时
（APITimeoutError/APIConnectionError）不在那套重试范围内，会直接抛给调用方，
让已经跑了好几轮、真花了钱的 Agent Loop 因为一次网络抖动整体作废。
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import anthropic
import pytest

from app.services.agent.llm_client import (
    CONNECTION_RETRY_ATTEMPTS,
    REQUEST_TIMEOUT_SECONDS,
    AnthropicCompatibleClient,
)


def _fake_message(text: str = "ok") -> MagicMock:
    message = MagicMock()
    message.content = [MagicMock(type="text", text=text)]
    message.stop_reason = "end_turn"
    return message


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
    client._client.messages.create = AsyncMock(side_effect=[_timeout_error(), _fake_message("恢复成功")])

    with patch("app.services.agent.llm_client.asyncio.sleep", new=AsyncMock()):
        response = asyncio.run(client.create_message(system="s", messages=[], tools=[]))

    assert response.text == "恢复成功"
    assert client._client.messages.create.call_count == 2


def test_gives_up_after_exhausting_connection_retry_budget():
    client = AnthropicCompatibleClient(
        base_url="https://example.com", api_key="key", model="m", use_cache_control=False
    )
    # 预算是 CONNECTION_RETRY_ATTEMPTS 次重试，也就是总共 CONNECTION_RETRY_ATTEMPTS+1
    # 次尝试，全部失败才应该真正把异常抛给调用方
    client._client.messages.create = AsyncMock(
        side_effect=[_connection_error() for _ in range(CONNECTION_RETRY_ATTEMPTS + 1)]
    )

    with patch("app.services.agent.llm_client.asyncio.sleep", new=AsyncMock()):
        with pytest.raises(anthropic.APIConnectionError):
            asyncio.run(client.create_message(system="s", messages=[], tools=[]))

    assert client._client.messages.create.call_count == CONNECTION_RETRY_ATTEMPTS + 1


def test_does_not_retry_non_connection_errors():
    """比如请求格式错误这类重试了也不会成功的错误，不该白白多等——直接抛出去，
    不占用连接重试预算。"""
    client = AnthropicCompatibleClient(
        base_url="https://example.com", api_key="key", model="m", use_cache_control=False
    )
    bad_request = anthropic.BadRequestError(
        message="bad request", response=MagicMock(status_code=400), body=None
    )
    client._client.messages.create = AsyncMock(side_effect=bad_request)

    with pytest.raises(anthropic.BadRequestError):
        asyncio.run(client.create_message(system="s", messages=[], tools=[]))

    assert client._client.messages.create.call_count == 1
