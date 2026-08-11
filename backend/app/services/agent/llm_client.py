"""LLM 调用适配层。

`loop.py` 只认这个文件里的 `LLMResponse`/`ToolCall` 数据结构和
`create_message`/`append_assistant_turn`/`append_tool_results` 三个方法，
不知道背后具体是哪家供应商——换供应商只改 `get_llm_client()` 的工厂逻辑，
不用碰 Loop 本身。

DeepSeek 提供了 Anthropic 格式兼容的端点（`base_url` 指向
`https://api.deepseek.com/anthropic`），跟以后测试用的原生 Claude API
走的是同一套 tool_use / stop_reason 协议，所以两个供应商可以共用同一份
客户端实现，只是 base_url / api_key / model 不同——这比在 OpenAI 格式
和 Anthropic 格式之间做一层转换要简单得多。
"""

import asyncio
from dataclasses import dataclass
from typing import cast

import anthropic
from anthropic.types import Message

from app.core.config import settings

DEEPSEEK_BASE_URL = "https://api.deepseek.com/anthropic"
DEEPSEEK_DEFAULT_MODEL = "deepseek-v4-flash"
CLAUDE_BASE_URL = "https://api.anthropic.com"
CLAUDE_DEFAULT_MODEL = "claude-sonnet-5"

MAX_TOKENS = 8192
# 超过这个字符数的 tool_result（典型情况是 get_filing_text 返回的财报全文）
# 才值得加 cache_control——小的工具结果加缓存标记只有写入成本，没有收益
CACHEABLE_CONTENT_THRESHOLD = 5000

# 之前完全没传 timeout，等于用 SDK 默认的10分钟——对一个交互式工具来说太宽松，
# 真卡住了用户要等10分钟才看到报错。财报全文塞进上下文后单轮确实可能慢，
# 120秒是在"给大上下文留够时间"和"卡住了用户能较快看到报错"之间的折中
REQUEST_TIMEOUT_SECONDS = 120.0

# anthropic SDK 自带 max_retries=2，但那套重试只认HTTP状态码（408/409/429/5xx，
# 见 anthropic._base_client.BaseClient._should_retry）——网络层直接断连/超时
# （httpx.TimeoutException 等，SDK 包装成 APITimeoutError/APIConnectionError）
# 是在拿到任何响应之前就失败的，不在那套retry覆盖范围内，会直接抛给调用方，
# 让整个 run_agent_loop（已经跑了好几轮、真花了钱）因为一次网络抖动整体作废。
# 这里单独补这一类的重试，不跟SDK内置重试重复处理已经有状态码的情况。
CONNECTION_RETRY_ATTEMPTS = 2
CONNECTION_RETRY_BACKOFF_SECONDS = 2.0

_STOP_REASON_MAP = {
    "end_turn": "end_turn",
    "tool_use": "tool_use",
    "max_tokens": "max_tokens",
    "refusal": "refusal",
    "stop_sequence": "end_turn",
    "pause_turn": "end_turn",  # 我们没有声明任何server-side工具，理论上不会真的出现
}


@dataclass
class ToolCall:
    id: str
    name: str
    input: dict


@dataclass
class ToolResult:
    tool_call_id: str
    content: str
    is_error: bool = False


@dataclass
class LLMResponse:
    stop_reason: str  # 归一化后的值，见 _STOP_REASON_MAP
    text: str | None
    tool_calls: list[ToolCall]
    raw: Message  # 原始响应对象，append_assistant_turn 靠它把上一轮回复原样喂回下一轮 messages


class AnthropicCompatibleClient:
    """同时服务 DeepSeek（走 /anthropic 兼容端点）和原生 Claude API。

    用的是 SDK 自带的 `AsyncAnthropic`，不是拿线程池硬包同步客户端——
    `create_message` 是在 `loop.py` 的 `async def run_agent_loop` 里被直接
    调用的，如果内部是同步阻塞调用，每一次LLM往返都会真正卡住整个事件循环，
    这跟 Stage 4 想做的"Agent推理过程可异步观察"直接冲突。
    """

    def __init__(self, base_url: str, api_key: str, model: str, use_cache_control: bool):
        self._client = anthropic.AsyncAnthropic(
            base_url=base_url, api_key=api_key, timeout=REQUEST_TIMEOUT_SECONDS
        )
        self._model = model
        self._use_cache_control = use_cache_control

    async def create_message(
        self,
        system: str,
        messages: list[dict],
        tools: list[dict],
        temperature: float | None = None,
    ) -> LLMResponse:
        system_param: str | list[dict]
        if self._use_cache_control:
            system_param = [
                {"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}
            ]
        else:
            system_param = system

        kwargs: dict = {
            "model": self._model,
            "max_tokens": MAX_TOKENS,
            "system": system_param,
            "messages": messages,
            "tools": tools,
        }
        # 只有显式传入时才带这个参数——默认不传等于沿用API默认值，不改变
        # Best-of-N之前所有调用方（单次Agent Loop）已验证过的行为
        if temperature is not None:
            kwargs["temperature"] = temperature

        response = await self._create_with_connection_retry(kwargs)

        text_blocks = [block.text for block in response.content if block.type == "text"]
        text = "\n".join(text_blocks) if text_blocks else None
        tool_calls = [
            ToolCall(id=block.id, name=block.name, input=block.input)
            for block in response.content
            if block.type == "tool_use"
        ]
        stop_reason = _STOP_REASON_MAP.get(response.stop_reason, "error") if response.stop_reason else "error"

        return LLMResponse(stop_reason=stop_reason, text=text, tool_calls=tool_calls, raw=response)

    async def _create_with_connection_retry(self, kwargs: dict) -> Message:
        for attempt in range(CONNECTION_RETRY_ATTEMPTS + 1):
            try:
                return cast(Message, await self._client.messages.create(**kwargs))
            except (anthropic.APITimeoutError, anthropic.APIConnectionError):
                if attempt == CONNECTION_RETRY_ATTEMPTS:
                    raise
                await asyncio.sleep(CONNECTION_RETRY_BACKOFF_SECONDS * (attempt + 1))
        raise AssertionError("unreachable")  # for循环要么return要么raise，mypy需要这行收尾

    def append_assistant_turn(self, messages: list[dict], response: LLMResponse) -> list[dict]:
        return [*messages, {"role": "assistant", "content": response.raw.content}]

    def append_tool_results(
        self, messages: list[dict], tool_calls: list[ToolCall], results: list[ToolResult]
    ) -> list[dict]:
        content = []
        for result in results:
            block: dict = {
                "type": "tool_result",
                "tool_use_id": result.tool_call_id,
                "content": result.content,
                "is_error": result.is_error,
            }
            if self._use_cache_control and len(result.content) > CACHEABLE_CONTENT_THRESHOLD:
                block["cache_control"] = {"type": "ephemeral"}
            content.append(block)
        return [*messages, {"role": "user", "content": content}]


def get_llm_client() -> AnthropicCompatibleClient:
    if settings.llm_provider == "claude":
        return AnthropicCompatibleClient(
            base_url=CLAUDE_BASE_URL,
            api_key=settings.anthropic_api_key,
            model=settings.llm_model or CLAUDE_DEFAULT_MODEL,
            use_cache_control=True,
        )
    if settings.llm_provider == "deepseek":
        return AnthropicCompatibleClient(
            base_url=DEEPSEEK_BASE_URL,
            api_key=settings.deepseek_api_key,
            model=settings.llm_model or DEEPSEEK_DEFAULT_MODEL,
            # DeepSeek 是自动前缀缓存，不需要/未验证是否接受 cache_control 字段，先不发送
            use_cache_control=False,
        )
    raise ValueError(f"不支持的 LLM_PROVIDER: {settings.llm_provider!r}，只能是 'deepseek' 或 'claude'")
