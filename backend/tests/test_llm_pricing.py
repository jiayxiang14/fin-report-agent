import pytest

from app.services.agent.llm_client import LLMUsage
from app.services.agent.llm_pricing import estimate_cost_usd


def test_known_model_sums_all_four_token_categories():
    usage = LLMUsage(
        input_tokens=1_000_000,
        output_tokens=1_000_000,
        cache_creation_input_tokens=1_000_000,
        cache_read_input_tokens=1_000_000,
    )
    cost = estimate_cost_usd("claude-sonnet-5", usage)
    assert cost == pytest.approx(3.00 + 15.00 + 3.75 + 0.30)


def test_unknown_model_returns_none_not_zero():
    """算不出来的成本不能悄悄展示成0，那样看起来像"这次运行免费" ——调用方
    要能区分"真的不花钱"和"这个模型没在价格表里，成本未知"。"""
    usage = LLMUsage(input_tokens=100, output_tokens=100)
    assert estimate_cost_usd("some-future-model", usage) is None


def test_zero_usage_on_known_model_is_zero_cost_not_none():
    usage = LLMUsage(input_tokens=0, output_tokens=0)
    assert estimate_cost_usd("deepseek-v4-flash", usage) == 0.0
