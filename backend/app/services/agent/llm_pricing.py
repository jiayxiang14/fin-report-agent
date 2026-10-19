"""模型级美元成本估算——回答"这次跑大概花了多少钱"，不是精确对账的计费系统。

价格表是手动维护的常量，不接任何供应商的实时计价API，会过时。两个已知的
简化，都是往"不低估成本"的方向取整，不是往"好看"的方向取整：
- DeepSeek 有按UTC时段浮动的高峰/非高峰折扣价，这里统一按未打折的标准价算。
- Claude Sonnet 5 当前有到 2026-08-31 的促销价，这里按促销结束后的标准价算。
用这个数字之前，先去供应商官网核对最新价格——它存在的意义是支撑"context
compaction/prompt caching到底省了多少钱"这类同一个模型内部的相对比较，
不是用来对账真实账单的。

模型名不在表里时返回 None（不是拿最相近的模型价格去凑），调用方要如实展示
"这个模型的成本未知"，不能把估算不出来的东西悄悄估算成 0。
"""

from app.services.agent.llm_client import LLMUsage

# 单位：每百万token的美元价格
USD_PER_MILLION_TOKENS: dict[str, dict[str, float]] = {
    "claude-sonnet-5": {"input": 3.00, "output": 15.00, "cache_write": 3.75, "cache_read": 0.30},
    "deepseek-v4-flash": {"input": 0.44, "output": 1.32, "cache_write": 0.44, "cache_read": 0.014},
    # 2026-10-19 按标准价录入（当时有促销价 $0.435/$0.87，这里按标准价算，见模块注释）；
    # 只用于评测里的审阅/评分角色，生成简报默认仍是flash
    "deepseek-v4-pro": {"input": 1.74, "output": 3.48, "cache_write": 1.74, "cache_read": 0.174},
}


def estimate_cost_usd(model: str, usage: LLMUsage) -> float | None:
    rates = USD_PER_MILLION_TOKENS.get(model)
    if rates is None:
        return None
    total = (
        usage.input_tokens * rates["input"]
        + usage.output_tokens * rates["output"]
        + usage.cache_creation_input_tokens * rates["cache_write"]
        + usage.cache_read_input_tokens * rates["cache_read"]
    )
    return total / 1_000_000
