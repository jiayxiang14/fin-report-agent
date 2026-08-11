"""实际EPS vs 分析师一致预期EPS的超预期/低于预期判断。

跟 get_price_reaction 回答的是不同的问题：price_reaction 告诉Agent"市场怎么反应"，
这里告诉Agent"财报数字本身相对预期是好是坏"——两者放一起看才能判断"市场反应是否
跟数字本身相符"，这也是为什么 retrieval.md 里把这两个工具放在一起讲。

verdict 完全是代码按 surprise 的正负号计算，不经过LLM判断，符合项目"确定性数字用
代码算"的原则。
"""

import httpx

from app.models.earnings_surprise import EarningsSurpriseResponse
from app.services.alpha_vantage_client import AlphaVantageClientError, fetch_json

SURPRISE_EPSILON = 0.01  # |surprise| 小于这个阈值算"符合预期"，避免浮点误差被判成"超预期0.0001"


class EarningsSurpriseError(Exception):
    pass


def _parse_float(value: str | None) -> float | None:
    if value is None or value == "None":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _classify_verdict(surprise: float | None) -> str | None:
    if surprise is None:
        return None
    if surprise > SURPRISE_EPSILON:
        return "超预期"
    if surprise < -SURPRISE_EPSILON:
        return "低于预期"
    return "符合预期"


async def get_earnings_surprise(ticker: str) -> EarningsSurpriseResponse:
    ticker = ticker.upper()

    async with httpx.AsyncClient(timeout=15.0) as client:
        try:
            payload = await fetch_json("EARNINGS", ticker, client)
        except AlphaVantageClientError as exc:
            raise EarningsSurpriseError(str(exc)) from exc

    quarterly = payload.get("quarterlyEarnings") or []
    if not quarterly:
        return EarningsSurpriseResponse(
            ticker=ticker,
            has_data=False,
            note=f"Alpha Vantage 没有 '{ticker}' 的季度盈利预期数据",
        )

    # quarterlyEarnings 按 Alpha Vantage 的约定是按 fiscalDateEnding 降序排列的，
    # 但不依赖这个顺序假设——显式按日期取最新一条，避免上游哪天悄悄改了排序
    latest = max(quarterly, key=lambda q: q.get("fiscalDateEnding", ""))

    surprise = _parse_float(latest.get("surprise"))

    return EarningsSurpriseResponse(
        ticker=ticker,
        has_data=True,
        fiscal_date_ending=latest.get("fiscalDateEnding"),
        reported_date=latest.get("reportedDate"),
        reported_eps=_parse_float(latest.get("reportedEPS")),
        estimated_eps=_parse_float(latest.get("estimatedEPS")),
        surprise=surprise,
        surprise_percentage=_parse_float(latest.get("surprisePercentage")),
        verdict=_classify_verdict(surprise),
    )
