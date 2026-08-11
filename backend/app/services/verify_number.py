"""Agent 自我核查工具：重新拉一次 get_financials，把 Agent 报告里提到的具体
数字和真实数据源比对，返回是否吻合。比对逻辑本身是确定性代码，不经过LLM——
这是项目书要求的"自我核查机制"的落地工具。
"""

from app.models.verify import VerifyNumberResponse
from app.services.sec_edgar import METRIC_TAGS, SecEdgarError, get_financials

# free_cash_flow 不在 METRIC_TAGS 里（它是 operating_cash_flow - capital_expenditures 派生出来的，
# 没有对应的单一 us-gaap 标签），但 get_financials 返回的 metrics 字典里确实会有这个key，
# 所以补进可核实指标列表，不然 verify_number 拒绝核实一个实际存在的数字
VALID_METRICS = tuple(METRIC_TAGS.keys()) + ("free_cash_flow",)
VALID_PERIODS = ("annual", "quarterly")
DEFAULT_TOLERANCE_PCT = 1.0  # 允许1%相对误差——Agent引用的数字可能是四舍五入过的


async def verify_number(
    ticker: str, metric: str, claimed_value: float, period: str = "annual"
) -> VerifyNumberResponse:
    ticker = ticker.upper()

    if metric not in VALID_METRICS:
        return VerifyNumberResponse(
            ticker=ticker,
            metric=metric,
            period=period,
            claimed_value=claimed_value,
            actual_value=None,
            matches=None,
            note=f"不支持的指标 '{metric}'，可核实的指标只有：{', '.join(VALID_METRICS)}",
        )
    if period not in VALID_PERIODS:
        return VerifyNumberResponse(
            ticker=ticker,
            metric=metric,
            period=period,
            claimed_value=claimed_value,
            actual_value=None,
            matches=None,
            note=f"不支持的周期 '{period}'，只能是 'annual' 或 'quarterly'",
        )

    try:
        financials = await get_financials(ticker)
    except SecEdgarError as exc:
        return VerifyNumberResponse(
            ticker=ticker,
            metric=metric,
            period=period,
            claimed_value=claimed_value,
            actual_value=None,
            matches=None,
            note=f"核实时拉取财务数据失败：{exc}",
        )

    financial_metric = financials.metrics.get(metric)
    point = None
    if financial_metric is not None:
        point = (
            financial_metric.latest_annual
            if period == "annual"
            else financial_metric.latest_quarterly
        )

    if point is None:
        return VerifyNumberResponse(
            ticker=ticker,
            metric=metric,
            period=period,
            claimed_value=claimed_value,
            actual_value=None,
            matches=None,
            note=f"'{ticker}' 没有可用的 {metric}（{period}）数据，无法核实",
        )

    actual_value = point.val
    if actual_value == 0:
        matches = claimed_value == 0
        diff_pct = None
    else:
        diff_pct = abs(claimed_value - actual_value) / abs(actual_value) * 100
        matches = diff_pct <= DEFAULT_TOLERANCE_PCT

    return VerifyNumberResponse(
        ticker=ticker,
        metric=metric,
        period=period,
        claimed_value=claimed_value,
        actual_value=actual_value,
        matches=matches,
        difference_pct=diff_pct,
        note=f"数据截至 {point.end}（{point.form}）",
    )
