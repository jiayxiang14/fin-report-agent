from pydantic import BaseModel


class PriceReactionResponse(BaseModel):
    ticker: str
    has_data: bool
    earnings_filing_date: str | None = None  # 触发这次反应窗口的8-K申报日期
    pre_close: float | None = None
    post_close: float | None = None
    price_change_pct: float | None = None
    post_volume: float | None = None
    avg_volume_baseline: float | None = None  # 反应窗口之前20个交易日的平均成交量
    volume_ratio: float | None = None  # post_volume / avg_volume_baseline
    follow_through_window_days: int | None = None  # 回看窗口交易日数
    extreme_close_in_window: float | None = None  # 窗口内出现的极值收盘价（涨势看最低、跌势看最高）
    reversal_pct_of_initial_move: float | None = None  # 极值相对初始反应的回吐比例，可能>100%（说明反转）
    reversal_day_volume_ratio: float | None = None  # 出现极值那天的成交量 / avg_volume_baseline
    note: str | None = None
