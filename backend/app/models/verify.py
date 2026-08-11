from pydantic import BaseModel


class VerifyNumberResponse(BaseModel):
    ticker: str
    metric: str
    period: str  # "annual" / "quarterly"
    claimed_value: float
    actual_value: float | None
    matches: bool | None  # None 表示没能核实（指标不存在/数据缺失），不是"不匹配"
    difference_pct: float | None = None
    note: str | None = None
