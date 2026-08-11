from pydantic import BaseModel


class EarningsSurpriseResponse(BaseModel):
    ticker: str
    has_data: bool
    fiscal_date_ending: str | None = None
    reported_date: str | None = None
    reported_eps: float | None = None
    estimated_eps: float | None = None
    surprise: float | None = None
    surprise_percentage: float | None = None
    verdict: str | None = None  # "超预期" / "低于预期" / "符合预期"，代码按数值正负直接判断
    note: str = "数据来自 Alpha Vantage 分析师一致预期（非官方口径，仅供参考）"
