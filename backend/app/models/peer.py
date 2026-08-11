from pydantic import BaseModel


class PeerFinancialSnapshot(BaseModel):
    ticker: str
    entity_name: str
    revenue: float | None
    revenue_yoy_pct: float | None
    net_income: float | None
    net_income_yoy_pct: float | None


class PeerComparisonResponse(BaseModel):
    ticker: str
    sector_etf: str | None
    sector_name: str | None
    peers: list[PeerFinancialSnapshot]
    note: str | None = None
