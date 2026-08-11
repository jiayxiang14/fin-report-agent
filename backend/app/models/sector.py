from pydantic import BaseModel

Quadrant = str  # "leading" / "weakening" / "lagging" / "improving"


class RrgHistoryPoint(BaseModel):
    date: str
    rs_ratio: float
    rs_momentum: float


class SectorPosition(BaseModel):
    sector_etf: str  # 板块ETF代码，如 "XLK"
    sector_name: str
    rs_ratio: float
    rs_momentum: float
    quadrant: Quadrant
    history: list[RrgHistoryPoint]  # 近期轨迹，供走势小图/RRG尾迹使用


class SectorRotationResponse(BaseModel):
    ticker: str
    matched_sector_etf: str | None
    matched_sector_name: str | None
    benchmark: str
    as_of: str
    sectors: list[SectorPosition]
    source: str = "Polygon.io 日线行情，代码计算 RS-Ratio/RS-Momentum"
    note: str | None = None  # 例如 ticker 不在预设股票池时的说明
