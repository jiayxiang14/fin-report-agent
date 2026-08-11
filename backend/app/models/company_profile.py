from pydantic import BaseModel


class CompanyProfileResponse(BaseModel):
    ticker: str
    has_data: bool
    name: str | None = None
    description: str | None = None
    market_cap: float | None = None
    sic_description: str | None = None  # Polygon/SEC官方的细分行业分类，比11个SPDR大类更细
    homepage_url: str | None = None
    total_employees: int | None = None
    adr_20d_pct: float | None = None  # 最近20个交易日平均日振幅，占收盘价百分比
    pe_ratio: float | None = None  # 最新收盘价 / 最新一期年度稀释EPS，不是严格TTM
    note: str | None = None
