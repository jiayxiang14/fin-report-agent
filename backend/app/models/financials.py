from pydantic import BaseModel


class MetricPoint(BaseModel):
    """A single reported value for one financial metric."""

    end: str  # 期末日期，如 "2024-09-28"
    val: float
    fy: int | None = None
    fp: str | None = None  # "FY" / "Q1" / "Q2" / "Q3" / "Q4"
    form: str  # "10-K" / "10-Q"
    filed: str
    yoy_change_pct: float | None = None  # 代码计算的同比变化，不是模型估算


class FinancialMetric(BaseModel):
    tag: str  # XBRL us-gaap 标签名，如 "Revenues"
    label: str
    unit: str  # "USD" / "USD/shares" 等
    latest_annual: MetricPoint | None = None
    latest_quarterly: MetricPoint | None = None


class FinancialsResponse(BaseModel):
    ticker: str
    cik: str
    entity_name: str
    metrics: dict[str, FinancialMetric]
    source: str = "SEC EDGAR companyfacts"
    retrieved_at: str


class MetricHistory(BaseModel):
    label: str
    unit: str
    points: list[MetricPoint]  # 按时间正序（从早到晚）排列，方便前端直接画折线图


class FinancialsHistoryResponse(BaseModel):
    """多年历史趋势——只给前端同步展示浏览用，不接入Agent工具集（Agent走
    FinancialsResponse 的 latest_annual/latest_quarterly 就够判断"这一期怎么样"，
    完整多年历史没必要塞进每次工具调用的payload里增加token成本）。"""

    ticker: str
    cik: str
    entity_name: str
    history: dict[str, MetricHistory]
    source: str = "SEC EDGAR companyfacts"
    retrieved_at: str
