from pydantic import BaseModel

from app.models.sector import RrgHistoryPoint


class ThematicFlowPosition(BaseModel):
    theme_name: str
    chain_position: str  # "上游"/"中游"/"下游"——10个主题彼此的产业链位置，
    # 固定分类，跟具体分析哪家公司无关，不是猜测具体供应商/客户关系
    constituent_tickers: list[str]  # 篮子里的真实公司ticker，前端展示"这个主题包含哪些公司"
    rs_ratio: float
    rs_momentum: float
    quadrant: str
    history: list[RrgHistoryPoint]


class ThematicFlowResponse(BaseModel):
    ticker: str | None  # 传了ticker才会有；不传就是纯展示全部主题，不做匹配
    matched_themes: list[str]  # 篮子成分股匹配 + SIC行业分类匹配的并集——
    # 不是LLM猜的，代码算出来的，跟matched_sector_etf是同一个模式
    sic_matched_themes: list[str]  # matched_themes的子集，专指"不在篮子里但
    # SIC官方行业分类命中了"的主题——单独暴露出来，前端才能区分展示"这是篮子
    # 成分股"还是"官方行业分类识别出来的"，不然SIC匹配这个机制在UI上等于隐身
    benchmark: str
    as_of: str
    themes: list[ThematicFlowPosition]
    source: str = "Polygon.io 日线行情，代码计算 RS-Ratio/RS-Momentum"
    note: str  # 篮子说明 + 数据不足暂时无法展示的主题，在service层拼好
