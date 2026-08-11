"""细分主题板块轮动：AI基建/算力产业链上的10个细分主题，跟现有11个SPDR
大类板块（sector_rotation.py）是同一套RRG方法论（rrg.py），只是篮子更窄、
更贴近具体产业叙事。

背景：用户最初想要的是"某公司的上下游供应商/客户是谁、对哪些第三方股票
利好利空"，这跟之前明确拒绝过的"上下游产业链分析"是同一类请求——没有真实
数据源支撑供应链关系，只能靠LLM训练记忆去猜，而且对第三方股票下结论已经
越界到具体投资建议。这里做的是可落地的替代版本：几个细分产业主题的相对
强弱/动量位置，全部是真实公司的真实行情数据算出来的，不涉及任何"猜测公司
关系"的推测。第一版只给了4个主题当示例，用户反馈范围太窄，扩展到覆盖AI
算力产业链主要环节（芯片→存储/光互联→服务器/网络→数据中心→供电/散热）
的10个主题。

篮子里的ticker都是真实存在、当前交易的公司（不是凭训练记忆编的），扩展
这一批时额外用WebSearch核实过SMCI（2025年经历过Nasdaq退市风险，已确认
2026年仍在正常交易）、VRT（Vertiv，液冷这个细分是真实存在的AI基建叙事，
不是编的主题）。半导体/AI芯片这两个主题刻意分开：`SOXX`是覆盖面更广的
半导体ETF（含模拟芯片等非AI相关成分），"AI芯片/GPU"是只挑AI算力龙头的
更窄篮子，粒度不一样，不是重复。

`get_thematic_flow`同时是两个消费方共用的同一份函数：前端`GET
/api/thematic-flow`路由拿它做同步展示，Agent工具集里的`get_thematic_flow`
工具（`agent/tools.py`）也直接调用它——不是复制一份逻辑给Agent专用，跟
`get_financials`/`get_sector_position`这些"既是同步面板又是Agent工具"的
数据源是同一个模式，Agent自己判断这家公司跟AI算力产业链相不相关、要不要
调用这个工具，不是每次分析都强制查。
"""

import httpx
import pandas as pd

from app.models.sector import RrgHistoryPoint
from app.models.thematic_flow import ThematicFlowPosition, ThematicFlowResponse
from app.services.company_profile import CompanyProfileError, fetch_ticker_details
from app.services.polygon_client import PolygonClientError, fetch_daily_bars
from app.services.rrg import classify_quadrant, compute_rrg

BENCHMARK = "SPY"

THEMATIC_BASKETS: dict[str, list[str]] = {
    "半导体": ["SOXX"],  # iShares Semiconductor ETF，单一真实ETF，做法跟SECTOR_ETFS一致
    "AI芯片/GPU": ["NVDA", "AVGO", "AMD", "ASML"],  # 比SOXX更窄，只挑AI算力龙头
    "存储": ["MU", "WDC", "STX", "SNDK"],  # 美光/西数/希捷/闪迪（2025年从WDC分拆上市）
    "光模块": ["COHR", "LITE", "FN"],  # Coherent/Lumentum/Fabrinet
    "云计算": ["MSFT", "GOOGL", "AMZN", "META", "ORCL"],  # 超大规模数据中心运营商，AI资本开支主体
    "数据中心REITs": ["EQIX", "DLR"],  # Equinix/Digital Realty
    "网络设备": ["ANET", "CSCO"],  # Arista/Cisco，AI集群交换网络
    "服务器ODM": ["SMCI", "DELL"],  # Super Micro/Dell
    "电力": ["CEG", "VST", "GEV"],  # Constellation/Vistra/GE Vernova，AI数据中心供电相关
    "液冷散热": ["VRT"],  # Vertiv，AI数据中心液冷/散热的代表公司
}

# 10个主题彼此的产业链位置——固定分类，跟具体分析哪家公司完全无关，只是
# "这个主题在AI算力产业链里大致处于哪一环"这种公开、通用的产业常识分类：
# 上游=芯片/存储/光模块等硬件组件制造，中游=把组件整合成服务器/网络/数据中心
# 这类基础设施的建设与运营，下游=云计算这类实际部署AI服务、面向最终客户的
# 环节。这是简化的三层归类，不是精确的产业经济学模型，真实产业链里各环节
# 互相交织、边界不完全清晰——不涉及猜测任何具体公司是谁的供应商/客户。
CHAIN_POSITION: dict[str, str] = {
    "半导体": "上游",
    "AI芯片/GPU": "上游",
    "存储": "上游",
    "光模块": "上游",
    "服务器ODM": "中游",
    "网络设备": "中游",
    "数据中心REITs": "中游",
    "电力": "中游",
    "液冷散热": "中游",
    "云计算": "下游",
}

# SIC（美国政府官方行业分类代码）关键词 → 主题名，用来扩大matched_themes的
# 识别范围到不在THEMATIC_BASKETS手动名单里的公司。这个映射表是拿真实ticker
# 的sic_description核实过的（不是猜的格式），而且刻意只收录了SIC描述本身
# 在同一主题的多家公司之间高度一致、且不会跟其它主题混淆的5个主题——
# 半导体(NVDA/MU都是"SEMICONDUCTORS & RELATED DEVICES")、存储(WDC/STX都是
# "COMPUTER STORAGE DEVICES")、网络设备(ANET/CSCO都是"COMPUTER COMMUNICATIONS
# EQUIPMENT")、服务器ODM(SMCI/DELL都是"ELECTRONIC COMPUTERS")、电力(CEG是
# "ELECTRIC SERVICES")。另外5个主题（AI芯片/GPU、光模块、云计算、数据中心
# REITs、液冷散热）核实下来SIC本身就不可靠——比如光模块的COHR/LITE/FN三家
# 公司分别是完全不同的三个SIC代码，云计算的GOOGL/ORCL也不一致，数据中心REITs
# 的SIC就是通用"REAL ESTATE INVESTMENT TRUSTS"跟任何REIT都一样分不出来——
# 这5个主题不做SIC匹配，宁可覆盖面窄一点也不做不可靠的匹配，继续只依赖
# THEMATIC_BASKETS的成分股名单+Agent自己读业务描述判断。
SIC_KEYWORD_THEMES: dict[str, str] = {
    "SEMICONDUCTORS": "半导体",
    "COMPUTER STORAGE DEVICES": "存储",
    "COMPUTER COMMUNICATIONS EQUIPMENT": "网络设备",
    "ELECTRONIC COMPUTERS": "服务器ODM",
    "ELECTRIC SERVICES": "电力",
}

BASE_NOTE = "篮子为人工挑选的代表性公司，非官方指数成分，仅供参考"
HISTORY_TAIL = 90


class ThematicFlowError(Exception):
    pass


def _build_ticker_to_themes() -> dict[str, list[str]]:
    """反查表：某个ticker本身是哪些主题篮子的成分股——不用LLM去猜"NVDA跟
    AI芯片主题有没有关系"，篮子本来就是代码里写死的，反查一遍就是确定性
    的答案，跟`sector_rotation.PRESET_TICKER_SECTOR_MAP`是同一个思路。
    """
    mapping: dict[str, list[str]] = {}
    for theme_name, tickers in THEMATIC_BASKETS.items():
        for ticker in tickers:
            mapping.setdefault(ticker, []).append(theme_name)
    return mapping


TICKER_TO_THEMES: dict[str, list[str]] = _build_ticker_to_themes()


def _match_themes_by_sic(sic_description: str | None) -> list[str]:
    if not sic_description:
        return []
    upper_description = sic_description.upper()
    return [theme for keyword, theme in SIC_KEYWORD_THEMES.items() if keyword in upper_description]


async def _resolve_matched_themes(ticker: str, client: httpx.AsyncClient) -> tuple[list[str], list[str]]:
    """返回 (matched_themes, sic_matched_themes)——后者是matched_themes的
    子集，专指"篮子里没有这只票、但SIC行业分类命中了"这种情况。两者分开返回
    是因为SIC匹配是这次新加的机制，如果直接合并进同一个matched_themes，
    前端没法区分一个高亮到底是"本来就在篮子里"还是"SIC识别出来的"，这个新
    机制在UI上就等于隐身、测试的时候感知不到它到底做了什么。

    先查ticker本身是不是篮子成分股（100%确定性），再用SIC行业分类关键词
    补充匹配（覆盖面更广但只对5个SIC足够一致的主题生效，见SIC_KEYWORD_THEMES
    顶部注释）。SIC查询失败（比如Polygon查不到这个ticker）不影响篮子成分股
    的匹配结果，是软性降级不是硬错误——这跟company_profile.py处理pe_ratio
    获取失败是同一个哲学。
    """
    basket_matched = list(TICKER_TO_THEMES.get(ticker, []))

    try:
        details = await fetch_ticker_details(ticker, client)
    except CompanyProfileError:
        details = None

    sic_matched: list[str] = []
    if details:
        for theme in _match_themes_by_sic(details.get("sic_description")):
            if theme not in basket_matched:
                sic_matched.append(theme)

    return basket_matched + sic_matched, sic_matched


def _build_equal_weighted_index(closes_by_ticker: dict[str, pd.Series]) -> pd.Series:
    """把篮子里每只股票的收盘价在共同交易日对齐、各自归一化到起始日=100，
    再逐日取平均，合成一条"主题指数"序列。对单一ticker的篮子（比如半导体=SOXX）
    这个函数一样适用不需要特殊分支——单一序列归一化再"平均"数学上就是原序列的
    线性缩放，而RRG的z-score计算对线性缩放不敏感（缩放同时影响滚动均值和标准差，
    z-score结果不变）。
    """
    aligned = pd.concat(closes_by_ticker.values(), axis=1).dropna()
    normalized = aligned / aligned.iloc[0] * 100
    return normalized.mean(axis=1)


async def _fetch_theme_closes(tickers: list[str], client: httpx.AsyncClient) -> pd.Series | None:
    """单只股票拉取失败就跳过它，不让整个主题因为篮子里一只票有问题就报废；
    篮子里的股票全部拉取失败才返回None。"""
    closes_by_ticker: dict[str, pd.Series] = {}
    for ticker in tickers:
        try:
            bars = await fetch_daily_bars(ticker, client)
        except PolygonClientError:
            continue
        closes_by_ticker[ticker] = bars["close"]

    if not closes_by_ticker:
        return None
    return _build_equal_weighted_index(closes_by_ticker)


async def get_thematic_flow(ticker: str | None = None) -> ThematicFlowResponse:
    normalized_ticker = ticker.upper() if ticker else None

    async with httpx.AsyncClient(timeout=15.0) as client:
        if normalized_ticker:
            matched_themes, sic_matched_themes = await _resolve_matched_themes(normalized_ticker, client)
        else:
            matched_themes, sic_matched_themes = [], []

        try:
            benchmark_bars = await fetch_daily_bars(BENCHMARK, client)
        except PolygonClientError as exc:
            raise ThematicFlowError(str(exc)) from exc
        benchmark_closes = benchmark_bars["close"]

        themes: list[ThematicFlowPosition] = []
        unavailable_themes: list[str] = []

        for theme_name, tickers in THEMATIC_BASKETS.items():
            theme_closes = await _fetch_theme_closes(tickers, client)
            if theme_closes is None:
                unavailable_themes.append(theme_name)
                continue

            rrg = compute_rrg(theme_closes, benchmark_closes)
            if rrg.empty:
                unavailable_themes.append(theme_name)
                continue

            latest = rrg.iloc[-1]
            tail = rrg.iloc[-HISTORY_TAIL:]
            history = [
                RrgHistoryPoint(
                    date=idx.date().isoformat(),
                    rs_ratio=round(float(row["rs_ratio"]), 3),
                    rs_momentum=round(float(row["rs_momentum"]), 3),
                )
                for idx, row in tail.iterrows()
            ]

            themes.append(
                ThematicFlowPosition(
                    theme_name=theme_name,
                    chain_position=CHAIN_POSITION[theme_name],
                    constituent_tickers=tickers,
                    rs_ratio=round(float(latest["rs_ratio"]), 3),
                    rs_momentum=round(float(latest["rs_momentum"]), 3),
                    quadrant=classify_quadrant(latest["rs_ratio"], latest["rs_momentum"]),
                    history=history,
                )
            )

    as_of = themes[0].history[-1].date if themes and themes[0].history else ""
    note = BASE_NOTE
    if unavailable_themes:
        note += f"；以下主题因数据不足暂时无法展示：{'、'.join(unavailable_themes)}"

    return ThematicFlowResponse(
        ticker=normalized_ticker,
        matched_themes=matched_themes,
        sic_matched_themes=sic_matched_themes,
        benchmark=BENCHMARK,
        as_of=as_of,
        themes=themes,
        note=note,
    )
