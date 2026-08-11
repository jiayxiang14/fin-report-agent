"""同行对比：用 Stage 1 已经建好的预设板块映射表找同板块的其他公司，
对比营收/净利润两个核心指标。板块归属、指标数字全是确定性代码，不经过LLM。
"""

import httpx

from app.models.peer import PeerComparisonResponse, PeerFinancialSnapshot
from app.services.sec_client import SecClientError
from app.services.sec_edgar import get_financials
from app.services.sector_rotation import PRESET_TICKER_SECTOR_MAP, SECTOR_ETFS

MAX_PEERS = 3


def find_peers(ticker: str, max_peers: int = MAX_PEERS) -> list[str]:
    """在预设股票池里找同一个板块ETF下的其他ticker，按字母序取前N个。"""
    ticker = ticker.upper()
    sector_etf = PRESET_TICKER_SECTOR_MAP.get(ticker)
    if sector_etf is None:
        return []
    peers = sorted(
        t for t, etf in PRESET_TICKER_SECTOR_MAP.items() if etf == sector_etf and t != ticker
    )
    return peers[:max_peers]


async def get_peer_comparison(ticker: str) -> PeerComparisonResponse:
    ticker = ticker.upper()
    sector_etf = PRESET_TICKER_SECTOR_MAP.get(ticker)

    if sector_etf is None:
        return PeerComparisonResponse(
            ticker=ticker,
            sector_etf=None,
            sector_name=None,
            peers=[],
            note=f"'{ticker}' 不在预设股票池内，无法做同行对比（MVP阶段用小规模预设池验证）",
        )

    peer_tickers = find_peers(ticker)
    peers: list[PeerFinancialSnapshot] = []

    for peer_ticker in peer_tickers:
        try:
            fin = await get_financials(peer_ticker)
        except (SecClientError, httpx.HTTPError):
            # 单个同行拉取失败不影响整体对比，跳过即可。这里故意catch得比
            # SecEdgarError更宽——fetch_company_facts对非404状态码只是单纯
            # raise_for_status()，不会包装成SecEdgarError，只catch窄的类型会
            # 导致某一个同行的异常状况拖垮整个对比，而不是"少这一个"
            continue

        revenue = fin.metrics.get("revenue")
        net_income = fin.metrics.get("net_income")

        peers.append(
            PeerFinancialSnapshot(
                ticker=peer_ticker,
                entity_name=fin.entity_name,
                revenue=revenue.latest_annual.val if revenue and revenue.latest_annual else None,
                revenue_yoy_pct=(
                    revenue.latest_annual.yoy_change_pct if revenue and revenue.latest_annual else None
                ),
                net_income=(
                    net_income.latest_annual.val if net_income and net_income.latest_annual else None
                ),
                net_income_yoy_pct=(
                    net_income.latest_annual.yoy_change_pct
                    if net_income and net_income.latest_annual
                    else None
                ),
            )
        )

    return PeerComparisonResponse(
        ticker=ticker,
        sector_etf=sector_etf,
        sector_name=SECTOR_ETFS.get(sector_etf),
        peers=peers,
    )
