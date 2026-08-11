"""板块轮动位置计算（简化版 RRG / RS-Ratio·RS-Momentum 逻辑）。

行情数据来自 Polygon.io 的日线聚合接口；RS-Ratio、RS-Momentum、
象限归类全部是代码里的确定性数学计算，不经过 LLM。
"""

import httpx
import pandas as pd

from app.models.sector import RrgHistoryPoint, SectorPosition, SectorRotationResponse
from app.services.polygon_client import PolygonClientError, fetch_daily_bars
from app.services.rrg import classify_quadrant, compute_rrg

BENCHMARK = "SPY"

# 11个 SPDR 板块 ETF，业内做板块轮动/RRG 图最常用的板块代理
SECTOR_ETFS: dict[str, str] = {
    "XLK": "Technology",
    "XLF": "Financials",
    "XLV": "Health Care",
    "XLY": "Consumer Discretionary",
    "XLP": "Consumer Staples",
    "XLE": "Energy",
    "XLI": "Industrials",
    "XLB": "Materials",
    "XLU": "Utilities",
    "XLRE": "Real Estate",
    "XLC": "Communication Services",
}

# MVP阶段的预设股票池：只覆盖一批常见大盘股，不做通用的行业分类系统
# （项目书第九节路线图第1周："可以先用少量预设股票池验证"）
PRESET_TICKER_SECTOR_MAP: dict[str, str] = {
    # Technology
    "AAPL": "XLK", "MSFT": "XLK", "NVDA": "XLK", "AVGO": "XLK",
    "ORCL": "XLK", "CRM": "XLK", "ADBE": "XLK", "AMD": "XLK",
    # Financials
    "JPM": "XLF", "BAC": "XLF", "WFC": "XLF", "GS": "XLF", "MS": "XLF", "AXP": "XLF",
    # Health Care
    "JNJ": "XLV", "UNH": "XLV", "PFE": "XLV", "MRK": "XLV", "ABBV": "XLV", "LLY": "XLV",
    # Consumer Discretionary
    "AMZN": "XLY", "TSLA": "XLY", "HD": "XLY", "MCD": "XLY", "NKE": "XLY", "LOW": "XLY",
    # Consumer Staples
    "PG": "XLP", "KO": "XLP", "PEP": "XLP", "WMT": "XLP", "COST": "XLP", "PM": "XLP",
    # Energy
    "XOM": "XLE", "CVX": "XLE", "COP": "XLE", "SLB": "XLE", "EOG": "XLE",
    # Industrials
    "BA": "XLI", "HON": "XLI", "UPS": "XLI", "CAT": "XLI", "GE": "XLI", "LMT": "XLI",
    # Materials
    "LIN": "XLB", "APD": "XLB", "SHW": "XLB", "ECL": "XLB", "NEM": "XLB",
    # Utilities
    "NEE": "XLU", "DUK": "XLU", "SO": "XLU", "D": "XLU", "AEP": "XLU",
    # Real Estate
    "PLD": "XLRE", "AMT": "XLRE", "EQIX": "XLRE", "PSA": "XLRE", "O": "XLRE",
    # Communication Services
    "GOOGL": "XLC", "GOOG": "XLC", "META": "XLC", "NFLX": "XLC", "DIS": "XLC",
    "VZ": "XLC", "T": "XLC", "CMCSA": "XLC",
}

HISTORY_TAIL = 90  # 返回给前端走势小图用的历史点数


class SectorDataError(Exception):
    pass


async def _fetch_closes(ticker: str, client: httpx.AsyncClient) -> pd.Series:
    """RRG计算只需要收盘价，从共享的日线拉取里取一列。把 PolygonClientError 包装成
    SectorDataError，跟 require_api_key() 之前的做法一致——调用方（这个模块）只应该
    往外抛自己的领域异常，不应该让 polygon_client 的通用异常类型泄漏出去。"""
    try:
        bars = await fetch_daily_bars(ticker, client)
    except PolygonClientError as exc:
        raise SectorDataError(str(exc)) from exc
    return bars["close"]


async def get_sector_rotation(ticker: str) -> SectorRotationResponse:
    ticker = ticker.upper()
    matched_etf = PRESET_TICKER_SECTOR_MAP.get(ticker)
    note = None
    if matched_etf is None:
        note = f"'{ticker}' 不在当前预设股票池内，暂不支持自动板块识别（MVP阶段用小规模预设池验证）"

    async with httpx.AsyncClient(timeout=15.0) as client:
        benchmark_closes = await _fetch_closes(BENCHMARK, client)

        sectors: list[SectorPosition] = []
        for etf, sector_name in SECTOR_ETFS.items():
            sector_closes = await _fetch_closes(etf, client)
            rrg = compute_rrg(sector_closes, benchmark_closes)
            if rrg.empty:
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

            sectors.append(
                SectorPosition(
                    sector_etf=etf,
                    sector_name=sector_name,
                    rs_ratio=round(float(latest["rs_ratio"]), 3),
                    rs_momentum=round(float(latest["rs_momentum"]), 3),
                    quadrant=classify_quadrant(latest["rs_ratio"], latest["rs_momentum"]),
                    history=history,
                )
            )

    as_of = sectors[0].history[-1].date if sectors and sectors[0].history else ""
    matched_sector_name = SECTOR_ETFS.get(matched_etf) if matched_etf else None

    return SectorRotationResponse(
        ticker=ticker,
        matched_sector_etf=matched_etf,
        matched_sector_name=matched_sector_name,
        benchmark=BENCHMARK,
        as_of=as_of,
        sectors=sectors,
        note=note,
    )
