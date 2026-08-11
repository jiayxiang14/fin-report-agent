"""公司概览：简介、市值、细分行业、20日平均日振幅（ADR）。

背景：用户看完前端后觉得缺一个"分析这家公司"的概览部分（公司简介、市值、所属细分行业、
波动性），且现有面板（财务数据/板块位置）全是数字和文字表格，缺一个更直观的"第一眼"
概览。这些数据不需要Agent"决定要不要查"——跟财务数据、板块位置一样，属于同步展示的
确定性数据，所以做成REST路由直接给前端用，不接入Agent工具集（保持工具数量不变）。

Polygon的Ticker Details接口一次性把公司简介文本、市值（不需要自己拿股数×股价去算）、
细分行业分类（比我们11个SPDR大类更细）都给出来了。20日ADR需要日线的最高/最低价，
复用 polygon_client 里三个模块共享的 fetch_daily_bars。
"""

import json
from datetime import datetime, timedelta
from pathlib import Path

import httpx

from app.models.company_profile import CompanyProfileResponse
from app.services.cache_lock import get_lock
from app.services.polygon_client import (
    CACHE_DIR,
    PolygonClientError,
    fetch_daily_bars,
    require_api_key,
    throttled_get,
)
from app.services.sec_client import SecClientError
from app.services.sec_edgar import get_financials

TICKER_DETAILS_URL = "https://api.polygon.io/v3/reference/tickers/{ticker}"
TICKER_DETAILS_CACHE_TTL = timedelta(hours=20)  # 公司简介/市值这类信息变化不快，跟日线一致的刷新频率足够

ADR_WINDOW = 20  # 最近20个交易日的平均日振幅


class CompanyProfileError(Exception):
    pass


def _details_cache_file(ticker: str) -> Path:
    return CACHE_DIR / f"polygon_ticker_details_{ticker}.json"


async def fetch_ticker_details(ticker: str, client: httpx.AsyncClient) -> dict | None:
    """拉取Polygon的公司参考信息，磁盘缓存20小时。查不到这个ticker（404）返回None，
    不当成硬错误——跟其他"降级"工具的处理方式一致。

    这个函数是共享的（不只是company_profile自己用）：`thematic_flow.py`也调用
    它拿`sic_description`做SIC行业分类的关键词匹配——同一份磁盘缓存，两边谁先
    调用谁写缓存，后调用的直接命中缓存，不会对同一个ticker重复打两次Polygon。
    这个"谁先调用谁写"的保证是靠get_lock做出来的：前端加载时公司概览面板和
    主题轮动面板本来就会并发请求同一个ticker，没有锁的话这个保证只是口头
    描述，实际会被并发竞态打破。
    """
    cache_file = _details_cache_file(ticker)
    async with get_lock(f"polygon_ticker_details_{ticker}"):
        if cache_file.exists():
            age = datetime.now() - datetime.fromtimestamp(cache_file.stat().st_mtime)
            if age < TICKER_DETAILS_CACHE_TTL:
                cached = json.loads(cache_file.read_text())
                return cached if cached else None

        url = TICKER_DETAILS_URL.format(ticker=ticker)
        api_key = require_api_key()
        params = {"apiKey": api_key}

        try:
            response = await throttled_get(client, url, params)
        except httpx.HTTPError as exc:
            raise CompanyProfileError(f"请求 Polygon 公司信息失败：{exc}") from exc
        if response.status_code == 404:
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            cache_file.write_text(json.dumps(None))
            return None
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise CompanyProfileError(f"Polygon 返回错误状态码 {response.status_code}") from exc

        payload = response.json()
        results = payload.get("results")
        if not results:
            return None

        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(json.dumps(results))
        return results


def _compute_adr_pct(bars, window: int = ADR_WINDOW) -> float | None:
    """最近 window 个交易日的平均日振幅，占收盘价的百分比：((high-low)/close*100).mean()。"""
    recent = bars.tail(window)
    if recent.empty:
        return None
    daily_range_pct = (recent["high"] - recent["low"]) / recent["close"] * 100
    return round(float(daily_range_pct.mean()), 2)


async def _fetch_pe_ratio(ticker: str, latest_close: float) -> float | None:
    """市盈率 = 最新收盘价 / 最新一期年度稀释EPS。用的是最近一个完整财年的EPS，
    不是严格意义上的TTM（近四个季度滚动）——`extract_key_metrics` 目前只保留
    最新一期年度和最新一期季度两个点，没有完整的近四季度历史可供滚动求和，这是
    一个明确的简化，不是假装精确的TTM，前端要如实标注。EPS为负或缺失时P/E没有
    意义（不是"负的市盈率"），返回None。SEC EDGAR拉取失败时同样返回None——市盈率
    算不出来不应该拖垮整个公司概览请求，这是一个软性降级，不是硬错误。
    """
    try:
        financials = await get_financials(ticker)
    except SecClientError:
        return None
    eps_metric = financials.metrics.get("eps_diluted")
    if eps_metric is None or eps_metric.latest_annual is None:
        return None
    eps = eps_metric.latest_annual.val
    if eps <= 0:
        return None
    return round(latest_close / eps, 2)


async def get_company_profile(ticker: str) -> CompanyProfileResponse:
    ticker = ticker.upper()

    async with httpx.AsyncClient(timeout=15.0) as client:
        details = await fetch_ticker_details(ticker, client)

        if details is None:
            return CompanyProfileResponse(
                ticker=ticker,
                has_data=False,
                note=f"Polygon 找不到 '{ticker}' 的公司参考信息",
            )

        try:
            bars = await fetch_daily_bars(ticker, client)
        except PolygonClientError as exc:
            raise CompanyProfileError(str(exc)) from exc

    adr_20d_pct = _compute_adr_pct(bars)
    latest_close = float(bars["close"].iloc[-1])
    pe_ratio = await _fetch_pe_ratio(ticker, latest_close)

    return CompanyProfileResponse(
        ticker=ticker,
        has_data=True,
        name=details.get("name"),
        description=details.get("description"),
        market_cap=details.get("market_cap"),
        sic_description=details.get("sic_description"),
        homepage_url=details.get("homepage_url"),
        total_employees=details.get("total_employees"),
        adr_20d_pct=adr_20d_pct,
        pe_ratio=pe_ratio,
    )
