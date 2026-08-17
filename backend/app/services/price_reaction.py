"""财报发布前后的股价/成交量反应（Stage 3工具集补丁）。

背景：现有工具里没有一个能告诉Agent"这只股票自己的价格表现"——`get_sector_position`
给的是板块ETF相对大盘的强弱，不是个股涨跌。用户实测AMZN时发现Agent完全没提到"财报发布后
股价大涨15%"，根因是工具集本身缺这块数据，不是prompt能调好的。

关键设计：不能直接拿10-K/10-Q的申报日期算价格反应——大部分公司先发8-K业绩快报，股价当天
就反应了，正式10-Q/10-K往往滞后几天到几周才报送SEC，拿10-Q的日期算反应窗口很可能完全错过
真正的反应窗口。这里锚定的是最近一次带"2.02"条目（Results of Operations and Financial
Condition，即业绩快报）的8-K申报日期，并且用申报的精确时间戳（而不只是日期）判断反应窗口
落在申报当天还是下一个交易日——8-K常见在收盘后才提交，如果直接用申报当天收盘价，那时候
市场早就已经反应完了（或者还没开始反应）。

后续补丁：加了"反应后回吐程度"（`_compute_follow_through`）——初始反应之后几个交易日内，
这段涨跌被回吐了多少、回吐当天有没有放量，都是代码算的客观事实。**故意不做"洗盘/诱多/
诱空/获利了结"这类意图判断**：那是在猜测市场参与者的动机，价格数据本身验证不了，本质上
和被拒绝过的"上下游产业链利好利空判断"是同一类问题——没有数据支撑、又容易被读成买卖信号。
"""

from datetime import UTC, date, datetime
from zoneinfo import ZoneInfo

import httpx
import pandas as pd

from app.models.price_reaction import PriceReactionResponse
from app.services.alpha_vantage_client import AlphaVantageClientError, fetch_json
from app.services.filing_text import fetch_submissions
from app.services.polygon_client import PolygonClientError, fetch_daily_bars
from app.services.sec_client import SecClientError
from app.services.sec_edgar import resolve_cik

VOLUME_BASELINE_WINDOW = 20  # 成交量基线：反应窗口之前20个交易日的平均成交量
FOLLOW_THROUGH_WINDOW = 5  # 回看窗口：反应之后5个交易日内看这段涨跌有没有被回吐
_MARKET_CLOSE_HOUR_LOCAL = 16  # 美东时间下午4点收盘
_NY_TZ = ZoneInfo("America/New_York")
_ASSUMED_PRE_CLOSE_HOUR_LOCAL = 9  # 方案B（日期精度回退）假设的发布时间：早于收盘


class PriceReactionError(Exception):
    pass


async def find_latest_earnings_8k(cik: str, client: httpx.AsyncClient) -> dict | None:
    """在 submissions 的 recent 区块里找最近一次 items 包含"2.02"的8-K（业绩快报），
    不需要额外抓取解析8-K正文——submissions JSON本身就带 items 字段。找不到返回 None
    （比如这家公司不常规通过8-K发布业绩）。
    """
    try:
        submissions = await fetch_submissions(cik, client)
    except SecClientError as exc:
        raise PriceReactionError(str(exc)) from exc

    recent = submissions.get("filings", {}).get("recent", {})
    forms = recent.get("form", [])
    items_list = recent.get("items", [])
    filing_dates = recent.get("filingDate", [])
    acceptance_datetimes = recent.get("acceptanceDateTime", [])

    best: dict | None = None
    for i, form in enumerate(forms):
        if form != "8-K":
            continue
        items = items_list[i] if i < len(items_list) else ""
        if "2.02" not in [item.strip() for item in items.split(",")]:
            continue
        candidate_date = filing_dates[i]
        if best is None or candidate_date > best["filing_date"]:
            best = {
                "filing_date": candidate_date,
                "acceptance_datetime": (
                    acceptance_datetimes[i] if i < len(acceptance_datetimes) else None
                ),
            }
    return best


async def _find_fallback_earnings_date(
    ticker: str, client: httpx.AsyncClient
) -> tuple[str | None, str | None]:
    """方案A（SEC 8-K的Item 2.02业绩快报）查不到时的方案B：常见于境外发行人——
    走6-K/20-F申报，6-K没有items分类字段，没法自动识别哪一条是业绩公告。退化到
    Alpha Vantage季度盈利数据里的reportedDate当锚点，只有日期没有精确时间戳，
    调用方需要在note里如实标注精度下降。

    返回 (报告日期, 失败原因)，两者不会同时非None。失败原因只在真正的
    AlphaVantageClientError（比如没配置ALPHA_VANTAGE_API_KEY、或者当天25次
    配额用完）时才有值——不能把"配置/配额问题"和"这只票在Alpha Vantage确实
    没有盈利数据"都笼统地报成"没有数据"，前者换个时间/补上Key就能查到，
    后者是真的查不到，调用方需要能区分这两种情况分别如实告知用户。
    交给调用方决定怎么处理，不在这里抛出让整个请求失败。
    """
    try:
        payload = await fetch_json("EARNINGS", ticker, client)
    except AlphaVantageClientError as exc:
        return None, str(exc)
    quarterly = payload.get("quarterlyEarnings") or []
    if not quarterly:
        return None, None
    latest = max(quarterly, key=lambda q: q.get("fiscalDateEnding", ""))
    return latest.get("reportedDate"), None


def _synthesize_acceptance_datetime(filing_date: str) -> str:
    """把方案B只有日期没有时间戳的锚点，合成一个"当天收盘前"的时间戳喂给
    _resolve_reaction_window，复用同一套收盘前/收盘后分支逻辑——多数公司选择盘前
    或盘中发布财报（这个回退路径本来就是为了覆盖NBIS这类实际盘前发布财报的
    境外发行人），所以统一假设"发布早于当天收盘"：反应窗口是"发布前一交易日收盘
    -> 发布当天收盘"。如果实际是收盘后发布，这个窗口测的是发布当天而不是真正的
    次日反应，这就是为什么这条路径比SEC 8-K口径精度低，需要调用方如实标注。
    """
    local_at = datetime.combine(date.fromisoformat(filing_date), datetime.min.time()).replace(
        hour=_ASSUMED_PRE_CLOSE_HOUR_LOCAL, tzinfo=_NY_TZ
    )
    return local_at.astimezone(UTC).isoformat()


async def _fetch_bars(ticker: str, client: httpx.AsyncClient) -> pd.DataFrame:
    """价格反应只用得上收盘价+成交量，共享拉取额外带的最高/最低价用不上就忽略。
    把 PolygonClientError 包装成 PriceReactionError，跟 sector_rotation 的
    _fetch_closes 是同一个模式——调用方只应该往外抛自己的领域异常。"""
    try:
        return await fetch_daily_bars(ticker, client)
    except PolygonClientError as exc:
        raise PriceReactionError(str(exc)) from exc


def _resolve_reaction_window(
    acceptance_datetime: str, trading_dates: pd.DatetimeIndex
) -> tuple[pd.Timestamp, pd.Timestamp] | None:
    """把8-K的申报时间戳（UTC）转成美东本地时间，跟当地16:00收盘时间比较：
    - 收盘前申报：反应窗口的"事后"是申报当天收盘
    - 收盘后申报：反应窗口的"事后"是下一个交易日收盘
    ">="/">"这两种比较对"申报日恰好是周末/节假日（不在trading_dates里）"的情况天然
    也是对的——不管走哪个分支，都会自动跳到申报日之后第一个真实存在的交易日，不需要
    额外特判。"事前"固定是"事后"那个交易日紧邻的前一个交易日。
    """
    accepted_at = datetime.fromisoformat(acceptance_datetime.replace("Z", "+00:00"))
    local_at = accepted_at.astimezone(_NY_TZ)
    filing_calendar_date = pd.Timestamp(local_at.date())

    if local_at.hour < _MARKET_CLOSE_HOUR_LOCAL:
        candidates = trading_dates[trading_dates >= filing_calendar_date]
    else:
        candidates = trading_dates[trading_dates > filing_calendar_date]

    if len(candidates) == 0:
        return None
    post_date = candidates[0]

    earlier = trading_dates[trading_dates < post_date]
    if len(earlier) == 0:
        return None
    pre_date = earlier[-1]

    return pre_date, post_date


def _avg_volume_baseline(
    bars: pd.DataFrame, pre_date: pd.Timestamp, window: int = VOLUME_BASELINE_WINDOW
) -> float | None:
    """反应窗口之前 window 个交易日（不含pre_date本身）的平均成交量，作为"正常成交量"基线。"""
    prior = bars.loc[bars.index < pre_date, "volume"]
    if prior.empty:
        return None
    baseline = prior.tail(window)
    if baseline.empty:
        return None
    return float(baseline.mean())


def _compute_follow_through(
    bars: pd.DataFrame,
    pre_close: float,
    post_close: float,
    post_date: pd.Timestamp,
    baseline: float | None,
    window: int = FOLLOW_THROUGH_WINDOW,
) -> tuple[float | None, float | None, float | None]:
    """反应之后 window 个交易日内，这段涨跌被"回吐"了多少——不判断这是洗盘还是诱多还是
    获利了结（那是意图判断，价格数据本身验证不了），只算一个客观的回吐比例：涨势看窗口内
    最低收盘价、跌势看最高收盘价，(post_close - 极值) / 初始涨跌幅 * 100，可能超过100%
    （说明不仅回吐完还反向突破了）。数据不足 window 个交易日（反应太新）时返回全 None。
    """
    future = bars.loc[bars.index > post_date]
    future_window = future.iloc[:window]
    if len(future_window) < window:
        return None, None, None

    initial_move = post_close - pre_close
    if initial_move >= 0:
        extreme_date = future_window["close"].idxmin()
    else:
        extreme_date = future_window["close"].idxmax()
    extreme_close = float(future_window.loc[extreme_date, "close"])

    if initial_move == 0:
        reversal_pct = None
    else:
        giveback = (post_close - extreme_close) if initial_move > 0 else (extreme_close - post_close)
        reversal_pct = round(giveback / abs(initial_move) * 100, 2)

    extreme_volume = float(future_window.loc[extreme_date, "volume"])
    volume_ratio = round(extreme_volume / baseline, 2) if baseline else None

    return round(extreme_close, 2), reversal_pct, volume_ratio


async def get_price_reaction(ticker: str) -> PriceReactionResponse:
    ticker = ticker.upper()
    precision_note: str | None = None

    async with httpx.AsyncClient(timeout=15.0) as client:
        try:
            cik, _entity_name = await resolve_cik(ticker, client)
        except SecClientError as exc:
            raise PriceReactionError(str(exc)) from exc

        earnings_8k = await find_latest_earnings_8k(cik, client)

        if earnings_8k is not None and earnings_8k["acceptance_datetime"] is not None:
            acceptance_datetime = earnings_8k["acceptance_datetime"]
            filing_date = earnings_8k["filing_date"]
        else:
            # 方案A查不到（典型场景：境外发行人走6-K/20-F，没有items分类字段可以
            # 识别业绩公告）——退化到方案B，用Alpha Vantage的报告日期当锚点
            fallback_date, fallback_error = await _find_fallback_earnings_date(ticker, client)
            if fallback_date is None:
                # fallback_error 有值说明是配置/配额问题（能恢复），不是"这只票真的
                # 没有数据"——两种情况如实分开说，不能笼统地都说成"没有数据"
                reason = (
                    f"尝试用Alpha Vantage的报告日期兜底时失败：{fallback_error}"
                    if fallback_error is not None
                    else "Alpha Vantage 也没有可用的盈利报告日期"
                )
                return PriceReactionResponse(
                    ticker=ticker,
                    has_data=False,
                    note=(
                        f"'{ticker}' 最近的申报记录里找不到包含业绩快报（Item 2.02）的8-K，"
                        f"{reason}，无法计算财报发布前后的价格反应"
                    ),
                )
            filing_date = fallback_date
            acceptance_datetime = _synthesize_acceptance_datetime(fallback_date)
            precision_note = (
                "锚点日期来自Alpha Vantage盈利数据的报告日期，非SEC申报的精确时间戳，"
                "按'发布早于当天收盘'假设计算反应窗口，精度低于SEC 8-K口径"
            )

        bars = await _fetch_bars(ticker, client)

    window = _resolve_reaction_window(acceptance_datetime, bars.index)
    if window is None:
        return PriceReactionResponse(
            ticker=ticker,
            has_data=False,
            earnings_filing_date=filing_date,
            note="行情数据覆盖范围不足以定位这次财报前后的交易日，无法计算价格反应",
        )

    pre_date, post_date = window
    pre_close = float(bars.loc[pre_date, "close"])
    post_close = float(bars.loc[post_date, "close"])
    post_volume = float(bars.loc[post_date, "volume"])

    price_change_pct = round((post_close - pre_close) / pre_close * 100, 2) if pre_close else None

    baseline = _avg_volume_baseline(bars, pre_date)
    volume_ratio = round(post_volume / baseline, 2) if baseline else None

    extreme_close, reversal_pct, reversal_volume_ratio = _compute_follow_through(
        bars, pre_close, post_close, post_date, baseline
    )
    note = f"反应窗口：{pre_date.date().isoformat()} 收盘 -> {post_date.date().isoformat()} 收盘"
    if precision_note:
        note += f"；{precision_note}"
    if extreme_close is None:
        note += f"；反应后交易日不足{FOLLOW_THROUGH_WINDOW}天，暂无法计算回吐情况"

    return PriceReactionResponse(
        ticker=ticker,
        has_data=True,
        earnings_filing_date=filing_date,
        pre_close=pre_close,
        post_close=post_close,
        price_change_pct=price_change_pct,
        post_volume=post_volume,
        avg_volume_baseline=round(baseline, 2) if baseline else None,
        volume_ratio=volume_ratio,
        follow_through_window_days=FOLLOW_THROUGH_WINDOW if extreme_close is not None else None,
        extreme_close_in_window=extreme_close,
        reversal_pct_of_initial_move=reversal_pct,
        reversal_day_volume_ratio=reversal_volume_ratio,
        note=note,
    )
