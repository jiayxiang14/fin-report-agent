"""SEC EDGAR 结构化财务数据客户端。

只负责：ticker -> CIK 映射、拉取 companyfacts（XBRL JSON）、
从原始 facts 里挑出关键指标并用代码计算同比变化。
不做任何"估算"或语义解读——这些交给 Agent。
"""

import json
from datetime import datetime, timedelta
from pathlib import Path

import httpx

from app.models.financials import (
    FinancialMetric,
    FinancialsHistoryResponse,
    FinancialsResponse,
    MetricHistory,
    MetricPoint,
)
from app.services.cache_lock import get_lock
from app.services.sec_client import SecClientError, throttled_get

TICKER_MAP_URL = "https://www.sec.gov/files/company_tickers.json"
COMPANY_FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"

CACHE_DIR = Path(__file__).resolve().parent.parent.parent / ".cache"
TICKER_MAP_CACHE_FILE = CACHE_DIR / "company_tickers.json"
TICKER_MAP_CACHE_TTL = timedelta(days=1)
COMPANY_FACTS_CACHE_TTL = timedelta(hours=6)  # 新财报只在季报/年报发布时出现，缓存几小时足够新鲜
HISTORY_YEARS = 15  # 多年趋势图默认回看年数

# 年度/季度报表的表格类型族：境外私人发行人（Foreign Private Issuer，比如NBIS这类
# 荷兰/开曼壳公司结构的公司）不用10-K/10-Q，改用20-F（年报，一年一次）/20-F/A（更正版），
# 且没有对应的季度报表义务——所以QUARTERLY_FORMS里没有20-F系列的等价物，这不是遗漏。
ANNUAL_FORMS = ("10-K", "20-F", "20-F/A")
QUARTERLY_FORMS = ("10-Q",)

# 每类指标按优先级尝试的 us-gaap 标签（不同公司披露习惯不同，取第一个命中的）
METRIC_TAGS: dict[str, list[str]] = {
    "revenue": [
        "Revenues",
        # 银行类公司的营收标签（JPM等）——它们的"Revenues"标签只剩年度数据，
        # 单季度数据早在2014年就停了，新的季度营收披露在这个标签下
        "RevenuesNetOfInterestExpense",
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "RevenueFromContractWithCustomerIncludingAssessedTax",
        "SalesRevenueNet",
    ],
    "net_income": ["NetIncomeLoss", "ProfitLoss"],
    "gross_profit": ["GrossProfit"],
    "operating_income": ["OperatingIncomeLoss"],
    "eps_diluted": ["EarningsPerShareDiluted"],
    "total_assets": ["Assets"],
    "total_liabilities": ["Liabilities"],
    "stockholders_equity": [
        "StockholdersEquity",
        "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
    ],
    "operating_cash_flow": [
        "NetCashProvidedByUsedInOperatingActivities",
        "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations",
    ],
    "capital_expenditures": [
        "PaymentsToAcquirePropertyPlantAndEquipment",
        "PaymentsForCapitalImprovements",
        "PaymentsToAcquireProductiveAssets",
    ],
}

METRIC_LABELS: dict[str, str] = {
    "revenue": "营业收入",
    "net_income": "净利润",
    "gross_profit": "毛利润",
    "operating_income": "营业利润",
    "eps_diluted": "稀释每股收益",
    "total_assets": "总资产",
    "total_liabilities": "总负债",
    "stockholders_equity": "股东权益",
    "operating_cash_flow": "经营活动现金流",
    "capital_expenditures": "资本支出",
    "free_cash_flow": "自由现金流",  # 派生指标，不在 METRIC_TAGS 里（没有对应的单一 us-gaap 标签）
}


class SecEdgarError(SecClientError):
    pass


class TickerNotFoundError(SecEdgarError):
    pass


async def _load_ticker_map(client: httpx.AsyncClient) -> dict[str, dict]:
    """加载 ticker -> {cik, title} 映射，本地磁盘缓存1天，避免每次请求都拉一次全量文件。
    这份缓存是全局共享的（不分ticker），并发保护用固定key而不是按参数区分。
    """
    async with get_lock("sec_ticker_map"):
        if TICKER_MAP_CACHE_FILE.exists():
            age = datetime.now() - datetime.fromtimestamp(TICKER_MAP_CACHE_FILE.stat().st_mtime)
            if age < TICKER_MAP_CACHE_TTL:
                raw = json.loads(TICKER_MAP_CACHE_FILE.read_text())
                return {entry["ticker"].upper(): entry for entry in raw.values()}

        try:
            response = await throttled_get(client, TICKER_MAP_URL)
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise SecEdgarError(f"拉取 SEC ticker 映射表失败：{exc}") from exc
        raw = response.json()

        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        TICKER_MAP_CACHE_FILE.write_text(json.dumps(raw))

        return {entry["ticker"].upper(): entry for entry in raw.values()}


async def resolve_cik(ticker: str, client: httpx.AsyncClient) -> tuple[str, str]:
    """返回 (cik补零到10位的字符串, 公司名)。"""
    ticker_map = await _load_ticker_map(client)
    entry = ticker_map.get(ticker.upper())
    if entry is None:
        raise TickerNotFoundError(f"在 SEC EDGAR 的 ticker 映射表里找不到 '{ticker}'")
    return f"{entry['cik_str']:010d}", entry["title"]


def _company_facts_cache_file(cik: str) -> Path:
    return CACHE_DIR / f"companyfacts_{cik}.json"


async def fetch_company_facts(cik: str, client: httpx.AsyncClient) -> dict:
    cache_file = _company_facts_cache_file(cik)
    async with get_lock(f"companyfacts_{cik}"):
        if cache_file.exists():
            age = datetime.now() - datetime.fromtimestamp(cache_file.stat().st_mtime)
            if age < COMPANY_FACTS_CACHE_TTL:
                return json.loads(cache_file.read_text())

        url = COMPANY_FACTS_URL.format(cik=int(cik))
        try:
            response = await throttled_get(client, url)
        except httpx.HTTPError as exc:
            raise SecEdgarError(f"请求 companyfacts 失败：{exc}") from exc
        if response.status_code == 404:
            raise SecEdgarError(f"CIK {cik} 没有可用的 companyfacts 数据（可能不是 XBRL 报告主体）")
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise SecEdgarError(f"SEC EDGAR 返回错误状态码 {response.status_code}") from exc
        data = response.json()

        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(json.dumps(data))

        return data


def _parse_points(unit_facts: list[dict]) -> list[dict]:
    """过滤只保留年度/季度报表（含境外发行人的20-F系列）的披露值，按期末日期倒序排列。"""
    valid_forms = ANNUAL_FORMS + QUARTERLY_FORMS
    points = [f for f in unit_facts if f.get("form") in valid_forms and f.get("end")]
    points.sort(key=lambda f: f["end"], reverse=True)
    return points


def _duration_days(point: dict) -> int | None:
    """流量型指标（营收/净利润等）同时带 start/end；存量型指标（总资产等）只有 end。"""
    start = point.get("start")
    if start is None:
        return None
    return (datetime.fromisoformat(point["end"]) - datetime.fromisoformat(start)).days


def _is_single_quarter(point: dict) -> bool:
    """单季度口径（约80~100天）。同一个 end 日期下，10-Q 里常常还混着9个月累计值等
    别的口径，必须靠 start~end 的实际跨度区分，不能只看 end。"""
    days = _duration_days(point)
    return days is None or 80 <= days <= 100


def _is_full_year(point: dict) -> bool:
    """整年口径（约350~380天），排除混进来的季度/半年累计值。"""
    days = _duration_days(point)
    return days is None or 350 <= days <= 380


def _find_yoy_point(points: list[dict], anchor_end: str, forms: tuple[str, ...]) -> dict | None:
    """在同一批 points 里找一个期末日期比 anchor 早约1年（350~380天）、同类型报表族的值。"""
    anchor_date = datetime.fromisoformat(anchor_end)
    for point in points:
        if point["form"] not in forms:
            continue
        candidate_date = datetime.fromisoformat(point["end"])
        delta_days = (anchor_date - candidate_date).days
        if 350 <= delta_days <= 380:
            return point
    return None


def _build_metric_point(point: dict, yoy_point: dict | None) -> MetricPoint:
    yoy_pct = None
    if yoy_point is not None and yoy_point["val"] != 0:
        yoy_pct = round((point["val"] - yoy_point["val"]) / abs(yoy_point["val"]) * 100, 2)
    return MetricPoint(
        end=point["end"],
        val=point["val"],
        fy=point.get("fy"),
        fp=point.get("fp"),
        form=point["form"],
        filed=point["filed"],
        yoy_change_pct=yoy_pct,
    )


def _match_points_by_end(a_points: list[dict], b_points: list[dict]) -> list[dict]:
    """两组同类型（都是年度或都是季度）的点位按 end 日期对齐后相减（a - b），只保留
    双方在同一个 end 日期都有披露的点——不假设两个标签的披露节奏完全同步，缺一侧的
    期间直接跳过，不硬凑。a_points 已经是按 end 倒序排列的，过滤后顺序不变。"""
    b_by_end = {p["end"]: p for p in b_points}
    derived = []
    for a in a_points:
        b = b_by_end.get(a["end"])
        if b is None:
            continue
        derived.append(
            {
                "end": a["end"],
                "val": a["val"] - b["val"],
                "fy": a.get("fy"),
                "fp": a.get("fp"),
                "form": a["form"],
                "filed": a["filed"],
            }
        )
    return derived


def _add_free_cash_flow(
    metrics: dict[str, FinancialMetric], raw_points: dict[str, tuple[list[dict], list[dict]]]
) -> None:
    """自由现金流 = 经营活动现金流 - 资本支出。两个源标签都必须真的取到数据才计算，
    任何一个缺失就跳过，不用0去凑一个虚假的自由现金流出来。"""
    if "operating_cash_flow" not in raw_points or "capital_expenditures" not in raw_points:
        return
    ocf_annual, ocf_quarterly = raw_points["operating_cash_flow"]
    capex_annual, capex_quarterly = raw_points["capital_expenditures"]

    annual_points = _match_points_by_end(ocf_annual, capex_annual)
    quarterly_points = _match_points_by_end(ocf_quarterly, capex_quarterly)
    if not annual_points and not quarterly_points:
        return

    latest_annual = None
    if annual_points:
        yoy = _find_yoy_point(annual_points, annual_points[0]["end"], ANNUAL_FORMS)
        latest_annual = _build_metric_point(annual_points[0], yoy)

    latest_quarterly = None
    if quarterly_points:
        yoy = _find_yoy_point(quarterly_points, quarterly_points[0]["end"], QUARTERLY_FORMS)
        latest_quarterly = _build_metric_point(quarterly_points[0], yoy)

    metrics["free_cash_flow"] = FinancialMetric(
        tag="NetCashProvidedByUsedInOperatingActivities - PaymentsToAcquirePropertyPlantAndEquipment",
        label=METRIC_LABELS["free_cash_flow"],
        unit=metrics["operating_cash_flow"].unit,
        latest_annual=latest_annual,
        latest_quarterly=latest_quarterly,
    )


def _dedupe_by_end(points: list[dict]) -> list[dict]:
    """同一个 end 日期的数值经常在好几份连续年报里重复出现——比如FY2023的营收，
    会作为对比年份分别出现在FY2023、FY2024、FY2025三份年报里（美股年报惯例披露
    近3年的对比利润表）。取多年历史序列时如果不去重，会把同一年重复算出好几个点。

    这里按 end 分组，每组只保留 filed 最晚的那一条——即公司对这个财年数字"最新的
    权威表述"。不能取最早那条：正常情况下（无业务重大变化）后续年报里的对比数据
    跟原始披露数值完全一样，取哪条都无所谓；但真实遇到过公司剥离主要业务后把历史
    可比期间重述为"持续经营口径"的情况（NBIS/Nebius剥离Yandex俄罗斯业务后，把
    2022/2023年营收从原始披露的数十亿美元重述成了几千万美元的持续经营口径——
    公司自己披露的正式财报新闻稿也是用重述后的数字），以及20-F/A更正版明确改写
    原始20-F数值的情况——这两种场景下"最早那条"就是过时或者被公司自己否定掉的
    数字，取最晚才是公司当前认可的口径。"""
    best_by_end: dict[str, dict] = {}
    for p in points:
        existing = best_by_end.get(p["end"])
        if existing is None or p["filed"] > existing["filed"]:
            best_by_end[p["end"]] = p
    return sorted(best_by_end.values(), key=lambda p: p["end"], reverse=True)


# 过时判断：同一个指标的候选标签里，有的公司换过标签、有的只剩年度或只剩季度
# 数据——真实踩过的坑：JPM的"Revenues"年度数据更新到2025、单季度数据停在2014，
# 原来的选法把年度和季度混在一起比"最新日期"，选中这个标签后把2014年Q3当成了
# "最新一季"返回；KO/WMT的"EarningsPerShareDiluted"整个标签2009年后就停用了，
# 照样被当成最新数据返回（公司概况的市盈率因此是拿2009年的EPS算的）。
# 现在年度、季度分别选标签，并且拿这家公司所有指标里最新的那个期末日期做参照：
# 比参照日期早太多的"最新一期"判定为过时，不返回，在notes里说明原因——宁可
# 告诉Agent"这项数据没有"，也不能让它拿到一个看起来正常的陈旧数字。
# 阈值留了余量：年报一年一次，最新年报期末可以比最新季报期末早将近一年；季报
# 允许落后一个季度（财年第四季度只出现在10-K里、没有单独的单季度数据）。
ANNUAL_STALE_DAYS = 550
QUARTERLY_STALE_DAYS = 200


def _select_series(
    us_gaap: dict, tag_candidates: list[str], kind: str
) -> tuple[str, str, list[dict]] | None:
    """从候选标签里，为年度（kind="annual"）或季度（kind="quarterly"）数据各自选出
    最新一期最靠后的那个标签，返回 (tag, unit_name, points)。points按end倒序。
    年度和季度分开选，避免一个标签"年度数据新、季度数据早就停了"时把旧季度
    数据带出来。所有候选标签都没有这类数据时返回 None。"""
    best: tuple[str, str, list[dict]] | None = None
    for tag in tag_candidates:
        tag_data = us_gaap.get(tag)
        if not tag_data:
            continue
        units = tag_data.get("units", {})
        if not units:
            continue
        # 少数公司（比如NBIS这类有俄罗斯/独联体业务历史的境外发行人）同一个标签下
        # 会同时披露本币和USD两种单位，随手取字典第一个键可能选到非USD单位——
        # 明确优先选USD，没有USD时才退回随便取一个（比如纯外国公司只用本币披露的情况）。
        unit_name = "USD" if "USD" in units else next(iter(units), None)
        if unit_name is None:
            continue
        points = _parse_points(units[unit_name])
        # 同一个 end 日期下，10-Q/10-K/20-F 里常常混着单季度、累计9个月、整年等多种口径
        # 的重复披露（start 不同），必须按 start~end 的实际跨度分开挑，不能只看 form+end。
        if kind == "annual":
            series = _dedupe_by_end([p for p in points if p["form"] in ANNUAL_FORMS and _is_full_year(p)])
        else:
            series = _dedupe_by_end([p for p in points if p["form"] in QUARTERLY_FORMS and _is_single_quarter(p)])
        if series and (best is None or series[0]["end"] > best[2][0]["end"]):
            best = (tag, unit_name, series)
    return best


def _days_between(later: str, earlier: str) -> int:
    return (datetime.fromisoformat(later) - datetime.fromisoformat(earlier)).days


def _reference_end(selections: dict[str, dict[str, tuple[str, str, list[dict]] | None]]) -> str:
    """这家公司所有指标、年度和季度里最新的那个期末日期——判断某一项是否过时的参照。"""
    ends = [sel[2][0]["end"] for per_metric in selections.values() for sel in per_metric.values() if sel]
    return max(ends, default="")


def _is_stale(point_end: str, reference_end: str, kind: str) -> bool:
    if not reference_end:
        return False
    limit = ANNUAL_STALE_DAYS if kind == "annual" else QUARTERLY_STALE_DAYS
    return _days_between(reference_end, point_end) > limit


def _build_history_points(annual_points: list[dict], years: int) -> list[MetricPoint]:
    """annual_points 已经按 end 倒序排列；取最近 years 个，逐个算自己那一期的同比，
    再倒转成正序（从早到晚）——前端折线图从左到右应该是时间顺序，不是披露倒序。"""
    recent = annual_points[:years]
    points = [_build_metric_point(p, _find_yoy_point(annual_points, p["end"], ANNUAL_FORMS)) for p in recent]
    return list(reversed(points))


def extract_metrics_history(facts_json: dict, years: int = HISTORY_YEARS) -> dict[str, MetricHistory]:
    """按 metric_key 返回最近 years 年的年度历史序列（含逐年同比），用于多年趋势图。

    只做年度（不含季度）——多年趋势图关心的是长期走势，季度数据在10-15年跨度下点数
    太密、季节性噪音也大，年度更适合看趋势。只用于同步展示给用户浏览，不接入Agent
    工具集：Agent走 extract_key_metrics 的 latest_annual/latest_quarterly 就够判断
    "这一期怎么样"，没必要把完整历史塞进每次工具调用的payload里增加token成本。
    """
    us_gaap = facts_json.get("facts", {}).get("us-gaap", {})
    history: dict[str, MetricHistory] = {}
    raw_annual: dict[str, list[dict]] = {}

    selections = {key: {"annual": _select_series(us_gaap, tags, "annual")} for key, tags in METRIC_TAGS.items()}
    reference_end = _reference_end(selections)

    for metric_key, per_metric in selections.items():
        selected = per_metric["annual"]
        if selected is None:
            continue
        _tag, unit_name, annual_points = selected
        # 整个标签早已停用（最新一年都过时了）——多年趋势图里画一段十几年前的
        # 序列只会误导，直接不展示
        if _is_stale(annual_points[0]["end"], reference_end, "annual"):
            continue

        raw_annual[metric_key] = annual_points
        history[metric_key] = MetricHistory(
            label=METRIC_LABELS[metric_key],
            unit=unit_name,
            points=_build_history_points(annual_points, years),
        )

    if "operating_cash_flow" in raw_annual and "capital_expenditures" in raw_annual:
        fcf_points = _match_points_by_end(raw_annual["operating_cash_flow"], raw_annual["capital_expenditures"])
        if fcf_points:
            history["free_cash_flow"] = MetricHistory(
                label=METRIC_LABELS["free_cash_flow"],
                unit=history["operating_cash_flow"].unit,
                points=_build_history_points(fcf_points, years),
            )

    return history


async def get_financials_history(ticker: str, years: int = HISTORY_YEARS) -> FinancialsHistoryResponse:
    async with httpx.AsyncClient(timeout=15.0) as client:
        cik, entity_name = await resolve_cik(ticker, client)
        facts_json = await fetch_company_facts(cik, client)

    history = extract_metrics_history(facts_json, years)

    return FinancialsHistoryResponse(
        ticker=ticker.upper(),
        cik=cik,
        entity_name=entity_name,
        history=history,
        retrieved_at=datetime.utcnow().isoformat() + "Z",
    )


def extract_key_metrics(facts_json: dict) -> dict[str, FinancialMetric]:
    us_gaap = facts_json.get("facts", {}).get("us-gaap", {})
    metrics: dict[str, FinancialMetric] = {}
    raw_points: dict[str, tuple[list[dict], list[dict]]] = {}

    selections = {
        key: {"annual": _select_series(us_gaap, tags, "annual"), "quarterly": _select_series(us_gaap, tags, "quarterly")}
        for key, tags in METRIC_TAGS.items()
    }
    reference_end = _reference_end(selections)

    for metric_key, per_metric in selections.items():
        annual_sel, quarterly_sel = per_metric["annual"], per_metric["quarterly"]
        if annual_sel is None and quarterly_sel is None:
            continue
        notes: list[str] = []
        annual_points: list[dict] = []
        quarterly_points: list[dict] = []
        if annual_sel is not None:
            if _is_stale(annual_sel[2][0]["end"], reference_end, "annual"):
                notes.append(
                    f"年度数据的最新一期停留在 {annual_sel[2][0]['end']}，比这家公司最新一期数据早太多（标签 {annual_sel[0]} "
                    "可能已停用），判定为过时数据，不予返回"
                )
            else:
                annual_points = annual_sel[2]
        if quarterly_sel is not None:
            if _is_stale(quarterly_sel[2][0]["end"], reference_end, "quarterly"):
                notes.append(
                    f"单季度数据的最新一期停留在 {quarterly_sel[2][0]['end']}，比这家公司最新一期数据早太多（标签 {quarterly_sel[0]} "
                    "可能已停用，或这项数据只按年初至今累计口径披露），判定为过时数据，不予返回"
                )
            else:
                quarterly_points = quarterly_sel[2]

        raw_points[metric_key] = (annual_points, quarterly_points)

        latest_annual = None
        if annual_points:
            yoy = _find_yoy_point(annual_points, annual_points[0]["end"], ANNUAL_FORMS)
            latest_annual = _build_metric_point(annual_points[0], yoy)

        latest_quarterly = None
        if quarterly_points:
            yoy = _find_yoy_point(quarterly_points, quarterly_points[0]["end"], QUARTERLY_FORMS)
            latest_quarterly = _build_metric_point(quarterly_points[0], yoy)

        primary = annual_sel or quarterly_sel
        assert primary is not None  # 上面已经排除了两者都是None的情况
        quarterly_tag = quarterly_sel[0] if quarterly_sel is not None and quarterly_sel[0] != primary[0] else None
        metrics[metric_key] = FinancialMetric(
            tag=primary[0],
            quarterly_tag=quarterly_tag,
            label=METRIC_LABELS[metric_key],
            unit=primary[1],
            latest_annual=latest_annual,
            latest_quarterly=latest_quarterly,
            notes=notes,
        )

    _add_free_cash_flow(metrics, raw_points)

    return metrics


async def get_financials(ticker: str) -> FinancialsResponse:
    async with httpx.AsyncClient(timeout=15.0) as client:
        cik, entity_name = await resolve_cik(ticker, client)
        facts_json = await fetch_company_facts(cik, client)

    metrics = extract_key_metrics(facts_json)

    return FinancialsResponse(
        ticker=ticker.upper(),
        cik=cik,
        entity_name=entity_name,
        metrics=metrics,
        retrieved_at=datetime.utcnow().isoformat() + "Z",
    )
