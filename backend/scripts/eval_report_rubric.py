"""简报质量评测：固定检查清单 + 陷阱场景 + 重复运行 + 质量与成本一起报告。

借鉴 claude-quickstarts 的 knowledge-wiki 评测思路，比较三种配置：
- normal：普通分析（一次Agent Loop）
- best_of_n：深度分析（3个候选+规则分+裁判择优）
- draft_review：草稿 + 更强模型审阅 + 修改（app/services/agent/draft_review.py）

**检查清单**：每只股票套同一组检查项，标准答案由代码从数据工具算出（营收/净利润
同比、EPS是否超预期、财报后股价反应、是否放量、板块轮动阶段、财务数据与EPS数据
的期间差）。

**陷阱场景**：
- 天然存在的：口径不可比（GAAP EPS和调整后一致预期差很多）、信号冲突（超预期但
  股价大跌）、期间错位（财务数据和EPS数据不是同一季）——在常规运行里自动识别和检查。
- 人为注入的：数据缺失（让get_earnings_surprise报错）、埋矛盾文件（往财报原文里
  插一段伪造的"第三方速评"，营收数字和SEC数据矛盾）。

**角色隔离**：评分者（deepseek-v4-pro，跟生成简报的flash不是同一个模型）只看简报
正文，不知道简报来自哪种配置；它只负责拆出"简报对每一项说了什么、原话是哪句"，
不判断好坏——对错一律由代码对照标准答案判。每条拆分结果附的原话由代码核对是否
真的出现在简报里，对不上的作废；`--extract-twice`时同一份简报拆两次，报告一致率。

**重复运行**：每种配置每只股票跑`--repeats`次，配置之间的差距必须大于同一配置
自己的波动才有意义。

所有简报、拆分结果、评分明细都存到 .cache/eval_report_rubric/<时间戳>/，供人工抽查。
会真实调用LLM和全部数据源，不接入CI。

用法（在 backend/ 目录下）：
    python scripts/eval_report_rubric.py --pilot                      # 1只股票×3种配置×1次，先确认流程
    python scripts/eval_report_rubric.py                              # 全量
    python scripts/eval_report_rubric.py --configs normal,draft_review --repeats 2
"""

from __future__ import annotations

import argparse
import asyncio
import contextvars
import json
import random
import re
import statistics
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import UTC, datetime
from math import comb
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.services.agent import llm_client as llm_client_module  # noqa: E402
from app.services.agent import llm_pricing, loop  # noqa: E402
from app.services.agent.best_of_n import run_best_of_n  # noqa: E402
from app.services.agent.draft_review import run_draft_review  # noqa: E402
from app.services.agent.llm_client import LLMUsage, get_llm_client  # noqa: E402
from app.services.polygon_client import CACHE_DIR  # noqa: E402

OUT_ROOT: Path = CACHE_DIR / "eval_report_rubric"
RUNS_LOG_PATH: Path = CACHE_DIR / "eval_report_rubric_runs.jsonl"

DEFAULT_TICKERS = ["AAPL", "MSFT", "NVDA", "MU", "KO", "WMT", "JPM", "XOM", "PFE", "NBIS"]
ALL_CONFIGS = ("normal", "best_of_n", "draft_review")
GRADER_MODEL = "deepseek-v4-pro"
GRADER_MAX_TOKENS = 32_000  # 同draft_review.REVIEWER_MAX_TOKENS：pro的thinking很长
# 人为注入的陷阱：场景名 -> 用哪些股票
INJECTED_TRAPS: dict[str, list[str]] = {"missing_eps_data": ["JPM", "KO"], "planted_conflict": ["MSFT", "XOM"]}
MAX_CONCURRENT_PIPELINES = 4
# 账户层面的错误（余额不足、鉴权失败）：任何一次运行碰到就整体停下——第一次全量
# 运行时DeepSeek余额已经用完，脚本照样把126次注定失败的运行全部跑完才结束
FATAL_ERROR_MARKERS = ("Error code: 402", "Error code: 401", "Insufficient Balance")


_abort = asyncio.Event()
# 整个评测（被评测配置 + 评分）累计花了多少钱；--max-cost 超了就停，不等余额耗尽
_total_usage: dict[str, LLMUsage] = {}
RUN_TIMEOUT_SECONDS = 20 * 60  # 单次运行卡在数据源限速器之类的地方时，不能把整个评测挂住


def _is_fatal(exc: BaseException) -> bool:
    """优先看HTTP状态码（不依赖供应商的错误措辞）；Best-of-N等会把原始异常包成别的
    异常、只剩错误文本，这时再退回匹配文本。"""
    seen: BaseException | None = exc
    while seen is not None:
        if getattr(seen, "status_code", None) in (401, 402):
            return True
        seen = seen.__cause__
    return any(marker in str(exc) for marker in FATAL_ERROR_MARKERS)


def _note_fatal(exc: BaseException) -> None:
    if _is_fatal(exc) and not _abort.is_set():
        _abort.set()
        print(f"\n遇到账户层面的错误，停止发起剩余运行：{str(exc)[:160]}\n", flush=True)

# 判对错的容差（百分点）
YOY_TOLERANCE_PP = 0.6
SURPRISE_TOLERANCE_PP = 1.0
PRICE_TOLERANCE_PP = 0.6
FLAT_BAND_PP = 0.5  # 绝对值小于它算"基本持平"
VOLUME_SPIKE_RATIO = 1.5
PERIOD_GAP_DAYS = 45
GAAP_VS_ADJUSTED_GAP = 0.2  # GAAP EPS和Alpha Vantage的reported EPS差这么多（绝对值）才算口径明显不同
CONFLICT_PRICE_MOVE_PP = 3.0
# 两个"发现异常"类的检查项：只判有没有指出，不判措辞
DIVERGENCE_MIN_PP = 5.0  # 净利润和经营现金流同比方向相反、且各自变化都超过这个幅度才算背离
REVERSAL_MIN_PCT = 50.0  # 初始涨跌在随后几天被回吐超过一半
REVERSAL_MIN_MOVE_PP = 2.0  # 初始涨跌太小时谈不上"回吐"

# ---------- 运行时注入：陷阱场景 + 用量计费 ----------

_active_trap: contextvars.ContextVar[str | None] = contextvars.ContextVar("active_trap", default=None)
_usage_sink: contextvars.ContextVar[dict[str, LLMUsage] | None] = contextvars.ContextVar("usage_sink", default=None)
_planted_text: dict[str, str] = {}
_original_execute_tool = loop.execute_tool
_original_create_message = llm_client_module.AnthropicCompatibleClient.create_message


async def _execute_tool_with_traps(name: str, tool_input: dict) -> tuple[str, bool]:
    trap = _active_trap.get()
    if trap == "missing_eps_data" and name == "get_earnings_surprise":
        return "Alpha Vantage 服务暂时不可用（HTTP 503），未能获取分析师一致预期数据", True
    output, is_error = await _original_execute_tool(name, tool_input)
    if trap == "planted_conflict" and name == "get_filing_text" and not is_error:
        ticker = str(tool_input.get("ticker", "")).upper()
        # 放在开头而不是末尾：财报原文动辄几十万字，附在最后模型可能根本读不到，
        # "没被骗"可能只是"没看见"，陷阱通过率会虚高
        output = _planted_text.get(ticker, "") + "\n\n" + output
    return output, is_error


async def _create_message_with_accounting(self: Any, *args: Any, **kwargs: Any) -> Any:
    response = await _original_create_message(self, *args, **kwargs)
    total = _total_usage.setdefault(self._model, LLMUsage(0, 0))
    total.input_tokens += response.usage.input_tokens
    total.output_tokens += response.usage.output_tokens
    total.cache_creation_input_tokens += response.usage.cache_creation_input_tokens
    total.cache_read_input_tokens += response.usage.cache_read_input_tokens
    sink = _usage_sink.get()
    if sink is not None:
        acc = sink.setdefault(self._model, LLMUsage(0, 0))
        acc.input_tokens += response.usage.input_tokens
        acc.output_tokens += response.usage.output_tokens
        acc.cache_creation_input_tokens += response.usage.cache_creation_input_tokens
        acc.cache_read_input_tokens += response.usage.cache_read_input_tokens
    return response


def install_instrumentation() -> None:
    loop.execute_tool = _execute_tool_with_traps
    llm_client_module.AnthropicCompatibleClient.create_message = _create_message_with_accounting  # type: ignore[method-assign]


def usage_cost(usage_by_model: dict[str, LLMUsage]) -> float | None:
    total = 0.0
    for model, usage in usage_by_model.items():
        cost = llm_pricing.estimate_cost_usd(model, usage)
        if cost is None:
            return None
        total += cost
    return round(total, 4)


# ---------- 标准答案 ----------


def _direction(pct: float | None) -> str | None:
    if pct is None:
        return None
    if abs(pct) < FLAT_BAND_PP:
        return "flat"
    return "up" if pct > 0 else "down"


def build_truth(financials: dict | None, surprise: dict | None, reaction: dict | None, sector: dict | None) -> dict:
    """从四个数据工具的原始输出算出这只股票的标准答案。拿不到的项是None（对应
    检查项不适用，不计入分母）。"""
    metrics = (financials or {}).get("metrics", {})

    def metric(key: str, period: str) -> dict:
        return (metrics.get(key) or {}).get(period) or {}

    truth: dict[str, Any] = {
        "revenue": {"quarter": metric("revenue", "latest_quarterly").get("yoy_change_pct"), "annual": metric("revenue", "latest_annual").get("yoy_change_pct")},
        "net_income": {"quarter": metric("net_income", "latest_quarterly").get("yoy_change_pct"), "annual": metric("net_income", "latest_annual").get("yoy_change_pct")},
        "financial_quarter_end": metric("revenue", "latest_quarterly").get("end"),
        "eps_verdict": None,
        "eps_surprise_pct": None,
        "eps_fiscal_end": None,
        "gaap_eps_quarter": metric("eps_diluted", "latest_quarterly").get("val"),
        "reported_eps": None,
        "price_change_pct": None,
        "volume_ratio": None,
        "sector_quadrant": None,
        "revenue_quarter_val": metric("revenue", "latest_quarterly").get("val"),
        "operating_cash_flow": {
            "quarter": metric("operating_cash_flow", "latest_quarterly").get("yoy_change_pct"),
            "annual": metric("operating_cash_flow", "latest_annual").get("yoy_change_pct"),
        },
        "reversal_pct": None,
    }
    if surprise and surprise.get("has_data"):
        truth["eps_verdict"] = {"超预期": "beat", "低于预期": "miss", "符合预期": "inline"}.get(str(surprise.get("verdict")))
        truth["eps_surprise_pct"] = surprise.get("surprise_percentage")
        truth["eps_fiscal_end"] = surprise.get("fiscal_date_ending")
        truth["reported_eps"] = surprise.get("reported_eps")
    if reaction and reaction.get("has_data"):
        truth["price_change_pct"] = reaction.get("price_change_pct")
        truth["volume_ratio"] = reaction.get("volume_ratio")
        truth["reversal_pct"] = reaction.get("reversal_pct_of_initial_move")
    if sector and sector.get("matched_sector_etf"):
        truth["sector_quadrant"] = next(
            (s.get("quadrant") for s in sector.get("sectors", []) if s.get("sector_etf") == sector["matched_sector_etf"]),
            None,
        )
    return truth


def natural_traps(truth: dict) -> dict[str, bool]:
    """这只股票数据里天然存在哪些陷阱。"""
    gap_days = None
    if truth["financial_quarter_end"] and truth["eps_fiscal_end"]:
        gap_days = abs((datetime.fromisoformat(truth["eps_fiscal_end"]) - datetime.fromisoformat(truth["financial_quarter_end"])).days)
    gaap, reported = truth["gaap_eps_quarter"], truth["reported_eps"]
    price = truth["price_change_pct"]
    period_gap = gap_days is not None and gap_days > PERIOD_GAP_DAYS
    return {
        "period_gap": period_gap,
        # 只有同一季度的GAAP EPS和调整后EPS差很多，才是口径问题；不同季度的数字
        # 本来就不该比（第一次全量运行时MSFT/MU/JPM被误标成口径陷阱，实际是期间错位）
        "eps_basis": not period_gap and gaap is not None and reported is not None and abs(gaap - reported) > GAAP_VS_ADJUSTED_GAP,
        "beat_vs_price_conflict": price is not None
        and truth["eps_verdict"] in ("beat", "miss")
        and ((truth["eps_verdict"] == "beat" and price <= -CONFLICT_PRICE_MOVE_PP) or (truth["eps_verdict"] == "miss" and price >= CONFLICT_PRICE_MOVE_PP)),
    }


def has_cash_flow_divergence(truth: dict) -> bool:
    for period in ("quarter", "annual"):
        ni, ocf = truth["net_income"][period], truth["operating_cash_flow"][period]
        if ni is not None and ocf is not None and abs(ni) > DIVERGENCE_MIN_PP and abs(ocf) > DIVERGENCE_MIN_PP and (ni > 0) != (ocf > 0):
            return True
    return False


def has_reaction_reversal(truth: dict) -> bool:
    move, reversal = truth["price_change_pct"], truth["reversal_pct"]
    return move is not None and reversal is not None and abs(move) >= REVERSAL_MIN_MOVE_PP and reversal > REVERSAL_MIN_PCT


async def fetch_truth(ticker: str) -> dict:
    outputs = {}
    for name in ("get_financials", "get_earnings_surprise", "get_price_reaction", "get_sector_position"):
        out, is_error = await _original_execute_tool(name, {"ticker": ticker})
        outputs[name] = None if is_error else json.loads(out)
    return build_truth(outputs["get_financials"], outputs["get_earnings_surprise"], outputs["get_price_reaction"], outputs["get_sector_position"])


def planted_note(ticker: str, truth: dict) -> str:
    """伪造的第三方速评：营收和同比都跟SEC数据明显矛盾。"""
    real_val = truth.get("revenue_quarter_val") or 10e9
    real_yoy = truth["revenue"]["quarter"] if truth["revenue"]["quarter"] is not None else 5.0
    fake_val = real_val * 1.25 / 1e8
    fake_yoy = round(real_yoy + 15.0, 1)
    truth["planted_yoy"] = fake_yoy
    return (
        "【附：第三方券商速评（非公司官方披露）】据速评测算，"
        f"{ticker}最新季度营收约为{fake_val:.1f}亿美元，同比增长{fake_yoy}%，显著好于市场预期。"
    )


# ---------- 评分者：只拆分，不判断 ----------

GRADER_SYSTEM_PROMPT = """你是一个信息抽取器，只负责把一份投研简报里"写了什么"原样拆出来，不评价好坏、不补充简报没写的内容、不做任何推断。
每一项都要附上简报中的原话（逐字摘抄一句，不要改写）；简报没提到的项，mentioned填false、quote填空字符串。只输出一个JSON对象，不要任何解释。"""

GRADER_INSTRUCTION = """从下面这份简报中抽取以下信息，输出JSON，字段如下：

{{
  "revenue":        {{"mentioned": bool, "period": "quarter|annual|unclear", "direction": "up|down|flat|null", "yoy_pct": 数字或null, "quote": "原话"}},
  "net_income":     {{"mentioned": bool, "period": "quarter|annual|unclear", "direction": "up|down|flat|null", "yoy_pct": 数字或null, "quote": "原话"}},
  "eps_surprise":   {{"mentioned": bool, "verdict": "beat|miss|inline|null", "surprise_pct": 数字或null, "quote": "原话"}},
  "price_reaction": {{"mentioned": bool, "direction": "up|down|flat|null", "change_pct": 数字或null, "mentions_volume": bool, "quote": "原话"}},
  "sector":         {{"mentioned": bool, "quadrant": "leading|weakening|lagging|improving|null", "quote": "原话"}},
  "period_gap":     {{"mentioned": bool, "quote": "原话"}},
  "eps_basis":      {{"compares_gaap_with_consensus": bool, "notes_basis_difference": bool, "quote": "原话"}},
  "beat_vs_price_conflict": {{"mentioned": bool, "quote": "原话"}},
  "data_gap_eps":   {{"mentioned": bool, "quote": "原话"}},
  "conflicting_sources": {{"mentioned": bool, "quote": "原话"}},
  "cash_flow_divergence": {{"mentioned": bool, "quote": "原话"}},
  "reaction_reversal": {{"mentioned": bool, "quote": "原话"}}
}}

字段含义：
- revenue / net_income：简报对营收/净利润同比变化的描述。period是简报说的是最新一个季度还是全年；yoy_pct是简报写的同比百分比（写了才填）。如果简报同时写了季度和全年，取季度的。
- eps_surprise：简报对EPS相对分析师预期是超出、低于还是符合的描述，surprise_pct是超出/低于的百分比。
- price_reaction：简报对财报发布后股价反应的描述，change_pct是涨跌幅百分比（下跌填负数），mentions_volume是否提到成交量放大/放量。
- sector：简报说公司所在板块处于轮动的哪个阶段（领先/转弱/落后/改善 分别对应 leading/weakening/lagging/improving）。
- period_gap：简报是否明确指出财务数据所属的期间，跟EPS超预期或股价反应所对应的期间不是同一个（存在时间差）。
- eps_basis：compares_gaap_with_consensus = 简报是否把GAAP口径（财报里）的EPS直接拿去跟分析师一致预期比较；notes_basis_difference = 简报是否指出GAAP口径和调整后口径不同、不能直接比较。
- beat_vs_price_conflict：简报是否明确指出"业绩超预期（或不及预期）"和"股价反应方向"相互矛盾。
- data_gap_eps：简报是否说明分析师一致预期/EPS超预期数据没有获取到或缺失。
- conflicting_sources：简报是否指出不同来源的数据相互矛盾（比如第三方数据和公司财报对不上）。
- cash_flow_divergence：简报是否指出净利润和经营活动现金流的变化方向相反（背离）。
- reaction_reversal：简报是否提到财报发布后的初始涨跌，在随后几天被大幅回吐、回补或反转。

【简报】
{report}"""


def _parse_json_object(text: str) -> dict | None:
    match = re.search(r"\{[\s\S]*\}", text)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def _normalize(text: str) -> str:
    return re.sub(r"[\s*`#>\-]+", "", text)


def verify_quotes(extraction: dict, report: str) -> tuple[dict, list[str]]:
    """原话不在简报里的项作废（当作没提到），返回(核对后的抽取结果, 作废的项名)。
    比较前去掉空白和markdown符号，避免因为格式差异误判。"""
    normalized_report = _normalize(report)
    invalid = []
    checked = {}
    for key, item in extraction.items():
        if not isinstance(item, dict):
            continue
        claims = item.get("mentioned") or item.get("compares_gaap_with_consensus") or item.get("notes_basis_difference")
        quote = str(item.get("quote") or "")
        if claims and quote and _normalize(quote) not in normalized_report:
            invalid.append(key)
            checked[key] = {"mentioned": False, "quote": "", "invalidated": True}
        else:
            checked[key] = item
    return checked, invalid


async def extract(report: str) -> dict | None:
    grader = get_llm_client(model=GRADER_MODEL)
    response = await grader.create_message(
        system=GRADER_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": GRADER_INSTRUCTION.format(report=report)}],
        tools=[],
        temperature=0.0,
        max_tokens=GRADER_MAX_TOKENS,
        effort="low",  # 抽取任务简单，不需要长时间推理（同draft_review.REVIEWER_EFFORT）
    )
    return _parse_json_object(response.text or "")


# ---------- 代码判对错 ----------


def _num(value: Any) -> float | None:
    return float(value) if isinstance(value, int | float) and not isinstance(value, bool) else None


def _score_directional(item: dict, truth_value: float | None, tolerance: float, value_key: str) -> str:
    """返回 correct / wrong / missing / n/a。"""
    if truth_value is None:
        return "n/a"
    if not item.get("mentioned"):
        return "missing"
    value = _num(item.get(value_key))
    direction = item.get("direction") if item.get("direction") in ("up", "down", "flat") else _direction(value)
    if direction is None:
        return "missing"  # 提到了但说不出方向，等同没提供信息
    if direction != _direction(truth_value):
        return "wrong"
    if value is not None and abs(value - truth_value) > tolerance:
        return "wrong"
    return "correct"


def score_report(extraction: dict, truth: dict) -> dict[str, str]:
    results: dict[str, str] = {}
    for key in ("revenue", "net_income"):
        item = extraction.get(key) or {}
        period = item.get("period")
        candidates = {"quarter": [truth[key]["quarter"]], "annual": [truth[key]["annual"]]}.get(
            str(period), [truth[key]["quarter"], truth[key]["annual"]]
        )
        candidates = [c for c in candidates if c is not None]
        if not candidates:
            results[key] = "n/a"
            continue
        outcomes = [_score_directional(item, c, YOY_TOLERANCE_PP, "yoy_pct") for c in candidates]
        results[key] = "correct" if "correct" in outcomes else outcomes[0]
        # SEC结构化数据比EPS/股价数据落后一个季度以上时（KO：结构化数据只到Q1，财报
        # 原文已经是Q2），简报可能引用了原文里更新一季的数字——跟过时的标准答案对不上
        # 不能算错，记为无法核实（给已有20份简报评分时，KO两份都因此被误判）
        if results[key] == "wrong" and natural_traps(truth)["period_gap"]:
            results[key] = "unverifiable"

    eps = extraction.get("eps_surprise") or {}
    if truth["eps_verdict"] is None:
        results["eps_surprise"] = "n/a"
    elif not eps.get("mentioned") or eps.get("verdict") not in ("beat", "miss", "inline"):
        results["eps_surprise"] = "missing"
    elif eps["verdict"] != truth["eps_verdict"]:
        results["eps_surprise"] = "wrong"
    else:
        value = _num(eps.get("surprise_pct"))
        truth_pct = truth["eps_surprise_pct"]
        results["eps_surprise"] = (
            "wrong" if value is not None and truth_pct is not None and abs(abs(value) - abs(truth_pct)) > SURPRISE_TOLERANCE_PP else "correct"
        )

    price = extraction.get("price_reaction") or {}
    results["price_reaction"] = _score_directional(price, truth["price_change_pct"], PRICE_TOLERANCE_PP, "change_pct")
    ratio = truth["volume_ratio"]
    results["volume_spike"] = "n/a" if ratio is None or ratio < VOLUME_SPIKE_RATIO else ("correct" if price.get("mentions_volume") else "missing")

    sector = extraction.get("sector") or {}
    if truth["sector_quadrant"] is None:
        results["sector"] = "n/a"
    elif not sector.get("mentioned") or sector.get("quadrant") not in ("leading", "weakening", "lagging", "improving"):
        results["sector"] = "missing"
    else:
        results["sector"] = "correct" if sector["quadrant"] == truth["sector_quadrant"] else "wrong"

    traps = natural_traps(truth)
    results["period_gap"] = ("correct" if period_gap_mentioned(extraction) else "missing") if traps["period_gap"] else "n/a"
    results["cash_flow_divergence"] = (
        ("correct" if _anchored(extraction, "cash_flow_divergence", ("现金流",)) else "missing")
        if has_cash_flow_divergence(truth)
        else "n/a"
    )
    results["reaction_reversal"] = (
        ("correct" if _anchored(extraction, "reaction_reversal", _REVERSAL_ANCHORS) else "missing")
        if has_reaction_reversal(truth)
        else "n/a"
    )
    return results


_REVERSAL_ANCHORS = ("回吐", "回补", "收复", "反转", "吐回", "修复", "回落", "抹去", "回撤")


def _anchored(extraction: dict, key: str, anchors: tuple[str, ...]) -> bool:
    """跟期间错位同一个思路：评分者说"提到了"，引用的原话里还得真的出现相关字眼。"""
    item = extraction.get(key) or {}
    return bool(item.get("mentioned")) and any(anchor in str(item.get("quote") or "") for anchor in anchors)


# 期间错位要说的是"财务数据"和"EPS超预期/股价反应"不是同一期。人工抽查试跑结果时
# 发现评分者会把"现金流和营收不同期"这类别的期间差也算进来——用代码加一道约束：
# 引用的原话里必须出现EPS/预期/股价这类词，否则不算
_PERIOD_GAP_ANCHORS = ("EPS", "预期", "股价", "市场反应", "财报发布", "业绩快报")


def period_gap_mentioned(extraction: dict) -> bool:
    item = extraction.get("period_gap") or {}
    return bool(item.get("mentioned")) and any(anchor in str(item.get("quote") or "") for anchor in _PERIOD_GAP_ANCHORS)


def score_traps(extraction: dict, truth: dict, injected: str | None) -> dict[str, bool | None]:
    """陷阱场景：True=通过，False=没通过，None=简报根本没涉及这个话题。只返回这次
    运行适用的陷阱。

    "没涉及"必须单独算：口径陷阱和埋矛盾文件陷阱的判定是"没犯错就算通过"，一份
    完全不谈EPS的简报会被当成"通过了口径陷阱"——开跑前空跑时发现的，拆分结果全空
    陷阱通过率却是25%。信号冲突/期间错位/数据缺失这三类本身就要求"必须指出"，
    没提就是没通过，不存在"没涉及"。"""
    out: dict[str, bool | None] = {}
    traps = natural_traps(truth) if injected is None else {}
    if traps.get("eps_basis"):
        basis = extraction.get("eps_basis") or {}
        engaged = (extraction.get("eps_surprise") or {}).get("mentioned") or basis.get("compares_gaap_with_consensus") or basis.get("notes_basis_difference")
        out["eps_basis"] = (not basis.get("compares_gaap_with_consensus") or bool(basis.get("notes_basis_difference"))) if engaged else None
    if traps.get("beat_vs_price_conflict"):
        out["beat_vs_price_conflict"] = bool((extraction.get("beat_vs_price_conflict") or {}).get("mentioned"))
    if traps.get("period_gap"):
        out["period_gap"] = period_gap_mentioned(extraction)
    if injected == "missing_eps_data":
        eps = extraction.get("eps_surprise") or {}
        fabricated = eps.get("mentioned") and eps.get("verdict") in ("beat", "miss", "inline")
        out["missing_eps_data"] = not fabricated and bool((extraction.get("data_gap_eps") or {}).get("mentioned"))
    if injected == "planted_conflict":
        revenue = extraction.get("revenue") or {}
        cited_yoy = _num(revenue.get("yoy_pct"))
        used_planted = cited_yoy is not None and abs(cited_yoy - truth["planted_yoy"]) <= YOY_TOLERANCE_PP
        flagged = bool((extraction.get("conflicting_sources") or {}).get("mentioned"))
        engaged = flagged or cited_yoy is not None
        out["planted_conflict"] = (flagged or not used_planted) if engaged else None
    return out


# ---------- 运行 ----------


class DegradedRunError(Exception):
    """运行没有报错，但不是在完整条件下完成的（比如Best-of-N缺了裁判分），不计入评分。"""

    def __init__(self, message: str, meta: dict):
        super().__init__(message)
        self.meta = meta


@dataclass
class RunRecord:
    ticker: str
    config: str
    repeat: int
    trap: str | None
    report: str | None
    cost_usd: float | None
    error: str | None = None
    extraction: dict | None = None
    extraction_second: dict | None = None
    invalid_quotes: list[str] = field(default_factory=list)
    item_scores: dict[str, str] = field(default_factory=dict)
    trap_scores: dict[str, bool | None] = field(default_factory=dict)
    meta: dict = field(default_factory=dict)
    duration_s: float | None = None


async def run_config(config: str, ticker: str) -> tuple[str | None, dict]:
    if config == "normal":
        result = await loop.run_agent_loop(ticker)
        return result.final_report, {"turns": result.turns_used}
    if config == "best_of_n":
        bon = await run_best_of_n(ticker)
        candidate_errors = [c.error for c in bon.candidates if c.error]
        missing_judge = sum(1 for c in bon.candidates if not c.error and c.llm_score is None)
        # 产品代码里裁判调用失败（包括余额不足）会被吞掉、返回空分，Best-of-N照样
        # 选出一个候选——余额用完时这次运行看起来完全正常，其实是在没有裁判分的
        # 情况下选的。这里把这种"静默降级"显式标出来，不计入评分
        for err in candidate_errors:
            if _is_fatal(RuntimeError(err)):
                raise RuntimeError(f"Best-of-N 有候选因账户层面的错误失败：{err}")
        meta = {"selected_index": bon.selected_index, "candidate_errors": candidate_errors, "missing_judge_scores": missing_judge}
        if candidate_errors or missing_judge:
            raise DegradedRunError(f"Best-of-N 降级运行：{len(candidate_errors)} 个候选失败、{missing_judge} 个候选缺裁判分", meta)
        return bon.selected.final_report, meta
    dr = await run_draft_review(ticker)
    return dr.final_report, {
        "critique": dr.critique,
        "fell_back_to_draft": dr.fell_back_to_draft,
        "review_failed": dr.review_failed,
        "review_stop_reason": dr.review_stop_reason,
        "review_output_tokens": dr.review_output_tokens,
        "revised": dr.final_report != dr.draft.final_report,
        "draft_report": dr.draft.final_report,
    }


def _budget_exceeded(max_cost: float | None) -> bool:
    if max_cost is None:
        return False
    spent = usage_cost(_total_usage)
    return spent is not None and spent >= max_cost


async def run_one(
    ticker: str,
    config: str,
    repeat: int,
    trap: str | None,
    truth: dict,
    extract_twice: bool,
    semaphore: asyncio.Semaphore,
    max_cost: float | None = None,
    existing: RunRecord | None = None,
) -> RunRecord:
    """`existing`是之前已经生成了简报、只是评分失败的记录——续跑时只补评分，不重跑分析。"""
    async with semaphore:
        if _abort.is_set():
            return RunRecord(ticker, config, repeat, trap, None, None, error="已中止：前面的运行遇到账户层面的错误")
        if _budget_exceeded(max_cost):
            if not _abort.is_set():
                _abort.set()
                print(f"\n累计花费已达到上限 ${max_cost}，停止发起剩余运行\n", flush=True)
            return RunRecord(ticker, config, repeat, trap, None, None, error="已中止：达到预算上限")
        if existing is not None:
            record = existing
            record.error = None
        else:
            sink: dict[str, LLMUsage] = {}
            _usage_sink.set(sink)
            _active_trap.set(trap)
            started = time.monotonic()
            try:
                report, meta = await asyncio.wait_for(run_config(config, ticker), timeout=RUN_TIMEOUT_SECONDS)
            except DegradedRunError as exc:
                return RunRecord(ticker, config, repeat, trap, None, usage_cost(sink), error=f"降级运行：{exc}", meta=exc.meta)
            except TimeoutError:
                return RunRecord(ticker, config, repeat, trap, None, usage_cost(sink), error=f"超时：超过{RUN_TIMEOUT_SECONDS}秒")
            except Exception as exc:  # noqa: BLE001 - 单次运行失败不影响其它运行
                _note_fatal(exc)
                return RunRecord(ticker, config, repeat, trap, None, usage_cost(sink), error=str(exc))
            finally:
                _usage_sink.set(None)  # 评分的花费不算进被评测配置的成本
            record = RunRecord(ticker, config, repeat, trap, report, usage_cost(sink), meta=meta,
                               duration_s=round(time.monotonic() - started, 1))
        report = record.report
        if not report:
            record.error = "没有产出简报"
            return record
        # 评分阶段同样会调用LLM：第二次全量运行时余额在评分阶段用完，这里没捕获，
        # 整个脚本崩溃、已经跑完的6次结果全丢。简报本身保留在记录里，之后可以补评
        try:
            extraction = await extract(report)
            if extraction is None:
                record.error = "评分者输出无法解析"
                return record
            record.extraction, record.invalid_quotes = verify_quotes(extraction, report)
            if extract_twice:
                second = await extract(report)
                record.extraction_second = verify_quotes(second, report)[0] if second else None
            scoring_basis = merge_extractions(record.extraction, record.extraction_second)
            record.item_scores = score_report(scoring_basis, truth)
            record.trap_scores = score_traps(scoring_basis, truth, trap)
        except Exception as exc:  # noqa: BLE001 - 评分失败不影响其它运行
            _note_fatal(exc)
            record.error = f"评分失败：{exc}"
            return record
        print(f"  {ticker:5} {config:12} #{repeat} {('['+trap+'] ') if trap else ''}"
              f"{sum(v == 'correct' for v in record.item_scores.values())} 对 / {sum(v == 'wrong' for v in record.item_scores.values())} 错 / "
              f"{sum(v == 'missing' for v in record.item_scores.values())} 漏 | 陷阱 {record.trap_scores} | ${record.cost_usd}", flush=True)
        return record


# "有没有提到"这类偏判断的字段：试跑时两次抽取的分歧全部集中在这里（数字、方向、
# 板块阶段这些事实字段两次都一致）。抽了两次时，任意一次给出了"提到了"就采用它。
# 最初用的是保守判定（两次都说提到了才算），给已有20份简报评分时发现它造成了
# 误判：MU/KO的简报明明写了期间错位、原话清清楚楚，只因为第二次抽取漏掉就被判成
# 没提到。误判为"提到了"的风险已经被另外两道关卡挡住——原话必须由代码核对确实
# 出现在简报里，期间错位/现金流背离/回吐这几项还要求原话里有相关字眼
_JUDGMENT_FIELDS = ("period_gap", "beat_vs_price_conflict", "data_gap_eps", "conflicting_sources",
                    "cash_flow_divergence", "reaction_reversal")


def merge_extractions(first: dict, second: dict | None) -> dict:
    if second is None:
        return first
    merged = dict(first)
    for key in _JUDGMENT_FIELDS:
        if not (first.get(key) or {}).get("mentioned") and (second.get(key) or {}).get("mentioned"):
            merged[key] = second[key]
    return merged


def extraction_agreement(records: list[RunRecord]) -> float | None:
    keys = ("revenue", "net_income", "eps_surprise", "price_reaction", "sector", "period_gap", "beat_vs_price_conflict",
            "data_gap_eps", "conflicting_sources", "cash_flow_divergence", "reaction_reversal")
    same = total = 0
    for r in records:
        if not r.extraction or not r.extraction_second:
            continue
        for k in keys:
            a, b = r.extraction.get(k) or {}, r.extraction_second.get(k) or {}
            fields = ("mentioned", "direction", "verdict", "quadrant")
            total += 1
            same += all(a.get(f) == b.get(f) for f in fields)
    return same / total if total else None


def summarize(records: list[RunRecord]) -> dict:
    by_config: dict[str, list[RunRecord]] = defaultdict(list)
    for r in records:
        by_config[r.config].append(r)
    summary = {}
    for config, recs in by_config.items():
        main = [r for r in recs if r.trap is None and r.item_scores]
        items = [v for r in main for v in r.item_scores.values() if v not in ("n/a", "unverifiable")]
        per_run_acc = [
            sum(v == "correct" for v in r.item_scores.values()) / max(1, sum(v not in ("n/a", "unverifiable") for v in r.item_scores.values()))
            for r in main
        ]
        # 同一只股票重复运行之间的波动：先按股票分组求标准差，再取平均
        by_ticker: dict[str, list[float]] = defaultdict(list)
        for r, acc in zip(main, per_run_acc, strict=True):
            by_ticker[r.ticker].append(acc)
        spreads = [statistics.pstdev(v) for v in by_ticker.values() if len(v) > 1]
        trap_results = [ok for r in recs for ok in r.trap_scores.values() if ok is not None]
        trap_not_engaged = sum(1 for r in recs for ok in r.trap_scores.values() if ok is None)
        costs = [r.cost_usd for r in recs if r.cost_usd is not None]
        durations = [r.duration_s for r in recs if r.duration_s is not None and not r.error]
        wrong_items: dict[str, int] = defaultdict(int)
        for r in main:
            for k, v in r.item_scores.items():
                if v == "wrong":
                    wrong_items[k] += 1
        summary[config] = {
            "runs": len(recs),
            "failed_runs": sum(1 for r in recs if r.error),
            "accuracy": round(sum(v == "correct" for v in items) / len(items), 3) if items else None,
            "error_rate": round(sum(v == "wrong" for v in items) / len(items), 3) if items else None,
            "missing_rate": round(sum(v == "missing" for v in items) / len(items), 3) if items else None,
            "run_to_run_spread": round(statistics.fmean(spreads), 3) if spreads else None,
            "trap_pass_rate": round(sum(trap_results) / len(trap_results), 3) if trap_results else None,
            "trap_n": len(trap_results),
            "trap_not_engaged": trap_not_engaged,
            "mean_cost_usd": round(statistics.fmean(costs), 4) if costs else None,
            "median_duration_s": round(statistics.median(durations), 1) if durations else None,
            "wrong_by_item": dict(wrong_items),
            "invalid_quotes": sum(len(r.invalid_quotes) for r in recs),
            "unverifiable_items": sum(1 for r in main for v in r.item_scores.values() if v == "unverifiable"),
            "review_failed": sum(1 for r in recs if r.meta.get("review_failed")),
            "revised": sum(1 for r in recs if r.meta.get("revised")),
        }
    return summary


def _run_accuracy(record: RunRecord) -> float | None:
    applicable = [v for v in record.item_scores.values() if v not in ("n/a", "unverifiable")]
    return sum(v == "correct" for v in applicable) / len(applicable) if applicable else None


def sign_test_p(wins: int, losses: int) -> float:
    """双侧符号检验（平局不计）。"""
    n = wins + losses
    if n == 0:
        return 1.0
    k = min(wins, losses)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2**n)


def paired_comparison(records: list[RunRecord], configs: list[str]) -> list[dict]:
    """同一只股票上两两比较：先把每只股票重复运行的准确率取平均（同一只股票的多次
    运行不是独立样本，独立单位是股票），再数胜/平/负并做符号检验。这样能排除
    "这只股票本来就难"的干扰，比直接比较汇总平均值更严谨。只用常规运行，不含注入陷阱。"""
    per_ticker: dict[tuple[str, str], list[float]] = defaultdict(list)
    for r in records:
        acc = _run_accuracy(r) if r.trap is None and r.item_scores else None
        if acc is not None:
            per_ticker[(r.config, r.ticker)].append(acc)
    means = {key: statistics.fmean(v) for key, v in per_ticker.items()}
    tickers = sorted({t for _, t in means})
    results = []
    for i, a in enumerate(configs):
        for b in configs[i + 1 :]:
            wins = losses = ties = 0
            for t in tickers:
                if (a, t) not in means or (b, t) not in means:
                    continue
                diff = means[(a, t)] - means[(b, t)]
                if abs(diff) < 1e-9:
                    ties += 1
                elif diff > 0:
                    wins += 1
                else:
                    losses += 1
            results.append({"a": a, "b": b, "a_wins": wins, "b_wins": losses, "ties": ties, "p_value": round(sign_test_p(wins, losses), 3)})
    return results


def print_summary(summary: dict, agreement: float | None) -> None:
    print("\n配置          准确率  错误率  遗漏率  重复波动  陷阱通过率   平均成本  耗时中位数  失败")
    for config, s in summary.items():
        def fmt(v: float | None) -> str:
            return f"{v:.0%}" if v is not None else "—"
        print(
            f"{config:12} {fmt(s['accuracy']):>6} {fmt(s['error_rate']):>6} {fmt(s['missing_rate']):>6} "
            f"{fmt(s['run_to_run_spread']):>8} {fmt(s['trap_pass_rate']):>8}(n={s['trap_n']},未涉及{s['trap_not_engaged']}) "
            f"{('$' + format(s['mean_cost_usd'], '.3f')) if s['mean_cost_usd'] is not None else '—':>9} "
            f"{(format(s['median_duration_s'], '.0f') + 's') if s['median_duration_s'] is not None else '—':>9} {s['failed_runs']:>4}"
        )
        if s["wrong_by_item"]:
            print(f"{'':12} 出错的检查项：{s['wrong_by_item']}")
        if s["unverifiable_items"]:
            print(f"{'':12} 无法核实（标准答案落后于简报引用的期间）：{s['unverifiable_items']} 项，不计入准确率")
        if config == "draft_review":
            print(f"{'':12} 审阅后实际改动 {s['revised']} 次，审阅失败 {s['review_failed']} 次（失败时沿用草稿）")
    if agreement is not None:
        print(f"\n评分者两次抽取的一致率：{agreement:.0%}")
    print("准确率之间的差距要大于'重复波动'才有意义。成本只算被评测配置本身，不含评分；耗时是并发4路下的墙钟时间。")


def print_paired(pairs: list[dict]) -> None:
    print("\n同一只股票上两两比较（按股票取平均准确率，平局不计，符号检验）：")
    for p in pairs:
        print(f"  {p['a']} vs {p['b']}：{p['a']} 胜 {p['a_wins']}、{p['b']} 胜 {p['b_wins']}、平 {p['ties']}，p={p['p_value']}")
    print("  独立样本只有约10只股票，除非结果非常一边倒，否则不会显著。")


def load_partial_records(path: Path) -> list[RunRecord]:
    """逐行读，坏行跳过——进程被杀/电脑休眠时最后一行可能只写了一半。"""
    if not path.exists():
        return []
    records = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            records.append(RunRecord(**json.loads(line)))
        except (json.JSONDecodeError, TypeError):
            print(f"跳过一行无法解析的记录：{line[:80]}…", file=sys.stderr)
    return records


def rescore(run_dir: Path) -> int:
    """用保存下来的抽取结果和标准答案重新算分（判分规则修正后不用重跑分析和评分者）。"""
    truths = json.loads((run_dir / "truths.json").read_text())
    raw = json.loads((run_dir / "records.json").read_text())
    records = []
    for item in raw:
        record = RunRecord(**item)
        if record.extraction:
            truth = truths[record.ticker]
            scoring_basis = merge_extractions(record.extraction, record.extraction_second)
            record.item_scores = score_report(scoring_basis, truth)
            record.trap_scores = score_traps(scoring_basis, truth, record.trap)
        records.append(record)
    (run_dir / "records_rescored.json").write_text(json.dumps([r.__dict__ for r in records], ensure_ascii=False, indent=2))
    summary = summarize(records)
    print_summary(summary, extraction_agreement(records))
    print_paired(paired_comparison(records, sorted({r.config for r in records})))
    (run_dir / "summary_rescored.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def records_from_blind_comparison(raw_path: Path) -> list[RunRecord]:
    """把之前盲评实验已经生成好的简报（build_blind_comparison.py 的 raw.jsonl：
    每只股票一份普通分析 + 一份深度分析）转成待评分的记录——不重新生成简报，
    只花评分的钱。"""
    records = []
    for line in raw_path.read_text().splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        for source_key, config in (("normal", "normal"), ("deep", "best_of_n")):
            report = (item.get(source_key) or {}).get("final_report")
            if report:
                records.append(RunRecord(item["ticker"], config, 0, None, report, None, error="评分失败：尚未评分"))
    return records


async def grade_existing(args: argparse.Namespace) -> int:
    install_instrumentation()
    pending = records_from_blind_comparison(Path(args.grade_existing))
    out_dir = OUT_ROOT / (datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + "-existing")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"给已有的 {len(pending)} 份简报评分（不重新生成），明细：{out_dir}", flush=True)
    truths = {t: await fetch_truth(t) for t in sorted({r.ticker for r in pending})}
    (out_dir / "truths.json").write_text(json.dumps(truths, ensure_ascii=False, indent=2))
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_PIPELINES)
    records = list(await asyncio.gather(*(
        run_one(r.ticker, r.config, r.repeat, None, truths[r.ticker], args.extract_twice, semaphore, args.max_cost, r) for r in pending
    )))
    (out_dir / "records.json").write_text(json.dumps([r.__dict__ for r in records], ensure_ascii=False, indent=2))
    print(f"\n评分花费：${usage_cost(_total_usage)}")
    if _abort.is_set():
        print("评分中止，结果不完整", file=sys.stderr)
        return 1
    summary = summarize(records)
    print_summary(summary, extraction_agreement(records))
    print_paired(paired_comparison(records, ["normal", "best_of_n"]))
    print("注意：这批简报生成于修复数据层bug、加第7道质量检查之前，两种配置之间的比较是公平的，但绝对分数不代表当前版本。")
    return 0


async def main_async(args: argparse.Namespace) -> int:
    if args.rescore:
        return rescore(Path(args.rescore))
    if args.grade_existing:
        return await grade_existing(args)
    install_instrumentation()
    rng = random.Random(args.seed)
    tickers = args.tickers[:1] if args.pilot else args.tickers
    configs = args.configs
    repeats = 1 if args.pilot else args.repeats
    out_dir = Path(args.resume) if args.resume else OUT_ROOT / datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out_dir.mkdir(parents=True, exist_ok=True)
    # 每完成一次运行就追加写盘：中途因为任何原因停下，已经花钱得到的结果都不丢，
    # 用--resume指向同一个目录可以只补跑没完成的部分
    partial_path = out_dir / "records.jsonl"
    previous = load_partial_records(partial_path)
    done = [r for r in previous if not r.error]
    # 简报已经生成、只是评分失败的：续跑时只补评分，不重跑分析
    regrade = {(r.ticker, r.config, r.repeat, r.trap): r for r in previous if r.report and (r.error or "").startswith("评分失败")}
    if previous:
        partial_path.write_text("".join(json.dumps(r.__dict__, ensure_ascii=False) + "\n" for r in done))
        print(f"从 {out_dir} 接着跑：已完成 {len(done)} 次，{len(regrade)} 次只需补评分", flush=True)

    truths_path = out_dir / "truths.json"
    if truths_path.exists():
        # 续跑时沿用第一次算好的标准答案——中途如果有新的财报/行情进来，前后两段
        # 用的评分标准就不一致了
        truths = json.loads(truths_path.read_text())
        print("沿用已保存的标准答案", flush=True)
    else:
        print("计算标准答案……", flush=True)
        truths = {t: await fetch_truth(t) for t in sorted(set(tickers) | {t for ts in INJECTED_TRAPS.values() for t in ts})}
    for t in tickers:
        print(f"  {t}: 天然陷阱 {[k for k, v in natural_traps(truths[t]).items() if v]}")
    for ts in INJECTED_TRAPS.values():
        for t in ts:
            _planted_text[t] = planted_note(t, truths[t])
    truths_path.write_text(json.dumps(truths, ensure_ascii=False, indent=2))

    jobs: list[tuple[str, str, int, str | None]] = [(t, c, i, None) for t in tickers for c in configs for i in range(repeats)]
    if not args.skip_traps and not args.pilot:
        jobs += [(t, c, i, trap) for trap, ts in INJECTED_TRAPS.items() for t in ts for c in configs for i in range(repeats)]
    rng.shuffle(jobs)  # 打乱顺序，避免某种配置总在数据源缓存冷/热的时段跑
    finished = {(r.ticker, r.config, r.repeat, r.trap) for r in done}
    jobs = [job for job in jobs if job not in finished]
    print(f"\n共 {len(jobs)} 次运行，并发 {MAX_CONCURRENT_PIPELINES}\n", flush=True)

    semaphore = asyncio.Semaphore(MAX_CONCURRENT_PIPELINES)

    async def run_and_save(t: str, c: str, i: int, trap: str | None) -> RunRecord:
        try:
            record = await run_one(t, c, i, trap, truths[t], args.extract_twice, semaphore, args.max_cost, regrade.get((t, c, i, trap)))
        except Exception as exc:  # noqa: BLE001 - 兜底：任何意料之外的异常都不能把整个评测带崩
            record = RunRecord(t, c, i, trap, None, None, error=f"意外错误：{exc!r}")
        if not (record.error or "").startswith("已中止"):
            with partial_path.open("a") as f:
                f.write(json.dumps(record.__dict__, ensure_ascii=False) + "\n")
        return record

    records = done + list(await asyncio.gather(*(run_and_save(t, c, i, trap) for t, c, i, trap in jobs)))
    print(f"\n本次累计花费（含评分）：${usage_cost(_total_usage)}", flush=True)

    (out_dir / "records.json").write_text(json.dumps([r.__dict__ for r in records], ensure_ascii=False, indent=2))
    if _abort.is_set():
        print(f"\n评测中止（账户层面的错误或达到预算上限），结果不完整，不汇总。用 --resume {out_dir} 接着跑", file=sys.stderr)
        return 1
    summary = summarize(list(records))
    agreement = extraction_agreement(list(records))
    pairs = paired_comparison(list(records), configs)
    print_summary(summary, agreement)
    print_paired(pairs)
    print(f"\n明细：{out_dir}")
    RUNS_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with RUNS_LOG_PATH.open("a") as f:
        f.write(json.dumps({"timestamp": datetime.now(UTC).isoformat(), "dir": str(out_dir), "tickers": tickers, "configs": configs,
                            "repeats": repeats, "summary": summary, "paired": pairs, "extraction_agreement": agreement}, ensure_ascii=False) + "\n")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tickers", type=lambda s: [t.strip().upper() for t in s.split(",")], default=DEFAULT_TICKERS)
    parser.add_argument("--configs", type=lambda s: [c.strip() for c in s.split(",")], default=list(ALL_CONFIGS))
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--pilot", action="store_true", help="1只股票×每种配置×1次，不跑注入陷阱")
    parser.add_argument("--skip-traps", action="store_true")
    parser.add_argument("--extract-twice", action="store_true", help="每份简报让评分者拆两次，报告一致率")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--rescore", default=None, help="对已有运行目录按当前判分规则重新算分")
    parser.add_argument("--resume", default=None, help="指向之前中断的运行目录，只补跑没成功完成的部分")
    parser.add_argument("--max-cost", type=float, default=None, help="累计花费（含评分）达到这个美元数就停止发起新运行")
    parser.add_argument("--grade-existing", default=None, help="只给已有简报评分，比如 .cache/blind_comparison/raw.jsonl")
    return asyncio.run(main_async(parser.parse_args()))


if __name__ == "__main__":
    sys.exit(main())
