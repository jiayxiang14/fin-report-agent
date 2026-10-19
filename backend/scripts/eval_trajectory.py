"""全链路评测：按环节看Agent的运行轨迹，而不是只看最终简报。

eval_report_rubric.py 只看结果（简报里的事实对不对），中间哪一步出了问题看不到。
这个脚本读每次运行的完整记录（AgentRunResult：工具调用序列、参数、是否报错、
7道质量检查的触发/解决情况、终止原因、轮数），按环节算指标，全部由代码判定，
不调用LLM：

| 环节       | 指标 |
|------------|------|
| 输入处理   | 异常输入（不存在的代码、ETF）有没有如实说明，而不是编出财务数字 |
| 工具选择   | 对照人工标注的精确率/召回率；重复调用率；自我核查是否发生在写出草稿之后 |
| 参数生成   | 参数是否符合工具定义（必填项、枚举值、类型）；ticker是否跟分析对象一致 |
| 结果解析   | 简报里数字的可追溯率（能在工具原始数据里找到依据的比例） |
| 纠错迭代   | 工具失败后的处理（重试成功/如实说明/都没有）；每道质量检查"触发后真的解决了"的比例 |
| 任务终止   | 完成率、终止原因分布、超出最大轮数的比例、轮数分布、提前收尾率 |
| 失败归因   | 每次没做好的运行，标出最先出问题的环节 |

这些指标衡量的是"过程有没有按规矩来"，不等于"分析得好"。

两种用法（在 backend/ 目录下）：
    python scripts/eval_trajectory.py --run                 # 采集：跑一批新的普通分析（含故障注入），保存完整记录后分析
    python scripts/eval_trajectory.py --records <jsonl>     # 只分析已有的完整记录，不花钱
    python scripts/eval_trajectory.py --from-traces         # 从trace日志里找真实的普通分析来分析，不花钱

采集模式会真实调用LLM和数据源（普通分析每次约$0.05），默认预算上限$3，遇到余额不足
立即停止，每完成一次就写盘，可以用 --resume 接着跑。不接入CI。
"""

from __future__ import annotations

import argparse
import asyncio
import contextvars
import glob
import json
import re
import statistics
import sys
import time
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.models.agent import AgentRunResult, TranscriptEntry  # noqa: E402
from app.services.agent import loop  # noqa: E402
from app.services.agent.loop import MAX_TURNS  # noqa: E402
from app.services.agent.tools import TOOL_SCHEMAS  # noqa: E402
from app.services.agent.traceability import report_mentions_value  # noqa: E402
from app.services.polygon_client import CACHE_DIR  # noqa: E402
from scripts import eval_report_rubric as rubric  # noqa: E402
from scripts.eval_tool_selection import (  # noqa: E402
    _actual_tools_called,
    _load_eval_set,
    _score_tool_selection,
)

OUT_ROOT: Path = CACHE_DIR / "eval_trajectory"
TRACE_DIR: Path = CACHE_DIR / "traces"

SELF_VERIFICATION_TOOL = "verify_number"
TRACEABILITY_OK_RATIO = 0.6  # 跟loop.py的TRACEABILITY_GATE_MIN_RATIO同一量级：低于它算"结果解析"环节出问题
DEFAULT_MAX_COST = 3.0
MAX_CONCURRENT = 4
RUN_TIMEOUT_SECONDS = 20 * 60

# 采集模式的场景：(场景名, 股票列表, 重复次数)
TRANSIENT_FAILURE_TICKERS = ["AAPL", "KO", "NVDA"]
PERSISTENT_FAILURE_TICKERS = ["MSFT", "AMZN", "GOOGL"]
ABNORMAL_INPUTS = ["ZZZZ", "SPY"]  # 不存在的代码；ETF不是运营公司，没有财报
_GAP_DISCLOSURE = re.compile(r"未能获取|无法获取|获取失败|不存在|找不到|未找到|无法识别|没有.{0,6}(财报|财务数据)|不是.{0,6}(公司|个股)|ETF|基金")

_SCHEMAS = {schema["name"]: schema["input_schema"] for schema in TOOL_SCHEMAS}


# ---------- 各环节的判定（纯函数） ----------


def validate_params(tool_name: str, tool_input: dict) -> list[str]:
    """按工具定义检查参数，返回问题列表（空=合规）。execute_tool只在缺必填参数时
    报错，枚举值写错这类问题会悄悄退回默认值或在下游报一个看不出原因的错。"""
    schema = _SCHEMAS.get(tool_name)
    if schema is None:
        return [f"未知工具 {tool_name}"]
    problems = []
    for key in schema.get("required", []):
        if key not in tool_input:
            problems.append(f"缺少必填参数 {key}")
    for key, value in tool_input.items():
        spec = schema["properties"].get(key)
        if spec is None:
            problems.append(f"多余参数 {key}")
            continue
        if "enum" in spec and value not in spec["enum"]:
            problems.append(f"{key}={value!r} 不在允许值 {spec['enum']} 里")
        expected_type = spec.get("type")
        if expected_type == "string" and not isinstance(value, str):
            problems.append(f"{key} 应为字符串")
        if expected_type == "number" and (isinstance(value, bool) or not isinstance(value, int | float)):
            problems.append(f"{key} 应为数字")
    return problems


def ticker_mismatches(transcript: list[TranscriptEntry], ticker: str) -> list[str]:
    """工具参数里的ticker跟分析对象不一致的调用（同行对比等工具本来就只接受分析对象本身）。"""
    return [
        f"{e.tool_name}({e.tool_input.get('ticker')})"
        for e in transcript
        if "ticker" in e.tool_input and str(e.tool_input["ticker"]).upper() != ticker.upper()
    ]


def duplicate_calls(transcript: list[TranscriptEntry]) -> int:
    """同一个工具、同样的参数、之前已经成功调用过还再调一次——白花时间和额度。
    失败后用同样参数重试不算（那是纠错）。"""
    seen_success: set[str] = set()
    duplicates = 0
    for e in transcript:
        key = f"{e.tool_name}:{json.dumps(e.tool_input, sort_keys=True, ensure_ascii=False)}"
        if key in seen_success:
            duplicates += 1
        if not e.is_error:
            seen_success.add(key)
    return duplicates


def verify_after_draft(result: AgentRunResult) -> bool | None:
    """自我核查应该发生在写出草稿之后（核查的是草稿里的数字）。没做核查返回None。"""
    verify_turns = [e.turn for e in result.transcript if e.tool_name == SELF_VERIFICATION_TOOL and not e.is_error]
    if not verify_turns:
        return None
    draft_turns = [n.turn for n in result.reasoning_notes if "<conclusion>" in n.text]
    return bool(draft_turns) and min(draft_turns) <= max(verify_turns)


_ACTUAL_VALUE = re.compile(r'"actual_value":\s*(-?[\d.]+(?:e[+-]?\d+)?)')


def verified_values(result: AgentRunResult) -> list[float]:
    """核实过的数值：参数里的claimed_value，加上输出摘要里能读到的actual_value。"""
    values = []
    for e in result.transcript:
        if e.tool_name != SELF_VERIFICATION_TOOL or e.is_error:
            continue
        claimed = e.tool_input.get("claimed_value")
        if isinstance(claimed, int | float) and not isinstance(claimed, bool):
            values.append(float(claimed))
        match = _ACTUAL_VALUE.search(e.tool_output_summary)
        if match:
            values.append(float(match.group(1)))
    return values


def verification_covers_report(result: AgentRunResult) -> bool | None:
    """核实过的数字至少有一个写进了最终简报（方案B：不管先后顺序，管核实的对象）。
    没做核实返回None。"""
    values = verified_values(result)
    if not values:
        return None
    return any(report_mentions_value(result.final_report or "", v) for v in values)


def tool_failure_handling(result: AgentRunResult) -> dict[str, str]:
    """每个失败过的工具（verify_number除外）的处理结果：recovered=之后重新调用成功；
    disclosed=没恢复但简报写明了；ignored=都没有。"""
    outcomes: dict[str, str] = {}
    failed = [e for e in result.transcript if e.is_error and e.tool_name != SELF_VERIFICATION_TOOL]
    for entry in failed:
        later_success = any(
            e.tool_name == entry.tool_name and not e.is_error and e.turn >= entry.turn for e in result.transcript
        )
        if later_success:
            outcomes[entry.tool_name] = "recovered"
        elif result.data_gap_disclosure_gate_resolved:
            outcomes.setdefault(entry.tool_name, "disclosed")
        else:
            outcomes.setdefault(entry.tool_name, "ignored")
    return outcomes


_GATES = ("structure", "tool_coverage", "verification_mismatch", "traceability", "sentiment_consistency",
          "data_gap_disclosure", "verification_coverage")


def gate_outcomes(result: AgentRunResult) -> dict[str, str]:
    """每道质量检查：not_triggered / resolved（触发后解决了） / unresolved（触发了没解决）。"""
    outcomes = {}
    for gate in _GATES:
        triggered = getattr(result, f"{gate}_gate_triggered", None)
        if triggered is None:  # verification_mismatch的字段名没有_gate
            triggered = getattr(result, f"{gate}_triggered")
            resolved = getattr(result, f"{gate}_resolved")
        else:
            resolved = getattr(result, f"{gate}_gate_resolved")
        outcomes[gate] = "not_triggered" if not triggered else ("resolved" if resolved else "unresolved")
    return outcomes


def premature_stop(result: AgentRunResult) -> bool:
    """宣布完成了，但必需的基础数据没拿到、或者没做自我核查——信息不全就收尾。"""
    return result.completed and (not result.tool_coverage_gate_resolved or not result.self_verification_triggered)


def traceability_ratio(result: AgentRunResult) -> float | None:
    if not result.traceable_numbers_total:
        return None
    return result.traceable_numbers_matched / result.traceable_numbers_total


# 财务指标后面紧跟着数字——不存在的代码/ETF根本拿不到财务数据，写出这种句子就是编的。
# 最初用"数字可追溯率低"来近似判断"编了数字"，真实运行里误判了ZZZZ：它的简报如实
# 说明了拿不到任何数据，可追溯率低只是因为引用了板块轮动数据里的数字
_FINANCIAL_FIGURE = re.compile(r"(营收|营业收入|净利润|毛利|营业利润|每股收益|EPS|自由现金流)[^。；\n]{0,10}?[-+]?\d[\d,.]*\s*(%|亿|万|美元|\$)")


def abnormal_input_handled(result: AgentRunResult) -> bool:
    """不存在的代码/ETF：不能编出财务数字，简报要如实说明拿不到数据。"""
    report = result.final_report or ""
    return not _FINANCIAL_FIGURE.search(report) and bool(_GAP_DISCLOSURE.search(report))


def analyze_run(record: dict, expected_tools: dict[str, set[str]]) -> dict:
    """单次运行按环节打分，并找出最先出问题的环节。"""
    result = AgentRunResult(**record["result"])
    ticker = record["ticker"]
    scenario = record.get("scenario", "normal")
    out: dict[str, Any] = {"ticker": ticker, "scenario": scenario}

    # 输入处理
    if scenario == "abnormal_input":
        out["input_ok"] = abnormal_input_handled(result)

    # 工具选择
    if ticker in expected_tools and scenario == "normal":
        out["tool_selection"] = _score_tool_selection(expected_tools[ticker], _actual_tools_called(result.transcript))
    out["duplicate_calls"] = duplicate_calls(result.transcript)
    out["verify_after_draft"] = verify_after_draft(result)  # 只作参考，归因用下面的核实覆盖
    # 异常输入拿不到财务数据，核实的往往是0，这一项不适用
    out["verification_covers_report"] = None if scenario == "abnormal_input" else verification_covers_report(result)

    # 参数生成
    param_problems = [(e.tool_name, p) for e in result.transcript for p in validate_params(e.tool_name, e.tool_input)]
    out["calls"] = len(result.transcript)
    out["param_problems"] = param_problems
    out["ticker_mismatches"] = ticker_mismatches(result.transcript, ticker)

    # 结果解析
    out["traceability"] = traceability_ratio(result)

    # 纠错迭代
    out["tool_failures"] = tool_failure_handling(result)
    out["gates"] = gate_outcomes(result)

    # 任务终止
    out["completed"] = result.completed
    out["stop_reason"] = result.stop_reason
    out["turns_used"] = result.turns_used
    # 异常输入本来就拿不到财务数据，没收集齐就结束是对的，不算提前收尾
    out["premature_stop"] = premature_stop(result) and scenario != "abnormal_input"

    out["first_failure"] = first_failing_stage(out)
    return out


def first_failing_stage(run: dict) -> str | None:
    """按链路顺序找最先出问题的环节；全部正常返回None。"""
    if "input_ok" in run:
        # 异常输入只考察输入处理：拿不到财务数据时，可追溯率、"必须调用过财务工具"
        # 这类检查注定过不了，不代表后面的环节出了问题
        return None if run["input_ok"] else "输入处理"
    selection = run.get("tool_selection")
    if selection and selection["recall"] is not None and selection["recall"] < 1:
        return "工具选择"
    if run["param_problems"] or run["ticker_mismatches"]:
        return "参数生成"
    if run["traceability"] is not None and run["traceability"] < TRACEABILITY_OK_RATIO:
        return "结果解析"
    # 自我核查没核到简报里真正写的数字也归在这里（最初按"先核实、后写简报"的顺序
    # 判定，24次运行全部命中；按方案B修复后改成判定核实的对象，不再管先后顺序）
    if "ignored" in run["tool_failures"].values() or "unresolved" in run["gates"].values() or run["verification_covers_report"] is False:
        return "纠错迭代"
    if not run["completed"] or run["premature_stop"]:
        return "任务终止"
    return None


def _rate(values: list[bool]) -> float | None:
    return round(sum(values) / len(values), 3) if values else None


def summarize(runs: list[dict]) -> dict:
    normal = [r for r in runs if r["scenario"] == "normal"]
    selections = [r["tool_selection"] for r in normal if r.get("tool_selection")]
    failures = [outcome for r in runs for outcome in r["tool_failures"].values()]
    gate_stats: dict[str, Counter] = defaultdict(Counter)
    for r in runs:
        for gate, outcome in r["gates"].items():
            gate_stats[gate][outcome] += 1
    traceability = [r["traceability"] for r in runs if r["traceability"] is not None]
    by_scenario: dict[str, list[dict]] = defaultdict(list)
    for r in runs:
        by_scenario[r["scenario"]].append(r)
    return {
        "runs": len(runs),
        "by_scenario": {k: len(v) for k, v in by_scenario.items()},
        "输入处理": {"abnormal_input_ok_rate": _rate([r["input_ok"] for r in runs if "input_ok" in r])},
        "工具选择": {
            "precision_mean": round(statistics.fmean(s["precision"] for s in selections if s["precision"] is not None), 3) if selections else None,
            "recall_mean": round(statistics.fmean(s["recall"] for s in selections if s["recall"] is not None), 3) if selections else None,
            "most_missed": Counter(t for s in selections for t in s["missing"]).most_common(3),
            "most_extra": Counter(t for s in selections for t in s["extra"]).most_common(3),
            "runs_with_duplicate_calls": _rate([r["duplicate_calls"] > 0 for r in runs]),
            "verify_after_draft_rate（仅参考）": _rate([r["verify_after_draft"] for r in runs if r["verify_after_draft"] is not None]),
            "verification_covers_report_rate": _rate([r["verification_covers_report"] for r in runs if r["verification_covers_report"] is not None]),
        },
        "参数生成": {
            "calls": sum(r["calls"] for r in runs),
            "invalid_param_calls": sum(len(r["param_problems"]) for r in runs),
            "examples": [p for r in runs for p in r["param_problems"]][:5],
            "ticker_mismatch_calls": sum(len(r["ticker_mismatches"]) for r in runs),
        },
        "结果解析": {
            "traceability_mean": round(statistics.fmean(traceability), 3) if traceability else None,
            "runs_below_threshold": sum(1 for t in traceability if t < TRACEABILITY_OK_RATIO),
        },
        "纠错迭代": {
            "tool_failures": len(failures),
            "failure_handling": dict(Counter(failures)),
            "gates": {g: dict(c) for g, c in gate_stats.items() if c.get("resolved") or c.get("unresolved")},
        },
        "任务终止": {
            "completion_rate": _rate([r["completed"] for r in runs]),
            "stop_reasons": dict(Counter(r["stop_reason"] for r in runs)),
            "max_turns_exceeded": sum(1 for r in runs if r["stop_reason"] == "max_turns_exceeded"),
            "turns_median": statistics.median(r["turns_used"] for r in runs) if runs else None,
            "turns_max": max((r["turns_used"] for r in runs), default=None),
            "premature_stop_rate": _rate([r["premature_stop"] for r in runs]),
        },
        "失败归因": dict(Counter(r["first_failure"] or "全部正常" for r in runs)),
        # 某个问题在几乎所有运行里都出现时，会盖住其它问题——另给一份不计它的归因
        "失败归因（不计核实覆盖）": dict(Counter(first_failing_stage({**r, "verification_covers_report": None}) or "全部正常" for r in runs)),
    }


def print_report(summary: dict) -> None:
    print(f"\n共 {summary['runs']} 次运行：{summary['by_scenario']}\n")
    for stage in ("输入处理", "工具选择", "参数生成", "结果解析", "纠错迭代", "任务终止"):
        print(f"【{stage}】")
        for key, value in summary[stage].items():
            print(f"  {key}: {value}")
    for title in ("失败归因", "失败归因（不计核实覆盖）"):
        print(f"\n【{title}：最先出问题的环节】")
        for stage, count in sorted(summary[title].items(), key=lambda kv: -kv[1]):
            print(f"  {stage}: {count}")
    print(f"\n这些指标衡量的是过程有没有按规矩来，不等于分析得好。最大轮数 {MAX_TURNS}。")


# ---------- 数据来源 ----------


def records_from_traces(trace_dir: Path) -> list[dict]:
    """从trace日志里找真实的普通分析（有完整AgentRunResult、简报够长、真的调用过LLM）。"""
    records = []
    for path in sorted(glob.glob(str(trace_dir / "*.jsonl"))):
        events = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
        if any("candidate_index" in e for e in events) or not any(e.get("type") == "llm_usage" for e in events):
            continue
        done = [e for e in events if e.get("type") == "done"]
        result = (done[-1].get("result") or {}) if done else {}
        if "transcript" in result and len(result.get("final_report") or "") > 300:
            records.append({"ticker": result["ticker"], "scenario": "normal", "result": result, "source": Path(path).name})
    return records


def load_records(path: Path) -> list[dict]:
    records = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            print(f"跳过一行无法解析的记录：{line[:80]}…", file=sys.stderr)
    return records


# ---------- 采集：跑新的普通分析（含故障注入） ----------

_fault: contextvars.ContextVar[dict | None] = contextvars.ContextVar("fault", default=None)
_original_execute_tool = loop.execute_tool


async def _execute_tool_with_faults(name: str, tool_input: dict) -> tuple[str, bool]:
    fault = _fault.get()
    if fault is not None:
        if fault["kind"] == "transient_failure" and name == "get_financials" and not fault["fired"]:
            fault["fired"] = True  # 只让第一次调用失败，重试就能成功
            return "工具调用失败：SEC EDGAR 暂时不可用（HTTP 503），请稍后重试", True
        if fault["kind"] == "persistent_failure" and name == "get_peer_comparison":
            return "工具调用失败：同行数据服务不可用（HTTP 503）", True
    return await _original_execute_tool(name, tool_input)


def collection_jobs(repeats: int, only_tickers: set[str] | None = None) -> list[tuple[str, str, int]]:
    labeled = [entry["ticker"] for entry in _load_eval_set()]
    jobs = [(t, "normal", i) for t in labeled for i in range(repeats)]
    jobs += [(t, "transient_failure", 0) for t in TRANSIENT_FAILURE_TICKERS]
    jobs += [(t, "persistent_failure", 0) for t in PERSISTENT_FAILURE_TICKERS]
    jobs += [(t, "abnormal_input", 0) for t in ABNORMAL_INPUTS]
    if only_tickers:
        jobs = [job for job in jobs if job[0] in only_tickers]
    return jobs


async def collect(out_path: Path, repeats: int, max_cost: float, only_tickers: set[str] | None = None) -> None:
    rubric.install_instrumentation()  # 用量计费 + 熔断复用简报评测脚本的实现
    loop.execute_tool = _execute_tool_with_faults
    done = {(r["ticker"], r["scenario"], r.get("repeat", 0)) for r in load_records(out_path)} if out_path.exists() else set()
    jobs = [job for job in collection_jobs(repeats, only_tickers) if job not in done]
    print(f"需要跑 {len(jobs)} 次（已完成 {len(done)} 次），预算上限 ${max_cost}\n", flush=True)
    semaphore = asyncio.Semaphore(MAX_CONCURRENT)

    async def run(ticker: str, scenario: str, repeat: int) -> None:
        async with semaphore:
            if rubric._abort.is_set():
                return
            if rubric._budget_exceeded(max_cost):
                rubric._abort.set()
                print(f"累计花费达到上限 ${max_cost}，停止", flush=True)
                return
            _fault.set({"kind": scenario, "fired": False} if scenario in ("transient_failure", "persistent_failure") else None)
            started = time.monotonic()
            try:
                result = await asyncio.wait_for(loop.run_agent_loop(ticker), timeout=RUN_TIMEOUT_SECONDS)
            except Exception as exc:  # noqa: BLE001 - 单次失败不影响其它运行；余额不足触发熔断
                rubric._note_fatal(exc)
                print(f"  {ticker:5} {scenario:20} 失败：{str(exc)[:100]}", flush=True)
                return
            record = {
                "ticker": ticker,
                "scenario": scenario,
                "repeat": repeat,
                "duration_s": round(time.monotonic() - started, 1),
                "result": result.model_dump(),
            }
            with out_path.open("a") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
            print(f"  {ticker:5} {scenario:20} 完成，{result.turns_used} 轮，{result.stop_reason}", flush=True)

    await asyncio.gather(*(run(t, s, i) for t, s, i in jobs))
    print(f"\n累计花费：${rubric.usage_cost(rubric._total_usage)}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", action="store_true", help="采集新的运行记录后分析")
    parser.add_argument("--resume", default=None, help="接着往这个records.jsonl里采集")
    parser.add_argument("--records", default=None, help="只分析这个records.jsonl")
    parser.add_argument("--from-traces", action="store_true", help="只分析trace日志里的真实普通分析")
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--max-cost", type=float, default=DEFAULT_MAX_COST)
    parser.add_argument("--tickers", type=lambda s: {t.strip().upper() for t in s.split(",")}, default=None,
                        help="只跑这几只股票（试跑用）")
    args = parser.parse_args()

    if args.run or args.resume:
        out_path = Path(args.resume) if args.resume else OUT_ROOT / f"{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}.jsonl"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        asyncio.run(collect(out_path, args.repeats, args.max_cost, args.tickers))
        records = load_records(out_path)
        print(f"记录：{out_path}")
    elif args.records:
        records = load_records(Path(args.records))
    elif args.from_traces:
        records = records_from_traces(TRACE_DIR)
    else:
        parser.print_help()
        return 1

    if not records:
        print("没有可分析的记录", file=sys.stderr)
        return 1
    expected = {entry["ticker"]: set(entry["expected_tools"]) for entry in _load_eval_set()}
    runs = [analyze_run(r, expected) for r in records]
    print_report(summarize(runs))
    return 1 if rubric._abort.is_set() else 0


if __name__ == "__main__":
    sys.exit(main())
