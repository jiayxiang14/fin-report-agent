"""工具选择准确率探测器：不是CI门禁，是回答"Agent自主决策的任务成功率有没有
量化指标"这个问题用的工具——之前只有`eval_report_quality.py`评"简报写得好不好"，
没有任何东西评"该调用的工具有没有调对"，这是两件不同的事：报告可以写得很漂亮，
但背后信息收集本来就漏了一块。

用法（在 backend/ 目录下）：
    python scripts/eval_tool_selection.py
    python scripts/eval_tool_selection.py --tickers AAPL,NBIS
    python scripts/eval_tool_selection.py --repeats 3          # 每个ticker跑3次取均值
    python scripts/eval_tool_selection.py --temperature 0.7    # 不固定为0，想看采样多样性时用

标注数据在 `tests/fixtures/tool_selection_eval_set.json`——人工标注"这个ticker
该调用哪些工具"，覆盖了几类有代表性的场景（普通大盘股/非科技对照/AI基建主题
篮子成分股/双重身份/境外发行人+两层匹配都落空但业务真相关/两层匹配都落空且
业务确实无关），每条标注带`last_verified`日期，预设股票池/主题篮子成分变化
时需要人工复核，不是标一次就永久有效。

`verify_number`不参与精确率/召回率计算，单独统计"自我核查触发率"——它是几乎
每次都该触发的协议性动作，跟"该不该主动调用get_thematic_flow"这种情境判断
混进同一个分数会稀释信号。

跟 `eval_report_quality.py` 一样固定 temperature=0（消除"这次刚好抽到不同
工具调用组合"的采样噪声，不是prompt/代码变化导致的），支持 `--repeats` 多次
取均值。不接入CI，真实调用SEC EDGAR/Polygon/Alpha Vantage/LLM，真花钱真要等。

Alpha Vantage每天限额25次：`get_earnings_surprise`每次调用必打Alpha Vantage，
`get_price_reaction`只有SEC 8-K查不到业绩快报时才退化到Alpha Vantage兜底
（次数没法提前精确预知）——运行前会打印一个基于`get_earnings_surprise`预期
调用次数的下限估算，实际消耗可能更高，避免在真实使用高峰期跑、也别把
`--repeats`调太高。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.models.agent import TranscriptEntry  # noqa: E402
from app.services.agent.loop import run_agent_loop  # noqa: E402
from app.services.polygon_client import CACHE_DIR  # noqa: E402

EVAL_SET_PATH = Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "tool_selection_eval_set.json"
RUNS_LOG_PATH: Path = CACHE_DIR / "eval_tool_selection_runs.jsonl"

DEFAULT_TEMPERATURE = 0.0
SELF_VERIFICATION_TOOL = "verify_number"


def _load_eval_set() -> list[dict]:
    return json.loads(EVAL_SET_PATH.read_text())


def _actual_tools_called(transcript: list[TranscriptEntry]) -> set[str]:
    """只算成功调用（is_error=False）的工具名，`verify_number`单独统计，
    不算进这个集合——精确率/召回率评的是"信息收集覆盖面"，不是自我核查
    这个协议性动作。"""
    return {entry.tool_name for entry in transcript if not entry.is_error and entry.tool_name != SELF_VERIFICATION_TOOL}


def _self_verification_triggered(transcript: list[TranscriptEntry]) -> bool:
    return any(entry.tool_name == SELF_VERIFICATION_TOOL and not entry.is_error for entry in transcript)


def _score_tool_selection(expected: set[str], actual: set[str]) -> dict:
    """精确率/召回率分开报，不合成一个数——多调工具（成本浪费但不算错）和
    漏调工具（可能漏掉关键信号）是两种不同严重程度的失败，混在一起会丢信息。
    """
    matched = expected & actual
    precision = len(matched) / len(actual) if actual else None
    recall = len(matched) / len(expected) if expected else None
    return {
        "matched": sorted(matched),
        "extra": sorted(actual - expected),
        "missing": sorted(expected - actual),
        "precision": round(precision, 3) if precision is not None else None,
        "recall": round(recall, 3) if recall is not None else None,
    }


def _estimate_alpha_vantage_calls(eval_set: list[dict], repeats: int) -> int:
    count = sum(1 for entry in eval_set if "get_earnings_surprise" in entry["expected_tools"])
    return count * repeats


async def _run_one(ticker: str, expected_tools: list[str], temperature: float | None) -> dict:
    run_result = await run_agent_loop(ticker, temperature=temperature)
    actual = _actual_tools_called(run_result.transcript)
    score = _score_tool_selection(set(expected_tools), actual)
    return {
        "ticker": ticker,
        "completed": run_result.completed,
        "actual_tools": sorted(actual),
        "self_verification_triggered": _self_verification_triggered(run_result.transcript),
        "error": None,
        **score,
    }


def _error_result(ticker: str, exc: Exception) -> dict:
    return {
        "ticker": ticker,
        "completed": False,
        "actual_tools": None,
        "self_verification_triggered": None,
        "matched": None,
        "extra": None,
        "missing": None,
        "precision": None,
        "recall": None,
        "error": str(exc),
    }


async def _run_repeated(ticker: str, expected_tools: list[str], temperature: float | None, repeats: int) -> dict:
    runs: list[dict] = []
    for attempt in range(repeats):
        print(f"正在跑 {ticker} ...（{attempt + 1}/{repeats}）", file=sys.stderr)
        try:
            runs.append(await _run_one(ticker, expected_tools, temperature))
        except Exception as exc:  # noqa: BLE001 - 单次失败（上游限速/网络/账户余额）不该拖垮整批评测
            print(f"{ticker} 第{attempt + 1}次运行失败：{exc}", file=sys.stderr)
            runs.append(_error_result(ticker, exc))

    precisions = [r["precision"] for r in runs if r["precision"] is not None]
    recalls = [r["recall"] for r in runs if r["recall"] is not None]
    self_verifications = [r["self_verification_triggered"] for r in runs if r["self_verification_triggered"] is not None]

    return {
        "ticker": ticker,
        "n": repeats,
        "n_succeeded": sum(1 for r in runs if r["error"] is None),
        "runs": runs,
        "mean_precision": statistics.fmean(precisions) if precisions else None,
        "mean_recall": statistics.fmean(recalls) if recalls else None,
        "self_verification_rate": (sum(self_verifications) / len(self_verifications)) if self_verifications else None,
    }


def _append_run_log(results: list[dict]) -> None:
    RUNS_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    record = {"timestamp": datetime.now(UTC).isoformat(), "results": results}
    with RUNS_LOG_PATH.open("a") as log_file:
        log_file.write(json.dumps(record, ensure_ascii=False) + "\n")


def _print_table(results: list[dict]) -> None:
    header = f"{'Ticker':<8}{'次数':<6}{'成功':<6}{'精确率':<8}{'召回率':<8}{'自我核查率'}"
    print(header)
    print("-" * len(header))
    precisions, recalls, self_verification_rates = [], [], []
    for result in results:
        if result["mean_precision"] is None:
            print(f"{result['ticker']:<8}全部{result['n']}次运行失败")
            continue
        precisions.append(result["mean_precision"])
        recalls.append(result["mean_recall"])
        if result["self_verification_rate"] is not None:
            self_verification_rates.append(result["self_verification_rate"])
        print(
            f"{result['ticker']:<8}"
            f"{result['n']:<6}"
            f"{result['n_succeeded']:<6}"
            f"{result['mean_precision']:<8.2f}"
            f"{result['mean_recall']:<8.2f}"
            f"{result['self_verification_rate']:.2f}"
        )
    if precisions:
        print("-" * len(header))
        print(
            f"{'整体均值':<8}{'':<6}{'':<6}"
            f"{statistics.fmean(precisions):<8.2f}"
            f"{statistics.fmean(recalls):<8.2f}"
            f"{statistics.fmean(self_verification_rates) if self_verification_rates else 0.0:.2f}"
        )


async def main(tickers: list[str] | None, temperature: float | None, repeats: int) -> int:
    eval_set = _load_eval_set()
    if tickers:
        wanted = set(tickers)
        eval_set = [entry for entry in eval_set if entry["ticker"] in wanted]

    estimate = _estimate_alpha_vantage_calls(eval_set, repeats)
    print(
        f"预计至少消耗Alpha Vantage {estimate}次调用配额（每天上限25次；"
        "get_price_reaction的兜底路径可能额外消耗，未计入这个估算），"
        "避免在真实使用高峰期跑",
        file=sys.stderr,
    )

    results = [await _run_repeated(entry["ticker"], entry["expected_tools"], temperature, repeats) for entry in eval_set]

    _append_run_log(results)
    print()
    _print_table(results)
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="工具选择准确率探测：对标注评测集跑真实Agent Loop并算精确率/召回率")
    parser.add_argument("--tickers", type=str, default=None, help="逗号分隔的ticker列表，默认用评测集里的全部ticker")
    parser.add_argument(
        "--temperature",
        type=float,
        default=DEFAULT_TEMPERATURE,
        help=f"传给run_agent_loop的temperature，默认{DEFAULT_TEMPERATURE}（固定输出，减少评估噪声）",
    )
    parser.add_argument(
        "--repeats", type=int, default=1, help="每个ticker重复跑几次取均值，默认1（不重复）"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    parsed_tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()] if args.tickers else None
    sys.exit(asyncio.run(main(parsed_tickers, temperature=args.temperature, repeats=args.repeats)))
