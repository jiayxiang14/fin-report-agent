"""简报质量回归探测器：不是CI门禁，是"改Prompt/换模型前后手动跑一遍"用的工具。

复用 reward.py 里 Best-of-N 已经写好的打分逻辑（数字可追溯性/自我核查是否触发/
三段式结构合规/长度合理性 + 结论裁判打分 + 过程裁判打分），对一个固定的真实ticker评测集各跑一次
完整Agent Loop（默认temperature，不是Best-of-N的3候选），把结果追加写进
backend/.cache/eval_runs.jsonl，并跟上一次记录的分数比较，把总分明显下滑的ticker
标出来。

这解决的是"简报质量目前只能靠人读、没有任何自动化回归检测"这个真实缺口——但它
检测的是"有没有变差"（数字对不对得上、结构还在不在），不是"这份分析写得好不好"，
后者规则打分和LLM裁判都只能给参考分，替代不了人工判断。

不接入CI：每次都是对真实ticker跑完整Agent Loop + 真实调用SEC EDGAR/Polygon/LLM，
真花钱、真要等，不适合每次commit都跑。

用法（在 backend/ 目录下）：
    python scripts/eval_report_quality.py
    python scripts/eval_report_quality.py --tickers AAPL,MSFT
    python scripts/eval_report_quality.py --no-llm-judge   # 只用免费的规则打分，更快
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.services.agent import reward  # noqa: E402
from app.services.agent.loop import run_agent_loop  # noqa: E402
from app.services.polygon_client import CACHE_DIR  # noqa: E402

# 覆盖科技(AAPL/MSFT/AMZN/NVDA)和非科技(KO)公司，财务数据稳定、容易人工核对——
# 沿用本项目此前手动验证 Best-of-N（AMZN）/ 主题匹配（NVDA/KO）时用过的同一批
# ticker，不是随手新选的
DEFAULT_TICKERS = ["AAPL", "MSFT", "AMZN", "KO", "NVDA"]

REGRESSION_THRESHOLD = 10.0  # 总分比上一次跑同一个ticker低这么多分，标出来提醒人工看

RUNS_LOG_PATH: Path = CACHE_DIR / "eval_runs.jsonl"


async def _run_one(ticker: str, use_llm_judge: bool) -> dict:
    raw_outputs: list[str] = []

    async def capture(_tool_name: str, content: str) -> None:
        raw_outputs.append(content)

    run_result = await run_agent_loop(ticker, on_tool_result=capture)
    rule_score = reward.score_rule_based(run_result, raw_outputs)
    llm_score, llm_reason = (
        await reward.score_llm_judge(run_result.final_report) if use_llm_judge else (None, None)
    )
    trajectory_score, trajectory_reason = (
        await reward.score_trajectory_judge(run_result.reasoning_notes, run_result.transcript)
        if use_llm_judge
        else (None, None)
    )
    total_score = reward.combine_scores(rule_score, llm_score, trajectory_score)

    return {
        "ticker": ticker,
        "completed": run_result.completed,
        "stop_reason": run_result.stop_reason,
        "rule_score": rule_score.model_dump(),
        "llm_score": llm_score,
        "llm_reason": llm_reason,
        "trajectory_score": trajectory_score,
        "trajectory_reason": trajectory_reason,
        "total_score": total_score,
        "error": None,
    }


def _load_previous_run() -> dict[str, dict] | None:
    """只看上一次记录（最后一行），不是全部历史——评测集会跟着改，跟太早以前的
    记录比没有意义，能看到"最近一次相比这一次"的变化就够用了。"""
    if not RUNS_LOG_PATH.exists():
        return None
    last_line = None
    with RUNS_LOG_PATH.open() as log_file:
        for line in log_file:
            stripped = line.strip()
            if stripped:
                last_line = stripped
    if last_line is None:
        return None
    record = json.loads(last_line)
    return {result["ticker"]: result for result in record["results"] if result.get("total_score") is not None}


def _append_run_log(results: list[dict]) -> None:
    RUNS_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    record = {"timestamp": datetime.now(UTC).isoformat(), "results": results}
    with RUNS_LOG_PATH.open("a") as log_file:
        log_file.write(json.dumps(record, ensure_ascii=False) + "\n")


def _print_table(results: list[dict], previous: dict[str, dict] | None) -> bool:
    has_regression = False
    header = (
        f"{'Ticker':<8}{'完成':<6}{'可追溯':<8}{'自查':<6}{'结构':<6}{'长度':<6}"
        f"{'结论裁判':<8}{'过程裁判':<8}{'总分':<8}{'变化'}"
    )
    print(header)
    print("-" * len(header))
    for result in results:
        if result.get("error"):
            print(f"{result['ticker']:<8}运行失败：{result['error']}")
            continue
        rule = result["rule_score"]
        llm_score = result["llm_score"]
        llm_display = f"{llm_score:.1f}" if llm_score is not None else "-"
        trajectory_score = result.get("trajectory_score")
        trajectory_display = f"{trajectory_score:.1f}" if trajectory_score is not None else "-"
        delta_str = ""
        if previous and result["ticker"] in previous:
            delta = result["total_score"] - previous[result["ticker"]]["total_score"]
            delta_str = f"{delta:+.1f}"
            if delta <= -REGRESSION_THRESHOLD:
                delta_str += " ⚠ 回归"
                has_regression = True
        print(
            f"{result['ticker']:<8}"
            f"{'是' if result['completed'] else '否':<6}"
            f"{rule['traceability']:<8.1f}"
            f"{rule['self_verification']:<6.1f}"
            f"{rule['structure']:<6.1f}"
            f"{rule['length']:<6.1f}"
            f"{llm_display:<8}"
            f"{trajectory_display:<8}"
            f"{result['total_score']:<8.1f}"
            f"{delta_str}"
        )
    return has_regression


async def main(tickers: list[str], use_llm_judge: bool) -> int:
    previous = _load_previous_run()
    results = []
    for ticker in tickers:
        print(f"正在跑 {ticker} ...", file=sys.stderr)
        try:
            results.append(await _run_one(ticker, use_llm_judge))
        except Exception as exc:  # noqa: BLE001 - 单个ticker失败（上游限速/网络/账户余额）不该拖垮整批评测
            print(f"{ticker} 运行失败：{exc}", file=sys.stderr)
            results.append(
                {
                    "ticker": ticker,
                    "completed": False,
                    "stop_reason": "error",
                    "rule_score": None,
                    "llm_score": None,
                    "llm_reason": None,
                    "trajectory_score": None,
                    "trajectory_reason": None,
                    "total_score": None,
                    "error": str(exc),
                }
            )

    _append_run_log(results)
    print()
    has_regression = _print_table(results, previous)
    if has_regression:
        print("\n发现总分明显下滑的ticker（标⚠），建议人工核对上面对应的报告内容。", file=sys.stderr)
    return 1 if has_regression else 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="简报质量回归探测：对固定ticker评测集跑真实Agent Loop并打分")
    parser.add_argument("--tickers", type=str, default=",".join(DEFAULT_TICKERS), help="逗号分隔的ticker列表")
    parser.add_argument(
        "--no-llm-judge", action="store_true", help="跳过LLM裁判调用，只用免费的规则打分（更快更省）"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    parsed_tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
    sys.exit(asyncio.run(main(parsed_tickers, use_llm_judge=not args.no_llm_judge)))
