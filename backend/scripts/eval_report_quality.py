"""简报质量回归探测器：不是CI门禁，是"改Prompt/换模型前后手动跑一遍"用的工具。

复用 reward.py 里 Best-of-N 已经写好的打分逻辑（数字可追溯性/自我核查是否触发/
三段式结构合规/长度合理性 + 结论裁判打分 + 过程裁判打分），对一个固定的真实ticker评测集
各跑`--repeats`次完整Agent Loop，把结果追加写进 backend/.cache/eval_runs.jsonl，
并跟上一次记录的均值比较，把均总分明显下滑的ticker标出来。

这解决的是"简报质量目前只能靠人读、没有任何自动化回归检测"这个真实缺口——但它
检测的是"有没有变差"（数字对不对得上、结构还在不在），不是"这份分析写得好不好"，
后者规则打分和LLM裁判都只能给参考分，替代不了人工判断。

默认固定 temperature=0（而不是普通分析路径的API默认随机值）+ 支持多次重复取均值——
这是2026-08-15跟一次真实的prompt A/B对比复盘时发现的问题：不固定temperature、
只跑一次的话，根本分不清"分数变化是prompt改动的效果"还是"这次刚好抽到较好/较差
的一次采样"。默认repeats=1保持向后兼容（不强制拉高每次跑的花费），真要做严谨对比
时应该显式传 --repeats 3 起步。

不接入CI：每次都是对真实ticker跑完整Agent Loop + 真实调用SEC EDGAR/Polygon/LLM，
真花钱、真要等，不适合每次commit都跑。

用法（在 backend/ 目录下）：
    python scripts/eval_report_quality.py
    python scripts/eval_report_quality.py --tickers AAPL,MSFT
    python scripts/eval_report_quality.py --repeats 3          # 每个ticker跑3次取均值/标准差
    python scripts/eval_report_quality.py --temperature 0.7    # 不固定为0，想看采样多样性时用
    python scripts/eval_report_quality.py --no-llm-judge       # 只用免费的规则打分，更快
    python scripts/eval_report_quality.py --no-compaction      # 关闭上下文压缩，跟默认(压缩开启)对照

`--no-compaction` 是2026-08-16给上下文压缩机制（loop.py 的
`_compact_seen_tool_results`）补的可重复验证手段——之前每次要验证压缩改动
有没有影响报告质量，得手动改代码把压缩逻辑临时改成 no-op、跑完再改回来，
容易操作失误也不好复用。现在把开关做成 `run_agent_loop` 的参数，跑两组
`--repeats 3` 对照（一组默认、一组带 `--no-compaction`）就能随时重新验证。
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

from app.services.agent import reward  # noqa: E402
from app.services.agent.loop import run_agent_loop  # noqa: E402
from app.services.polygon_client import CACHE_DIR  # noqa: E402

# 覆盖科技(AAPL/MSFT/AMZN/NVDA)和非科技(KO)公司，财务数据稳定、容易人工核对——
# 沿用本项目此前手动验证 Best-of-N（AMZN）/ 主题匹配（NVDA/KO）时用过的同一批
# ticker，不是随手新选的
DEFAULT_TICKERS = ["AAPL", "MSFT", "AMZN", "KO", "NVDA"]

# 普通分析路径不像Best-of-N那样特意制造采样多样性，这里固定成0（贪婪解码，
# 波动最小）而不是沿用API默认的随机值——评估要的是"能不能稳定复现"，不是
# "看看模型能写出多少种版本"，多样性探测是Best-of-N自己的事
DEFAULT_TEMPERATURE = 0.0

REGRESSION_THRESHOLD = 10.0  # 均总分比上一次记录的均值低这么多分，标出来提醒人工看

RUNS_LOG_PATH: Path = CACHE_DIR / "eval_runs.jsonl"


async def _run_one(ticker: str, use_llm_judge: bool, temperature: float | None, enable_compaction: bool) -> dict:
    raw_outputs: list[str] = []

    async def capture(_tool_name: str, content: str) -> None:
        raw_outputs.append(content)

    run_result = await run_agent_loop(
        ticker, on_tool_result=capture, temperature=temperature, enable_compaction=enable_compaction
    )
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


def _error_result(ticker: str, exc: Exception) -> dict:
    return {
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


async def _run_repeated(
    ticker: str, use_llm_judge: bool, temperature: float | None, repeats: int, enable_compaction: bool
) -> dict:
    """跑同一个ticker`repeats`次，汇总成均值/标准差——单次结果分不清"prompt效果"
    和"这次刚好抽到的随机波动"，重复采样是唯一能把两者分开的办法。"""
    runs: list[dict] = []
    for attempt in range(repeats):
        print(f"正在跑 {ticker} ...（{attempt + 1}/{repeats}）", file=sys.stderr)
        try:
            runs.append(await _run_one(ticker, use_llm_judge, temperature, enable_compaction))
        except Exception as exc:  # noqa: BLE001 - 单次失败（上游限速/网络/账户余额）不该拖垮整批评测
            print(f"{ticker} 第{attempt + 1}次运行失败：{exc}", file=sys.stderr)
            runs.append(_error_result(ticker, exc))

    scores = [run["total_score"] for run in runs if run["total_score"] is not None]
    mean_total_score = statistics.fmean(scores) if scores else None
    stdev_total_score = statistics.stdev(scores) if len(scores) >= 2 else 0.0 if scores else None

    return {
        "ticker": ticker,
        "n": repeats,
        "n_succeeded": len(scores),
        "runs": runs,
        "mean_total_score": mean_total_score,
        "stdev_total_score": stdev_total_score,
    }


def _load_previous_run() -> dict[str, dict] | None:
    """只看上一次记录（最后一行），不是全部历史——评测集会跟着改，跟太早以前的
    记录比没有意义，能看到"最近一次相比这一次"的变化就够用了。旧格式的记录
    （改重复采样之前跑的，`results`里直接是单次运行结果，没有`mean_total_score`)
    用`total_score`兜底，不然读旧日志会直接炸。"""
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
    summary: dict[str, dict] = {}
    for result in record["results"]:
        mean_score = result.get("mean_total_score", result.get("total_score"))
        if mean_score is not None:
            summary[result["ticker"]] = {**result, "mean_total_score": mean_score}
    return summary


def _append_run_log(results: list[dict]) -> None:
    RUNS_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    record = {"timestamp": datetime.now(UTC).isoformat(), "results": results}
    with RUNS_LOG_PATH.open("a") as log_file:
        log_file.write(json.dumps(record, ensure_ascii=False) + "\n")


def _print_table(results: list[dict], previous: dict[str, dict] | None) -> bool:
    has_regression = False
    header = f"{'Ticker':<8}{'次数':<6}{'成功':<6}{'均总分':<10}{'标准差':<8}{'变化'}"
    print(header)
    print("-" * len(header))
    for result in results:
        if result["mean_total_score"] is None:
            print(f"{result['ticker']:<8}全部{result['n']}次运行失败")
            continue
        delta_str = ""
        if previous and result["ticker"] in previous:
            delta = result["mean_total_score"] - previous[result["ticker"]]["mean_total_score"]
            delta_str = f"{delta:+.1f}"
            if delta <= -REGRESSION_THRESHOLD:
                delta_str += " ⚠ 回归"
                has_regression = True
        print(
            f"{result['ticker']:<8}"
            f"{result['n']:<6}"
            f"{result['n_succeeded']:<6}"
            f"{result['mean_total_score']:<10.1f}"
            f"{result['stdev_total_score']:<8.1f}"
            f"{delta_str}"
        )
    return has_regression


async def main(
    tickers: list[str],
    use_llm_judge: bool,
    temperature: float | None,
    repeats: int,
    enable_compaction: bool = True,
) -> int:
    previous = _load_previous_run()
    results = [
        await _run_repeated(ticker, use_llm_judge, temperature, repeats, enable_compaction) for ticker in tickers
    ]

    _append_run_log(results)
    print()
    has_regression = _print_table(results, previous)
    if has_regression:
        print("\n发现均总分明显下滑的ticker（标⚠），建议人工核对上面对应的报告内容。", file=sys.stderr)
    return 1 if has_regression else 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="简报质量回归探测：对固定ticker评测集跑真实Agent Loop并打分")
    parser.add_argument("--tickers", type=str, default=",".join(DEFAULT_TICKERS), help="逗号分隔的ticker列表")
    parser.add_argument(
        "--no-llm-judge", action="store_true", help="跳过LLM裁判调用，只用免费的规则打分（更快更省）"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=DEFAULT_TEMPERATURE,
        help=f"传给run_agent_loop的temperature，默认{DEFAULT_TEMPERATURE}（固定输出，减少评估噪声）",
    )
    parser.add_argument(
        "--repeats", type=int, default=1, help="每个ticker重复跑几次取均值/标准差，默认1（不重复）"
    )
    parser.add_argument(
        "--no-compaction",
        action="store_true",
        help="关闭上下文压缩（run_agent_loop的enable_compaction=False），跟默认(压缩开启)跑一遍做真实对照",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    parsed_tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
    sys.exit(
        asyncio.run(
            main(
                parsed_tickers,
                use_llm_judge=not args.no_llm_judge,
                temperature=args.temperature,
                repeats=args.repeats,
                enable_compaction=not args.no_compaction,
            )
        )
    )
