"""LLM裁判打分噪声：同一份简报重复打分，打分到底有多稳定？Best-of-N选出来的
"最优候选"是真的更好，还是被裁判的随机波动决定的？

背景：清理测试污染后，best_of_n_runs.jsonl 里只剩8次真实运行。离线分析发现
第一名比第二名平均只高3.6分（满分100），而裁判按1-10分打分，差1分换算到
总分就是2.5分（结论裁判权重25%）——差距跟裁判波动可能是同一个量级，但
日志里只存了5次采样的平均值，测不出波动有多大。这个脚本补上这个测量。

做法：
1. 从 best_of_n_runs.jsonl 读真实运行（过滤掉简报短于MIN_REPORT_LENGTH的
   测试残留）的每个候选简报。
2. 对每份简报调用结论裁判的单次采样（跟线上同一个prompt、同一个温度）
   `--samples`次，算单次采样的标准差，以及"5次取平均"（线上实际用的
   JUDGE_SAMPLE_COUNT）的标准差（自助法重抽样估计）。
3. 蒙特卡洛：保持规则分不变，把每个候选的结论裁判分换成按测得的波动重新
   抽样的值，重新跑很多遍"选最优"，主要看**可复现性**——最常被选中的那个
   候选占多大比例（100%=每次都选同一个，33%=跟随机三选一没区别）。
   "跟日志里选中的一致"只作参考：日志里的分数是旧版裁判打的（之后加过
   few-shot锚点、采样次数从3改成5），现在的裁判整体有偏移，一致率低会把
   "裁判版本变了"和"裁判有随机波动"两件事混在一起。
   过程裁判（trajectory）的输入（推理过程+工具调用序列）没有记进运行日志，
   测不了它自己的波动，分两种假设报告：只有结论裁判有噪声（下界），以及
   过程裁判的噪声跟结论裁判一样大。

会真实调用LLM（每份简报`--samples`次），不接入CI。

用法（在 backend/ 目录下）：
    python scripts/eval_judge_noise.py                 # 全部真实运行，每份简报采样20次
    python scripts/eval_judge_noise.py --samples 10    # 少采一点，省钱
    python scripts/eval_judge_noise.py --runs 2        # 只测前2次运行，先确认链路
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import statistics
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.services.agent import reward  # noqa: E402
from app.services.agent.best_of_n import RUNS_LOG_PATH as BEST_OF_N_RUNS_PATH  # noqa: E402
from app.services.polygon_client import CACHE_DIR  # noqa: E402

RUNS_LOG_PATH: Path = CACHE_DIR / "eval_judge_noise_runs.jsonl"
DEFAULT_SAMPLES = 20
MONTE_CARLO_TRIALS = 5000
MAX_CONCURRENT_JUDGE_CALLS = 8


def load_real_runs(path: Path) -> list[dict[str, Any]]:
    runs = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        run = json.loads(line)
        candidates = [c for c in run["candidates"] if c.get("total_score") is not None]
        if len(candidates) < 2:
            continue
        if not all(len(c.get("final_report") or "") >= reward.MIN_REPORT_LENGTH for c in candidates):
            continue
        runs.append({**run, "candidates": candidates})
    return runs


def ensemble_std(samples: list[float], ensemble_size: int, rng: random.Random, trials: int = 2000) -> float:
    """自助法估计"ensemble_size次采样取平均"这个量的标准差。"""
    means = [statistics.fmean(rng.choices(samples, k=ensemble_size)) for _ in range(trials)]
    return statistics.pstdev(means)


def combined_score(rule_total: float, outcome: float | None, trajectory: float | None) -> float:
    components = [(rule_total, reward.RULE_WEIGHT)]
    if outcome is not None:
        components.append((outcome, reward.OUTCOME_JUDGE_WEIGHT))
    if trajectory is not None:
        components.append((trajectory, reward.TRAJECTORY_JUDGE_WEIGHT))
    total_weight = sum(w for _, w in components)
    return sum(s * w for s, w in components) / total_weight


def simulate_selection(
    candidates: list[dict[str, Any]],
    samples_by_index: dict[int, list[float]],
    *,
    trajectory_noise_std: float,
    rng: random.Random,
    trials: int = MONTE_CARLO_TRIALS,
) -> dict[int, float]:
    """按测得的裁判波动重新抽样结论裁判分（过程裁判按给定标准差加高斯噪声），
    重复"选最优"trials次，返回每个候选被选中的比例。"""
    wins: Counter[int] = Counter()
    for _ in range(trials):
        best_index, best_score = None, float("-inf")
        for c in candidates:
            outcome = statistics.fmean(rng.choices(samples_by_index[c["index"]], k=reward.JUDGE_SAMPLE_COUNT))
            trajectory = c.get("trajectory_score")
            if trajectory is not None and trajectory_noise_std > 0:
                trajectory = trajectory + rng.gauss(0, trajectory_noise_std)
            score = combined_score(c["rule_score"]["total"], outcome, trajectory)
            if score > best_score:
                best_index, best_score = c["index"], score
        wins[best_index] += 1  # type: ignore[index]
    return {index: wins[index] / trials for index in samples_by_index}


async def _sample_report(report: str, samples: int, semaphore: asyncio.Semaphore) -> list[float]:
    prompt = reward._JUDGE_PROMPT_TEMPLATE.format(report=report, anchors=reward._JUDGE_SCORE_ANCHORS)

    async def one() -> float | None:
        async with semaphore:
            score, _ = await reward._call_judge(prompt, temperature=reward.JUDGE_SAMPLE_TEMPERATURE)
            return score

    results = await asyncio.gather(*(one() for _ in range(samples)))
    return [s for s in results if s is not None]


async def main_async(args: argparse.Namespace) -> int:
    runs = load_real_runs(BEST_OF_N_RUNS_PATH)[: args.runs]
    if not runs:
        print(f"{BEST_OF_N_RUNS_PATH} 里没有可用的真实运行记录", file=sys.stderr)
        return 1
    n_reports = sum(len(r["candidates"]) for r in runs)
    print(f"{len(runs)} 次真实运行、{n_reports} 份候选简报，每份采样 {args.samples} 次，共约 {n_reports * args.samples} 次裁判调用\n")

    rng = random.Random(args.seed)
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_JUDGE_CALLS)
    rows = []
    all_single_stds, all_ensemble_stds = [], []
    for run in runs:
        samples_by_index: dict[int, list[float]] = {}
        for c in run["candidates"]:
            samples = await _sample_report(c["final_report"], args.samples, semaphore)
            if len(samples) < 2:
                print(f"  {run['ticker']} 候选{c['index']}：有效采样不足，跳过这次运行", file=sys.stderr)
                break
            samples_by_index[c["index"]] = samples
        else:
            per_candidate = []
            for c in run["candidates"]:
                s = samples_by_index[c["index"]]
                single = statistics.pstdev(s)
                ens = ensemble_std(s, reward.JUDGE_SAMPLE_COUNT, rng)
                all_single_stds.append(single)
                all_ensemble_stds.append(ens)
                per_candidate.append(
                    {
                        "index": c["index"],
                        "logged_llm_score": c.get("llm_score"),
                        "resampled_mean": round(statistics.fmean(s), 2),
                        "single_std": round(single, 2),
                        "ensemble_std": round(ens, 2),
                        "samples": s,
                    }
                )
            mean_ens = statistics.fmean(p["ensemble_std"] for p in per_candidate)
            totals = sorted((c["total_score"] for c in run["candidates"]), reverse=True)
            outcome_only = simulate_selection(run["candidates"], samples_by_index, trajectory_noise_std=0, rng=rng)
            with_traj = simulate_selection(
                run["candidates"], samples_by_index, trajectory_noise_std=mean_ens, rng=rng
            )
            row = {
                "ticker": run["ticker"],
                "timestamp": run["timestamp"],
                "logged_gap_top1_top2": round(totals[0] - totals[1], 2),
                "win_share_outcome_noise_only": outcome_only,
                "win_share_with_trajectory_noise": with_traj,
                "reproducibility_outcome_noise_only": max(outcome_only.values()),
                "reproducibility_with_trajectory_noise": max(with_traj.values()),
                "agreement_with_logged_selection": with_traj.get(run["selected_index"], 0.0),
                "candidates": per_candidate,
            }
            rows.append(row)
            drift = statistics.fmean(
                p["resampled_mean"] - p["logged_llm_score"] for p in per_candidate if p["logged_llm_score"] is not None
            )
            print(
                f"{row['ticker']:<6} 日志里第一第二差 {row['logged_gap_top1_top2']:>5.1f} | "
                f"裁判单次标准差 {statistics.fmean(p['single_std'] for p in per_candidate):>4.1f}，"
                f"5次平均标准差 {mean_ens:>4.1f} | "
                f"可复现性 {row['reproducibility_outcome_noise_only']:.0%} / {row['reproducibility_with_trajectory_noise']:.0%} | "
                f"跟日志选择一致 {row['agreement_with_logged_selection']:.0%} | 新旧裁判分差 {drift:+.1f}"
            )

    if not rows:
        return 1
    summary = {
        "runs": len(rows),
        "single_std_mean": round(statistics.fmean(all_single_stds), 2),
        "ensemble_std_mean": round(statistics.fmean(all_ensemble_stds), 2),
        "reproducibility_outcome_noise_only_mean": round(
            statistics.fmean(r["reproducibility_outcome_noise_only"] for r in rows), 3
        ),
        "reproducibility_with_trajectory_noise_mean": round(
            statistics.fmean(r["reproducibility_with_trajectory_noise"] for r in rows), 3
        ),
        "agreement_with_logged_selection_mean": round(
            statistics.fmean(r["agreement_with_logged_selection"] for r in rows), 3
        ),
    }
    print(
        f"\n汇总（{summary['runs']} 次运行）：结论裁判单次采样标准差 {summary['single_std_mean']} 分，"
        f"线上实际用的5次平均标准差 {summary['ensemble_std_mean']} 分（满分100）\n"
        f"可复现性（重复选最优时最常选中的候选占比）：{summary['reproducibility_outcome_noise_only_mean']:.0%}（仅结论裁判噪声）/ "
        f"{summary['reproducibility_with_trajectory_noise_mean']:.0%}（假设过程裁判噪声同样大），"
        f"跟日志里的选择一致：{summary['agreement_with_logged_selection_mean']:.0%}\n"
        "100%=每次都选同一个，33%=跟随机三选一没区别。样本只有几次运行，结论只看量级，不是严谨统计。"
    )

    RUNS_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with RUNS_LOG_PATH.open("a") as f:
        f.write(
            json.dumps(
                {"timestamp": datetime.now(UTC).isoformat(), "samples": args.samples, "summary": summary, "rows": rows},
                ensure_ascii=False,
            )
            + "\n"
        )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--samples", type=int, default=DEFAULT_SAMPLES)
    parser.add_argument("--runs", type=int, default=None, help="只测前N次运行")
    parser.add_argument("--seed", type=int, default=0)
    return asyncio.run(main_async(parser.parse_args()))


if __name__ == "__main__":
    sys.exit(main())
