"""LLM裁判ground truth校准——第一步：从历史Best-of-N运行记录里抽取候选简报，
生成一份待人工标注的文件。不产生任何标注结果，标注必须由人完成（用LLM去标
等于用LLM验证LLM，起不到校准作用），这个脚本只负责"选出哪些候选值得标"。

用法（在 backend/ 目录下）：
    python scripts/build_report_quality_eval_set.py

默认从 .cache/best_of_n_runs.jsonl（Best-of-N每次运行都会追加写入的历史记录）
里按ticker分层抽样，每个ticker最多取PER_TICKER_TARGET条，覆盖该ticker历史
候选里的低分/中分/高分区间，优先选来自不同run（不同temperature、不同时间）
的候选，减少抽到近似重复内容的概率。

输出到 tests/fixtures/report_quality_eval_set.json，不包含任何历史LLM分数——
标注时不能看到模型自己给过的分数，否则人工判断会被无意识锚定，校准出来的
相关系数会虚高（这是复盘时发现的设计漏洞，特意留在这条注释里说明原因）。

标注完成后，从全部条目里挑3条分别代表低/中/高分，把role改成"anchor"，其余
条目role改成"eval"；再运行 eval_judge_calibration.py 用held-out的"eval"条目
计算裁判打分和人工打分的相关系数。
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.services.agent.reward import MIN_REPORT_LENGTH  # noqa: E402
from app.services.polygon_client import CACHE_DIR  # noqa: E402

RUNS_LOG_PATH: Path = CACHE_DIR / "best_of_n_runs.jsonl"
OUTPUT_PATH = Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "report_quality_eval_set.json"

PER_TICKER_TARGET = 3

# 排除掉明显不是真实报告的候选——.cache/best_of_n_runs.jsonl里混着单测跑测试
# 时写入的假数据（比如"<conclusion>c</conclusion>..."这种占位符，实测发现50字符
# 的阈值挡不住它）。直接复用reward.py自己认定"过短"的分界线：一份报告如果连
# 规则打分都认为太短，就没有资格代表"真实候选"去参与标注
MIN_REPORT_LENGTH_FOR_LABELING = MIN_REPORT_LENGTH

LABELING_INSTRUCTIONS = (
    "对每一条final_report，按跟裁判prompt完全相同的三个维度打1到10的整数分，"
    "填入human_score，并在human_reason里写一句话理由：\n"
    "1. 逻辑连贯性——前后是否自洽\n"
    "2. 有没有自相矛盾——结论和evidence/flags区块是否互相打架\n"
    "3. 洞察是否有价值——是不是只在复述数字，有没有真正有信息量的判断\n"
    "\n"
    "标注时不要参考任何模型给过的历史分数（本文件里也没有附带），打分应完全"
    "基于你自己独立阅读的判断。\n"
    "\n"
    "全部标完后，从中挑3条分别代表低/中/高分，把它们的role改成\"anchor\"，"
    "其余条目的role改成\"eval\"——不要有条目的role留空。"
)


def _load_candidates(runs_path: Path) -> list[dict[str, Any]]:
    """展开jsonl里每条run的candidates数组，只保留跑完且有正文的候选——失败的
    候选没有final_report可标，标了也没有意义。"""
    if not runs_path.exists():
        return []

    candidates: list[dict[str, Any]] = []
    with runs_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            run = json.loads(line)
            for candidate in run.get("candidates", []):
                report = candidate.get("final_report")
                if not candidate.get("completed") or not report:
                    continue
                if len(report) < MIN_REPORT_LENGTH_FOR_LABELING:
                    continue
                candidates.append(
                    {
                        "ticker": run["ticker"],
                        "timestamp": run["timestamp"],
                        "total_score": candidate["total_score"],
                        "final_report": candidate["final_report"],
                    }
                )
    return candidates


def _nearest_unused_timestamp_index(ranked: list[dict[str, Any]], anchor_idx: int, used_timestamps: set[str]) -> int:
    n = len(ranked)
    for offset in range(n):
        for idx in (anchor_idx - offset, anchor_idx + offset):
            if 0 <= idx < n and ranked[idx]["timestamp"] not in used_timestamps:
                return idx
    return anchor_idx  # 所有候选的timestamp都已经用过（极端情况），退化成直接取锚点位置


def _stratified_sample(pool: list[dict[str, Any]], target: int = PER_TICKER_TARGET) -> list[dict[str, Any]]:
    """按分数把候选分层抽样：候选数不超过target时全部保留；否则在分数从低到高
    排序后的序列上均匀取target个位置（覆盖低/中/高区间），每个位置优先选一个
    timestamp还没被选过的候选，避免抽到同一次run里的多个候选（内容相似度高，
    会稀释标注样本的真实多样性）。"""
    if len(pool) <= target:
        return sorted(pool, key=lambda c: c["total_score"])

    ranked = sorted(pool, key=lambda c: c["total_score"])
    n = len(ranked)
    picks: list[dict[str, Any]] = []
    used_timestamps: set[str] = set()
    for i in range(target):
        frac = i / (target - 1) if target > 1 else 0.0
        anchor_idx = round(frac * (n - 1))
        idx = _nearest_unused_timestamp_index(ranked, anchor_idx, used_timestamps)
        picks.append(ranked[idx])
        used_timestamps.add(ranked[idx]["timestamp"])
    return picks


def build_eval_set(runs_path: Path) -> tuple[dict[str, Any], list[str]]:
    """返回(待写入的文件内容, 多样性警告列表)。警告不阻止生成文件，只是如实
    提示哪些ticker的样本可能因为历史run太少而不够多样。"""
    candidates = _load_candidates(runs_path)
    by_ticker: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for c in candidates:
        by_ticker[c["ticker"]].append(c)

    items: list[dict[str, Any]] = []
    warnings: list[str] = []
    for ticker in sorted(by_ticker):
        pool = by_ticker[ticker]
        distinct_timestamps = len({c["timestamp"] for c in pool})
        if distinct_timestamps < 2:
            warnings.append(f"{ticker}: 只有{distinct_timestamps}个不同的run，样本多样性有限（候选内容可能相似）")

        for picked in _stratified_sample(pool):
            items.append(
                {
                    "ticker": ticker,
                    "timestamp": picked["timestamp"],
                    "final_report": picked["final_report"],
                    "human_score": None,
                    "human_reason": None,
                    "role": None,
                }
            )

    return {"labeling_instructions": LABELING_INSTRUCTIONS, "items": items}, warnings


def main() -> None:
    if OUTPUT_PATH.exists():
        print(f"{OUTPUT_PATH} 已存在，不会覆盖（可能已经标注了一部分）。如果确实要重新生成，先手动删除或改名。")
        return

    content, warnings = build_eval_set(RUNS_LOG_PATH)
    if not content["items"]:
        print(f"{RUNS_LOG_PATH} 里没有找到任何跑完的候选，无法生成标注文件。")
        return

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w") as f:
        json.dump(content, f, ensure_ascii=False, indent=2)

    print(f"已生成 {len(content['items'])} 条待标注样本 -> {OUTPUT_PATH}")
    for warning in warnings:
        print(f"  警告: {warning}")
    print("\n接下来：打开这份文件，按labeling_instructions手动标注每一条的human_score/human_reason，")
    print("再挑3条设为role=\"anchor\"，其余设为role=\"eval\"。")


if __name__ == "__main__":
    main()
