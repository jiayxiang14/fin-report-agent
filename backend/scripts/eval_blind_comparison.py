"""深度分析（Best-of-N）到底比普通分析好吗？——第二步：揭晓盲评结果。

读 .cache/blind_comparison/ 下的 labels.json（人工评判结果）和 mapping.json
（每一对A/B分别是哪种分析），算深度分析的胜率，并用精确二项检验判断"胜率
高于50%"是不是可能只是运气。"差不多"不计入胜负（符号检验的标准做法），
单独报告数量。

判断标准（实验开始前就定好，不看结果再改）：
- 深度分析胜率明显高于50%（p<0.1）→ 值得它3-4倍的成本，继续做A2断点续跑
- 跟普通分析差不多 → 简化成只按数字可追溯性筛选、去掉裁判，降成本
- 明显更差 → 砍掉深度分析

用法（在 backend/ 目录下）：
    python scripts/eval_blind_comparison.py
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.services.polygon_client import CACHE_DIR  # noqa: E402

OUT_DIR: Path = CACHE_DIR / "blind_comparison"


def binomial_two_sided_p(wins: int, n: int) -> float:
    """精确二项检验（p=0.5）的双侧p值：所有"至少跟观测结果一样极端"的结果的概率之和。"""
    if n == 0:
        return 1.0
    probs = [math.comb(n, k) / 2**n for k in range(n + 1)]
    observed = probs[wins]
    return min(1.0, sum(p for p in probs if p <= observed + 1e-12))


def tally(labels: list[dict], mapping: dict[str, dict[str, str]]) -> dict:
    deep_wins = normal_wins = ties = unlabeled = 0
    per_pair = []
    for label in labels:
        pick = label.get("preference")
        if pick not in ("A", "B", "tie"):
            unlabeled += 1
            continue
        if pick == "tie":
            ties += 1
            winner = "tie"
        else:
            winner = mapping[label["id"]][pick]
            if winner == "deep":
                deep_wins += 1
            else:
                normal_wins += 1
        per_pair.append({"id": label["id"], "winner": winner, "reason": label.get("reason", "")})
    decided = deep_wins + normal_wins
    return {
        "deep_wins": deep_wins,
        "normal_wins": normal_wins,
        "ties": ties,
        "unlabeled": unlabeled,
        "deep_win_rate": deep_wins / decided if decided else None,
        "p_value": binomial_two_sided_p(deep_wins, decided),
        "per_pair": per_pair,
    }


def main() -> int:
    labels_path, mapping_path = OUT_DIR / "labels.json", OUT_DIR / "mapping.json"
    if not labels_path.exists() or not mapping_path.exists():
        print(f"需要 {labels_path} 和 {mapping_path}，先运行 build_blind_comparison.py 并完成人工评判", file=sys.stderr)
        return 1
    result = tally(json.loads(labels_path.read_text()), json.loads(mapping_path.read_text()))

    for row in result["per_pair"]:
        label = {"deep": "深度分析更好", "normal": "普通分析更好", "tie": "差不多"}[row["winner"]]
        print(f"{row['id']}  {label}  {row['reason']}")
    decided = result["deep_wins"] + result["normal_wins"]
    print(
        f"\n深度分析胜 {result['deep_wins']}、普通分析胜 {result['normal_wins']}、差不多 {result['ties']}"
        + (f"、未评 {result['unlabeled']}" if result["unlabeled"] else "")
    )
    if decided:
        print(f"深度分析胜率（不计差不多）：{result['deep_win_rate']:.0%}，二项检验双侧 p={result['p_value']:.3f}")
    print("样本量很小，p值只用来防止把运气当成结论；n<10时几乎不可能达到显著，结论以方向为主。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
