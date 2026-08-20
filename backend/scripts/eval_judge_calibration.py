"""LLM裁判ground truth校准——第二步：拿人工标注好的held-out样本（role="eval"，
不含被塞进prompt当few-shot锚点的那几条），真实调用score_llm_judge，把裁判打分
跟人工打分算相关系数——回答"这个裁判打的分到底跟人的判断对不对得上"这个问题。

前置条件：tests/fixtures/report_quality_eval_set.json 里的所有条目都已经标注
完成（human_score非空、role是"anchor"或"eval"），且被选中当锚点的3条已经把
role改成了"anchor"、内容摘录手动誊写进了 reward._JUDGE_SCORE_ANCHORS。

同run泄漏提示：Best-of-N同一次运行会产出多个（通常3个，仅temperature不同）
候选——如果某个锚点和某条held-out样本来自同一个timestamp（同一次运行），
裁判在prompt里已经见过同一批底层财务数据的"姊妹候选"当参照，这条held-out
样本的相关性可能被人为拉高，不是真实泛化能力。脚本会自动标出这类样本，
分别报告"全部held-out"和"排除同run样本"两个相关系数，不要只看第一个。

用法（在 backend/ 目录下）：
    python scripts/eval_judge_calibration.py              # 跑全部held-out样本
    python scripts/eval_judge_calibration.py --limit 2     # 先跑前2条，确认链路没问题（API key配置对/裁判能正常解析）再跑全量
    python scripts/eval_judge_calibration.py --offset 2    # 跳过前2条（比如已经用--limit试跑过），只跑剩下的——
                                                            # 注意这样算出的相关系数只覆盖跳过之后的样本，不是全量，
                                                            # 要得到完整结论需要把试跑那几条的结果手动并进来

会真实调用LLM（每条held-out样本触发JUDGE_SAMPLE_COUNT次自我一致性采样），
真花钱，不接入CI。样本量通常只有十几条，算出来的相关系数只能给方向性判断
（"明显跑偏"还是"大致靠谱"），不是严谨的统计结论——这一点在下面输出里也会
如实提示，不只是写在注释里自己看。`--limit`跑出来的样本量更小，相关系数
本身没有参考价值，只用来确认脚本能跑通、裁判返回的格式能被正常解析。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.services.agent.reward import JUDGE_SAMPLE_COUNT, score_llm_judge  # noqa: E402
from app.services.polygon_client import CACHE_DIR  # noqa: E402

EVAL_SET_PATH = Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "report_quality_eval_set.json"
RUNS_LOG_PATH: Path = CACHE_DIR / "eval_judge_calibration_runs.jsonl"

MIN_SAMPLE_SIZE_FOR_CAUTION = 20


def _load_held_out_items(path: Path) -> tuple[list[dict[str, Any]], set[str]]:
    """返回 (held-out条目, 锚点timestamp集合)。held-out只取role=="eval"且已
    标注完成的条目——锚点(role=="anchor")本身被排除是因为它们的摘录已经进了
    裁判prompt，用它们验证会有直接的数据泄漏。锚点的timestamp额外返回，供
    调用方标出"跟某个锚点来自同一次Best-of-N运行"的held-out条目（同一次运行
    产出的多个候选只是temperature不同，底层财务数据完全相同，裁判可能是靠
    认出同一批数据在打分，不是真实评估这条held-out样本本身）。"""
    if not path.exists():
        raise FileNotFoundError(f"{path} 不存在，先运行 scripts/build_report_quality_eval_set.py 生成待标注文件")

    with path.open() as f:
        content = json.load(f)

    items = content.get("items", [])
    unlabeled = [i for i in items if i.get("role") is None or i.get("human_score") is None]
    if unlabeled:
        raise ValueError(
            f"还有 {len(unlabeled)} 条未标注完成（role或human_score为空），请先在"
            f"{path} 里标注完所有条目再运行这个脚本"
        )

    anchor_timestamps = {i["timestamp"] for i in items if i["role"] == "anchor"}
    held_out = [i for i in items if i["role"] == "eval"]
    return held_out, anchor_timestamps


def _rank_with_average_ties(values: list[float]) -> list[float]:
    """把values转成秩次(rank)，并列值取平均秩——Spearman相关系数的标准写法，
    小样本+人工整数打分通常会有大量并列值，用简化的d²公式（假设无并列）在
    有并列时是有偏的，必须用"平均秩+按秩算Pearson"这个等价写法。"""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        average_rank = (i + j) / 2 + 1  # 1-indexed
        for k in range(i, j + 1):
            ranks[order[k]] = average_rank
        i = j + 1
    return ranks


def _pearson(xs: list[float], ys: list[float]) -> float | None:
    n = len(xs)
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    covariance = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys, strict=True))
    variance_x = sum((x - mean_x) ** 2 for x in xs)
    variance_y = sum((y - mean_y) ** 2 for y in ys)
    if variance_x == 0 or variance_y == 0:
        return None
    return covariance / (variance_x * variance_y) ** 0.5


def spearman_correlation(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    return _pearson(_rank_with_average_ties(xs), _rank_with_average_ties(ys))


def mean_absolute_error(xs: list[float], ys: list[float]) -> float:
    return sum(abs(x - y) for x, y in zip(xs, ys, strict=True)) / len(xs)


async def _run_one(item: dict[str, Any]) -> tuple[float | None, str | None]:
    judge_score, reason = await score_llm_judge(item["final_report"])
    score = judge_score / 10 if judge_score is not None else None  # 0-100换回1-10，跟human_score同刻度
    return score, reason


def _print_table(rows: list[tuple[str, float, float | None, bool]]) -> None:
    print(f"{'ticker':<10}{'human':>8}{'judge':>8}{'diff':>8}  同run锚点")
    for ticker, human_score, judge_score, same_run in rows:
        flag = "  ← 是" if same_run else ""
        if judge_score is None:
            print(f"{ticker:<10}{human_score:>8.1f}{'解析失败':>8}{flag}")
            continue
        print(f"{ticker:<10}{human_score:>8.1f}{judge_score:>8.1f}{abs(human_score - judge_score):>8.1f}{flag}")


def _report_correlation(label: str, pairs: list[tuple[float, float]]) -> None:
    if len(pairs) < 2:
        print(f"\n[{label}] 样本不足2条，无法计算相关系数")
        return
    human_values = [h for h, _ in pairs]
    judge_values = [j for _, j in pairs]
    correlation = spearman_correlation(human_values, judge_values)
    mae = mean_absolute_error(human_values, judge_values)
    print(f"\n[{label}] n={len(pairs)}")
    print(f"  Spearman相关系数: {correlation:.2f}" if correlation is not None else "  Spearman相关系数: 无法计算（某一侧全部同值）")
    print(f"  平均绝对误差: {mae:.2f}分（1-10刻度）")
    if len(pairs) < MIN_SAMPLE_SIZE_FOR_CAUTION:
        print(f"  注意：样本量只有{len(pairs)}条，置信区间很宽，这个相关系数只能给方向性判断，不是严谨的统计结论。")


def _correlation_stats(pairs: list[tuple[float, float]]) -> dict[str, Any]:
    """跟`_report_correlation`算的是同一件事，返回结构化结果供`_append_run_log`
    持久化——之前只打印到stdout，跑完这次终端一关就什么都没留下，跟另外两个
    eval脚本（`eval_report_quality.py`/`eval_tool_selection.py`）都有的run log
    习惯不一致，补上。"""
    if len(pairs) < 2:
        return {"n": len(pairs), "spearman": None, "mae": None}
    human_values = [h for h, _ in pairs]
    judge_values = [j for _, j in pairs]
    return {
        "n": len(pairs),
        "spearman": spearman_correlation(human_values, judge_values),
        "mae": mean_absolute_error(human_values, judge_values),
    }


def _append_run_log(record: dict[str, Any]) -> None:
    RUNS_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with RUNS_LOG_PATH.open("a") as log_file:
        log_file.write(json.dumps(record, ensure_ascii=False) + "\n")


async def main(limit: int | None = None, offset: int = 0) -> None:
    items, anchor_timestamps = _load_held_out_items(EVAL_SET_PATH)
    if offset:
        items = items[offset:]
        print(f"--offset {offset}：跳过前{offset}条，只跑剩下的{len(items)}条——这次算出的相关系数只覆盖这部分样本，不是全量。\n")
    if limit is not None:
        items = items[:limit]
        print(f"--limit {limit}：只跑前{len(items)}条做试跑，不是完整校准；确认链路没问题后，去掉--limit再跑全量。\n")
    if len(items) < 2:
        print(f"held-out样本只有{len(items)}条，不足以计算相关系数（至少需要2条）")
        return

    results = await asyncio.gather(*(_run_one(item) for item in items))

    rows = [
        (item["ticker"], item["human_score"], score, item["timestamp"] in anchor_timestamps)
        for item, (score, _reason) in zip(items, results, strict=True)
    ]
    _print_table(rows)

    valid_pairs = [(h, j) for _, h, j, _ in rows if j is not None]
    valid_pairs_excluding_same_run = [(h, j) for _, h, j, same_run in rows if j is not None and not same_run]

    _report_correlation("全部held-out样本", valid_pairs)
    _report_correlation("排除与锚点同run的样本", valid_pairs_excluding_same_run)
    if len(valid_pairs) != len(valid_pairs_excluding_same_run):
        print(
            "\n上面两个数字如果差异明显，说明「全部held-out」那个相关系数被同run泄漏"
            "拉高了，应该以「排除同run样本」的数字为准。"
        )

    print(f"\n裁判打分理由（自我一致性{JUDGE_SAMPLE_COUNT}次采样里第一个成功样本的理由，不是{JUDGE_SAMPLE_COUNT}次的综合）：")
    for item, (score, reason) in zip(items, results, strict=True):
        score_display = f"{score:.1f}" if score is not None else "解析失败"
        print(f"- {item['ticker']}（human={item['human_score']}, judge={score_display}）：{reason or '(无)'}")

    _append_run_log(
        {
            "timestamp": datetime.now(UTC).isoformat(),
            "judge_sample_count": JUDGE_SAMPLE_COUNT,
            "limit": limit,
            "offset": offset,
            "full": _correlation_stats(valid_pairs),
            "excluding_same_run": _correlation_stats(valid_pairs_excluding_same_run),
            "rows": [
                {
                    "ticker": ticker,
                    "human": human_score,
                    "judge": judge_score,
                    "same_run_as_anchor": same_run,
                }
                for ticker, human_score, judge_score, same_run in rows
            ],
        }
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LLM裁判ground truth校准：held-out样本与人工打分的相关系数")
    parser.add_argument(
        "--limit", type=int, default=None, help="只跑前N条held-out样本，先小批试跑确认链路没问题，默认跑全部17条"
    )
    parser.add_argument("--offset", type=int, default=0, help="跳过前N条（比如已经用--limit试跑过），只跑剩下的")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    asyncio.run(main(limit=args.limit, offset=args.offset))
