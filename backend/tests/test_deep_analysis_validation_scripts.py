"""深度分析可靠性验证三个脚本的纯函数部分：裁判噪声的选择模拟、盲评配对与
页面生成、盲评结果统计。不真的调用LLM（那是脚本使用者手动执行、真花钱的部分）。"""

import json
import random

from scripts.build_blind_comparison import build_pairs, render_page
from scripts.eval_blind_comparison import binomial_two_sided_p, tally
from scripts.eval_judge_noise import combined_score, load_real_runs, simulate_selection

LONG = "x" * 400


def _candidate(index, rule, llm=80.0, traj=None, report=LONG):
    return {
        "index": index,
        "rule_score": {"total": rule},
        "llm_score": llm,
        "trajectory_score": traj,
        "total_score": rule,
        "final_report": report,
    }


def test_load_real_runs_skips_test_stub_reports_and_single_candidate_runs(tmp_path):
    path = tmp_path / "runs.jsonl"
    rows = [
        {"ticker": "AAPL", "timestamp": "t1", "selected_index": 0, "candidates": [_candidate(0, 90), _candidate(1, 80)]},
        {"ticker": "STUB", "timestamp": "t2", "selected_index": 0, "candidates": [_candidate(0, 50, report="候选0"), _candidate(1, 50, report="候选1")]},
        {"ticker": "ONE", "timestamp": "t3", "selected_index": 0, "candidates": [_candidate(0, 90)]},
    ]
    path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows))

    assert [r["ticker"] for r in load_real_runs(path)] == ["AAPL"]


def test_combined_score_renormalizes_when_a_judge_score_is_missing():
    assert combined_score(80, 60, 40) == 80 * 0.5 + 60 * 0.25 + 40 * 0.25
    assert combined_score(80, 60, None) == (80 * 0.5 + 60 * 0.25) / 0.75


def test_selection_is_fully_reproducible_when_judge_has_no_noise_and_rule_gap_is_large():
    candidates = [_candidate(0, 95), _candidate(1, 60)]
    shares = simulate_selection(candidates, {0: [80.0] * 5, 1: [80.0] * 5}, trajectory_noise_std=0, rng=random.Random(0), trials=200)
    assert shares == {0: 1.0, 1: 0.0}


def test_selection_is_close_to_a_coin_flip_when_candidates_differ_only_by_judge_noise():
    candidates = [_candidate(0, 80), _candidate(1, 80)]
    noisy = [60.0, 70.0, 80.0, 90.0, 100.0]
    shares = simulate_selection(candidates, {0: noisy, 1: noisy}, trajectory_noise_std=0, rng=random.Random(0), trials=4000)
    assert 0.4 < shares[0] < 0.6


def test_build_pairs_randomizes_sides_keeps_mapping_and_skips_failed_runs():
    raw = [
        {"ticker": f"T{i}", "normal": {"final_report": f"normal-{i}"}, "deep": {"final_report": f"deep-{i}"}} for i in range(20)
    ] + [{"ticker": "FAIL", "normal": {"error": "502"}, "deep": {"final_report": "deep"}}]

    items, mapping = build_pairs(raw, seed=1)

    assert len(items) == 20
    sides = {m["A"] for m in mapping.values()}
    assert sides == {"normal", "deep"}  # 两种分配都出现过，不是固定把某种放在A
    for item in items:
        which = mapping[item["id"]]
        i = item["ticker"][1:]
        assert item["A"] == f"{which['A']}-{i}"
        assert item["B"] == f"{which['B']}-{i}"
    assert build_pairs(raw, seed=1) == (items, mapping)  # 同一个seed可复现


def test_rendered_page_escapes_report_html_and_does_not_leak_the_source():
    items = [{"id": "pair-01", "ticker": "AAPL", "A": "<script>alert(1)</script>", "B": "普通简报"}]
    page = render_page(items)
    assert "<script>alert(1)</script>" not in page
    assert "&lt;script&gt;" in page
    assert "normal" not in page and "deep" not in page


def test_binomial_p_value():
    assert binomial_two_sided_p(5, 10) == 1.0
    assert abs(binomial_two_sided_p(9, 10) - 22 / 1024) < 1e-9
    assert binomial_two_sided_p(0, 0) == 1.0


def test_tally_maps_blind_picks_back_to_analysis_type_and_excludes_ties():
    mapping = {"pair-01": {"A": "deep", "B": "normal"}, "pair-02": {"A": "normal", "B": "deep"}, "pair-03": {"A": "deep", "B": "normal"}, "pair-04": {"A": "deep", "B": "normal"}}
    labels = [
        {"id": "pair-01", "preference": "A"},
        {"id": "pair-02", "preference": "A"},
        {"id": "pair-03", "preference": "tie"},
        {"id": "pair-04", "preference": ""},
    ]

    result = tally(labels, mapping)

    assert (result["deep_wins"], result["normal_wins"], result["ties"], result["unlabeled"]) == (1, 1, 1, 1)
    assert result["deep_win_rate"] == 0.5
