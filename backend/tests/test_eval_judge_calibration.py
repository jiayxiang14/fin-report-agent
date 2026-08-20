"""eval_judge_calibration.py 的纯函数部分测试：Spearman相关系数（必须正确处理
并列秩，复盘时发现的漏洞）、held-out集加载（排除锚点/拒绝未标注完成的数据/
标出跟锚点同一次Best-of-N运行的样本，防止同run泄漏被当成真实泛化能力）。"""

import asyncio
import json

import pytest

from scripts.eval_judge_calibration import (
    _append_run_log,
    _correlation_stats,
    _load_held_out_items,
    _rank_with_average_ties,
    _report_correlation,
    _run_one,
    mean_absolute_error,
    spearman_correlation,
)


def test_rank_with_average_ties_assigns_average_rank_to_tied_values():
    # 值: [10, 20, 20, 30] -> 秩本该是 [1,2,3,4]，但两个20并列，各拿(2+3)/2=2.5
    ranks = _rank_with_average_ties([10, 20, 20, 30])
    assert ranks == [1.0, 2.5, 2.5, 4.0]


def test_spearman_correlation_is_one_for_perfectly_monotonic_data():
    assert spearman_correlation([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)


def test_spearman_correlation_is_negative_one_for_inverted_data():
    assert spearman_correlation([1, 2, 3, 4], [40, 30, 20, 10]) == pytest.approx(-1.0)


def test_spearman_correlation_handles_ties_without_crashing():
    # 朴素的"无并列"简化公式在这种输入下会算出有偏结果；这里只要求不崩、
    # 且落在[-1, 1]合法区间内
    result = spearman_correlation([5, 5, 5, 8], [3, 3, 3, 9])
    assert result is not None
    assert -1.0 <= result <= 1.0


def test_spearman_correlation_returns_none_for_fewer_than_two_points():
    assert spearman_correlation([5], [5]) is None


def test_mean_absolute_error_computes_average_absolute_difference():
    assert mean_absolute_error([8, 6], [6, 6]) == pytest.approx(1.0)


def test_load_held_out_items_excludes_anchor_role_to_avoid_leakage(tmp_path):
    path = tmp_path / "eval_set.json"
    path.write_text(
        json.dumps(
            {
                "items": [
                    {
                        "ticker": "AAPL",
                        "timestamp": "t1",
                        "final_report": "r1",
                        "human_score": 9,
                        "human_reason": "好",
                        "role": "anchor",
                    },
                    {
                        "ticker": "MSFT",
                        "timestamp": "t2",
                        "final_report": "r2",
                        "human_score": 5,
                        "human_reason": "一般",
                        "role": "eval",
                    },
                ]
            }
        )
    )

    items, anchor_timestamps = _load_held_out_items(path)

    assert len(items) == 1
    assert items[0]["ticker"] == "MSFT"
    assert anchor_timestamps == {"t1"}


def test_load_held_out_items_flags_samples_from_the_same_run_as_an_anchor(tmp_path):
    # Best-of-N同一次运行会产出多个候选（仅temperature不同，底层财务数据相同）
    # ——如果锚点和某条held-out样本共享同一个timestamp，说明它们是"姊妹候选"，
    # 用它验证相关性会有泄漏风险，调用方需要能识别出这种样本
    path = tmp_path / "eval_set.json"
    path.write_text(
        json.dumps(
            {
                "items": [
                    {
                        "ticker": "AAPL",
                        "timestamp": "run-1",
                        "final_report": "r1",
                        "human_score": 5,
                        "human_reason": "x",
                        "role": "anchor",
                    },
                    {
                        "ticker": "AAPL",
                        "timestamp": "run-1",
                        "final_report": "r2",
                        "human_score": 8,
                        "human_reason": "x",
                        "role": "eval",
                    },
                    {
                        "ticker": "GOOGL",
                        "timestamp": "run-2",
                        "final_report": "r3",
                        "human_score": 7,
                        "human_reason": "x",
                        "role": "eval",
                    },
                ]
            }
        )
    )

    items, anchor_timestamps = _load_held_out_items(path)

    assert len(items) == 2
    assert items[0]["timestamp"] in anchor_timestamps  # AAPL held-out样本跟锚点同run
    assert items[1]["timestamp"] not in anchor_timestamps  # GOOGL held-out样本是独立run


def test_load_held_out_items_rejects_unlabeled_entries(tmp_path):
    path = tmp_path / "eval_set.json"
    path.write_text(
        json.dumps(
            {"items": [{"ticker": "AAPL", "timestamp": "t1", "final_report": "r1", "human_score": None, "role": None}]}
        )
    )

    with pytest.raises(ValueError, match="未标注完成"):
        _load_held_out_items(path)


def test_load_held_out_items_raises_when_file_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        _load_held_out_items(tmp_path / "does_not_exist.json")


def test_report_correlation_prints_sample_size_and_correlation(capsys):
    _report_correlation("测试标签", [(1.0, 1.0), (2.0, 2.0), (3.0, 3.0)])
    out = capsys.readouterr().out
    assert "测试标签" in out
    assert "n=3" in out
    assert "1.00" in out


def test_report_correlation_handles_fewer_than_two_pairs_without_crashing(capsys):
    _report_correlation("空标签", [])
    assert "样本不足2条" in capsys.readouterr().out


def test_correlation_stats_matches_report_correlation_numbers():
    stats = _correlation_stats([(1.0, 1.0), (2.0, 2.0), (3.0, 3.0)])
    assert stats == {"n": 3, "spearman": pytest.approx(1.0), "mae": pytest.approx(0.0)}


def test_correlation_stats_handles_fewer_than_two_pairs():
    assert _correlation_stats([]) == {"n": 0, "spearman": None, "mae": None}


def test_append_run_log_writes_one_jsonl_line_per_call(tmp_path, monkeypatch):
    import scripts.eval_judge_calibration as mod

    log_path = tmp_path / "eval_judge_calibration_runs.jsonl"
    monkeypatch.setattr(mod, "RUNS_LOG_PATH", log_path)

    _append_run_log({"timestamp": "t1", "full": {"n": 3, "spearman": 0.5}})
    _append_run_log({"timestamp": "t2", "full": {"n": 5, "spearman": 0.7}})

    lines = log_path.read_text().strip().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[1])["full"]["spearman"] == 0.7


def test_run_one_converts_zero_to_hundred_scale_back_to_one_to_ten(monkeypatch):
    import scripts.eval_judge_calibration as mod

    async def fake_score_llm_judge(report):
        return 80.0, "理由"

    monkeypatch.setattr(mod, "score_llm_judge", fake_score_llm_judge)

    score, reason = asyncio.run(_run_one({"final_report": "x"}))

    assert score == 8.0
    assert reason == "理由"


def test_run_one_returns_none_when_judge_score_is_none(monkeypatch):
    import scripts.eval_judge_calibration as mod

    async def fake_score_llm_judge(report):
        return None, None

    monkeypatch.setattr(mod, "score_llm_judge", fake_score_llm_judge)

    score, reason = asyncio.run(_run_one({"final_report": "x"}))

    assert score is None
    assert reason is None


def test_main_limit_only_scores_the_first_n_held_out_items(tmp_path, monkeypatch, capsys):
    # --limit是给"先小批试跑确认链路没问题"用的，这里验证它确实只对前N条
    # held-out样本发起打分调用，不是跑完全部再截断输出
    import scripts.eval_judge_calibration as mod

    path = tmp_path / "eval_set.json"
    path.write_text(
        json.dumps(
            {
                "items": [
                    {"ticker": t, "timestamp": f"t{i}", "final_report": f"r{i}", "human_score": 7, "role": "eval"}
                    for i, t in enumerate(["AAPL", "AMZN", "GOOGL", "NBIS"])
                ]
            }
        )
    )
    monkeypatch.setattr(mod, "EVAL_SET_PATH", path)
    monkeypatch.setattr(mod, "RUNS_LOG_PATH", tmp_path / "runs.jsonl")

    scored_reports: list[str] = []

    async def fake_score_llm_judge(report):
        scored_reports.append(report)
        return 70.0, "ok"

    monkeypatch.setattr(mod, "score_llm_judge", fake_score_llm_judge)

    asyncio.run(mod.main(limit=2))

    assert scored_reports == ["r0", "r1"]
    assert "只跑前2条" in capsys.readouterr().out


def test_main_offset_skips_the_first_n_held_out_items(tmp_path, monkeypatch, capsys):
    # --offset是给"已经用--limit试跑过前几条，只想跑剩下的"用的
    import scripts.eval_judge_calibration as mod

    path = tmp_path / "eval_set.json"
    path.write_text(
        json.dumps(
            {
                "items": [
                    {"ticker": t, "timestamp": f"t{i}", "final_report": f"r{i}", "human_score": 7, "role": "eval"}
                    for i, t in enumerate(["AAPL", "AMZN", "GOOGL", "NBIS"])
                ]
            }
        )
    )
    monkeypatch.setattr(mod, "EVAL_SET_PATH", path)
    monkeypatch.setattr(mod, "RUNS_LOG_PATH", tmp_path / "runs.jsonl")

    scored_reports: list[str] = []

    async def fake_score_llm_judge(report):
        scored_reports.append(report)
        return 70.0, "ok"

    monkeypatch.setattr(mod, "score_llm_judge", fake_score_llm_judge)

    asyncio.run(mod.main(offset=2))

    assert scored_reports == ["r2", "r3"]
    assert "跳过前2条" in capsys.readouterr().out
