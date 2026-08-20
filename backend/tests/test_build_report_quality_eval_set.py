"""build_report_quality_eval_set.py 的纯函数部分测试：候选加载/过滤、分层抽样、
输出结构不泄漏历史分数——这几点都是复盘设计时特意确认过的行为，不是随手实现。"""

import json

from scripts.build_report_quality_eval_set import (
    LABELING_INSTRUCTIONS,
    MIN_REPORT_LENGTH_FOR_LABELING,
    _load_candidates,
    _stratified_sample,
    build_eval_set,
)


def _run(ticker: str, timestamp: str, candidates: list[dict]) -> dict:
    return {"ticker": ticker, "timestamp": timestamp, "candidates": candidates}


_DEFAULT_REPORT = "<conclusion>x</conclusion><evidence>足够长，超过最短长度过滤阈值</evidence>" * (
    MIN_REPORT_LENGTH_FOR_LABELING // 30 + 1
)


def _candidate(score: float, report: str = _DEFAULT_REPORT, completed: bool = True) -> dict:
    return {"total_score": score, "final_report": report, "completed": completed}


def test_load_candidates_skips_incomplete_and_reportless_candidates(tmp_path):
    runs_path = tmp_path / "runs.jsonl"
    runs_path.write_text(
        "\n".join(
            json.dumps(r)
            for r in [
                _run("AAPL", "t1", [_candidate(80), {"total_score": 10, "final_report": None, "completed": True}]),
                _run("AAPL", "t2", [_candidate(90, completed=False)]),
            ]
        )
    )

    candidates = _load_candidates(runs_path)

    assert len(candidates) == 1
    assert candidates[0]["total_score"] == 80


def test_load_candidates_returns_empty_list_when_file_missing(tmp_path):
    assert _load_candidates(tmp_path / "does_not_exist.jsonl") == []


def test_load_candidates_filters_out_test_fixture_placeholder_reports(tmp_path):
    # .cache/best_of_n_runs.jsonl里混着单测跑测试时写入的假数据（比如"候选0"这种
    # 几个字符的占位符），真实简报不可能这么短，得在抽样前过滤掉，否则会把这些
    # 假数据当真实候选拿去给人标注
    runs_path = tmp_path / "runs.jsonl"
    runs_path.write_text(
        "\n".join(
            json.dumps(r)
            for r in [
                _run("AAPL", "t1", [_candidate(80, report="候选0")]),
                _run("AAPL", "t2", [_candidate(70)]),  # 用_DEFAULT_REPORT，长度已保证超过阈值
            ]
        )
    )

    candidates = _load_candidates(runs_path)

    assert len(candidates) == 1
    assert candidates[0]["final_report"] != "候选0"


def test_stratified_sample_returns_all_when_pool_at_or_below_target():
    pool = [{"total_score": 80, "timestamp": "t1"}, {"total_score": 60, "timestamp": "t2"}]

    picks = _stratified_sample(pool, target=3)

    assert len(picks) == 2
    assert [p["total_score"] for p in picks] == [60, 80]  # 排过序


def test_stratified_sample_spreads_across_score_range():
    pool = [{"total_score": score, "timestamp": f"t{score}"} for score in range(40, 101, 5)]

    picks = _stratified_sample(pool, target=3)

    scores = sorted(p["total_score"] for p in picks)
    assert scores[0] == 40  # 最低分
    assert scores[-1] == 100  # 最高分
    assert 60 <= scores[1] <= 80  # 中间那个落在中间区间，不是跟两端挤在一起


def test_stratified_sample_avoids_picking_duplicate_timestamps_when_alternatives_exist():
    # 分数完全一样，但timestamp不同——不该重复选中同一个timestamp
    pool = [
        {"total_score": 50, "timestamp": "run-a"},
        {"total_score": 50, "timestamp": "run-a"},
        {"total_score": 50, "timestamp": "run-b"},
    ]

    picks = _stratified_sample(pool, target=2)

    timestamps = {p["timestamp"] for p in picks}
    assert timestamps == {"run-a", "run-b"}


def test_build_eval_set_output_never_contains_historical_llm_scores(tmp_path):
    runs_path = tmp_path / "runs.jsonl"
    runs_path.write_text(json.dumps(_run("AAPL", "t1", [_candidate(80)])) + "\n")

    content, _warnings = build_eval_set(runs_path)

    item = content["items"][0]
    assert "total_score" not in item
    assert "llm_score" not in item
    assert item["human_score"] is None
    assert item["human_reason"] is None
    assert item["role"] is None
    assert item["final_report"] == _DEFAULT_REPORT
    assert content["labeling_instructions"] == LABELING_INSTRUCTIONS


def test_build_eval_set_warns_when_ticker_has_fewer_than_two_distinct_runs(tmp_path):
    runs_path = tmp_path / "runs.jsonl"
    runs_path.write_text(
        "\n".join(
            json.dumps(r)
            for r in [
                _run("NBIS", "t1", [_candidate(70), _candidate(80), _candidate(90)]),  # 同一个run里的3个候选
            ]
        )
    )

    _content, warnings = build_eval_set(runs_path)

    assert any("NBIS" in w for w in warnings)


def test_build_eval_set_does_not_warn_when_ticker_has_diverse_runs(tmp_path):
    runs_path = tmp_path / "runs.jsonl"
    runs_path.write_text(
        "\n".join(
            json.dumps(r)
            for r in [
                _run("AAPL", "t1", [_candidate(70)]),
                _run("AAPL", "t2", [_candidate(90)]),
            ]
        )
    )

    _content, warnings = build_eval_set(runs_path)

    assert warnings == []
