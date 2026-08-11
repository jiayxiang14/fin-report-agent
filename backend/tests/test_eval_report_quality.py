"""eval_report_quality.py 的纯函数部分测试：只测"上一次记录怎么加载/回归怎么
判定"这套逻辑本身，不真的跑Agent Loop（那需要真实调用LLM/SEC EDGAR/Polygon，
真花钱，属于脚本使用者手动执行的范畴，不适合放进单元测试）。
"""

import json

from scripts.eval_report_quality import (
    REGRESSION_THRESHOLD,
    _append_run_log,
    _load_previous_run,
    _print_table,
)


def _fake_result(ticker: str, total_score: float) -> dict:
    return {
        "ticker": ticker,
        "completed": True,
        "stop_reason": "end_turn",
        "rule_score": {
            "traceability": 40.0,
            "traceability_matched": 4,
            "traceability_total": 4,
            "self_verification": 20.0,
            "structure": 20.0,
            "length": 10.0,
            "total": 90.0,
        },
        "llm_score": 80.0,
        "llm_reason": "还行",
        "total_score": total_score,
        "error": None,
    }


def test_load_previous_run_returns_none_when_no_log_file_exists(tmp_path, monkeypatch):
    import scripts.eval_report_quality as mod

    monkeypatch.setattr(mod, "RUNS_LOG_PATH", tmp_path / "eval_runs.jsonl")
    assert _load_previous_run() is None


def test_append_then_load_round_trips_last_line_only(tmp_path, monkeypatch):
    import scripts.eval_report_quality as mod

    log_path = tmp_path / "eval_runs.jsonl"
    monkeypatch.setattr(mod, "RUNS_LOG_PATH", log_path)

    _append_run_log([_fake_result("AAPL", 85.0)])
    _append_run_log([_fake_result("AAPL", 70.0)])  # 第二次跑分数下滑

    previous = _load_previous_run()
    assert previous == {"AAPL": _fake_result("AAPL", 70.0)}  # 只看最后一行，不是历史累加

    lines = log_path.read_text().strip().splitlines()
    assert len(lines) == 2  # 每次调用都是追加，不是覆盖
    assert json.loads(lines[0])["results"][0]["total_score"] == 85.0


def test_load_previous_run_skips_failed_entries_without_total_score(tmp_path, monkeypatch):
    import scripts.eval_report_quality as mod

    log_path = tmp_path / "eval_runs.jsonl"
    monkeypatch.setattr(mod, "RUNS_LOG_PATH", log_path)

    failed = {
        "ticker": "ZZZZ",
        "completed": False,
        "stop_reason": "error",
        "rule_score": None,
        "llm_score": None,
        "llm_reason": None,
        "total_score": None,
        "error": "上游限速",
    }
    _append_run_log([failed, _fake_result("AAPL", 85.0)])

    previous = _load_previous_run()
    assert previous is not None
    assert "ZZZZ" not in previous
    assert "AAPL" in previous


def test_print_table_flags_regression_when_score_drops_past_threshold(capsys):
    previous = {"AAPL": _fake_result("AAPL", 90.0)}
    current = [_fake_result("AAPL", 90.0 - REGRESSION_THRESHOLD)]

    has_regression = _print_table(current, previous)

    assert has_regression is True
    assert "⚠ 回归" in capsys.readouterr().out


def test_print_table_does_not_flag_small_score_drift(capsys):
    previous = {"AAPL": _fake_result("AAPL", 90.0)}
    current = [_fake_result("AAPL", 85.0)]  # 跌了5分，小于10分阈值

    has_regression = _print_table(current, previous)

    assert has_regression is False
    assert "⚠" not in capsys.readouterr().out


def test_print_table_handles_failed_ticker_without_crashing(capsys):
    failed = {
        "ticker": "ZZZZ",
        "completed": False,
        "stop_reason": "error",
        "rule_score": None,
        "llm_score": None,
        "llm_reason": None,
        "total_score": None,
        "error": "网络超时",
    }

    has_regression = _print_table([failed], previous=None)

    assert has_regression is False
    assert "运行失败" in capsys.readouterr().out
