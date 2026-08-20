"""eval_report_quality.py 的纯函数部分测试：只测"上一次记录怎么加载/回归怎么
判定"这套逻辑本身，不真的跑Agent Loop（那需要真实调用LLM/SEC EDGAR/Polygon，
真花钱，属于脚本使用者手动执行的范畴，不适合放进单元测试）。

2026-08-15：脚本改成支持`--repeats`重复采样取均值（单次结果分不清"prompt
效果"和"随机波动"），日志schema从"单次结果"变成"每个ticker一组重复结果的
汇总"（`mean_total_score`/`stdev_total_score`），这里的测试跟着改。
"""

import json

from scripts.eval_report_quality import (
    REGRESSION_THRESHOLD,
    _append_run_log,
    _load_previous_run,
    _print_table,
    _run_one,
)


def _fake_run(total_score: float) -> dict:
    return {
        "ticker": "AAPL",
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
        "trajectory_score": 85.0,
        "trajectory_reason": "还行",
        "total_score": total_score,
        "error": None,
    }


def _fake_summary(ticker: str, mean_total_score: float, n: int = 1, stdev: float = 0.0) -> dict:
    return {
        "ticker": ticker,
        "n": n,
        "n_succeeded": n,
        "runs": [_fake_run(mean_total_score) for _ in range(n)],
        "mean_total_score": mean_total_score,
        "stdev_total_score": stdev,
    }


def _failed_summary(ticker: str, n: int = 1) -> dict:
    return {
        "ticker": ticker,
        "n": n,
        "n_succeeded": 0,
        "runs": [
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
                "error": "上游限速",
            }
            for _ in range(n)
        ],
        "mean_total_score": None,
        "stdev_total_score": None,
    }


def test_load_previous_run_returns_none_when_no_log_file_exists(tmp_path, monkeypatch):
    import scripts.eval_report_quality as mod

    monkeypatch.setattr(mod, "RUNS_LOG_PATH", tmp_path / "eval_runs.jsonl")
    assert _load_previous_run() is None


def test_append_then_load_round_trips_last_line_only(tmp_path, monkeypatch):
    import scripts.eval_report_quality as mod

    log_path = tmp_path / "eval_runs.jsonl"
    monkeypatch.setattr(mod, "RUNS_LOG_PATH", log_path)

    _append_run_log([_fake_summary("AAPL", 85.0)])
    _append_run_log([_fake_summary("AAPL", 70.0)])  # 第二次跑分数下滑

    previous = _load_previous_run()
    assert previous is not None
    assert previous["AAPL"]["mean_total_score"] == 70.0  # 只看最后一行，不是历史累加

    lines = log_path.read_text().strip().splitlines()
    assert len(lines) == 2  # 每次调用都是追加，不是覆盖
    assert json.loads(lines[0])["results"][0]["mean_total_score"] == 85.0


def test_load_previous_run_reads_old_single_run_format_via_total_score_fallback(tmp_path, monkeypatch):
    """兼容改`--repeats`之前跑出来的旧格式日志——那时候`results`里直接是单次
    运行结果，字段是`total_score`不是`mean_total_score`，不能读旧日志直接炸。"""
    import scripts.eval_report_quality as mod

    log_path = tmp_path / "eval_runs.jsonl"
    monkeypatch.setattr(mod, "RUNS_LOG_PATH", log_path)
    log_path.write_text(json.dumps({"timestamp": "x", "results": [_fake_run(91.4)]}) + "\n")

    previous = _load_previous_run()

    assert previous is not None
    assert previous["AAPL"]["mean_total_score"] == 91.4


def test_load_previous_run_skips_failed_entries_without_score(tmp_path, monkeypatch):
    import scripts.eval_report_quality as mod

    log_path = tmp_path / "eval_runs.jsonl"
    monkeypatch.setattr(mod, "RUNS_LOG_PATH", log_path)

    _append_run_log([_failed_summary("ZZZZ"), _fake_summary("AAPL", 85.0)])

    previous = _load_previous_run()
    assert previous is not None
    assert "ZZZZ" not in previous
    assert "AAPL" in previous


def test_print_table_flags_regression_when_mean_score_drops_past_threshold(capsys):
    previous = {"AAPL": _fake_summary("AAPL", 90.0)}
    current = [_fake_summary("AAPL", 90.0 - REGRESSION_THRESHOLD)]

    has_regression = _print_table(current, previous)

    assert has_regression is True
    assert "⚠ 回归" in capsys.readouterr().out


def test_print_table_does_not_flag_small_score_drift(capsys):
    previous = {"AAPL": _fake_summary("AAPL", 90.0)}
    current = [_fake_summary("AAPL", 85.0)]  # 跌了5分，小于10分阈值

    has_regression = _print_table(current, previous)

    assert has_regression is False
    assert "⚠" not in capsys.readouterr().out


def test_print_table_handles_failed_ticker_without_crashing(capsys):
    has_regression = _print_table([_failed_summary("ZZZZ")], previous=None)

    assert has_regression is False
    assert "运行失败" in capsys.readouterr().out


def test_run_one_forwards_temperature_to_agent_loop(monkeypatch):
    """固定temperature是这次改动的核心目的（消除评估噪声），这里验证传给
    `_run_one`的temperature确实原样传到了`run_agent_loop`，不是被脚本自己
    的默认值悄悄覆盖掉。"""
    import scripts.eval_report_quality as mod

    captured_temperature = None

    class _FakeRunResult:
        completed = True
        stop_reason = "end_turn"
        final_report = "<conclusion>a</conclusion><evidence>b</evidence><flags>c</flags>"
        transcript = []
        reasoning_notes = []

    async def fake_run_agent_loop(ticker, on_tool_result=None, temperature=None, enable_compaction=True):
        nonlocal captured_temperature
        captured_temperature = temperature
        return _FakeRunResult()

    monkeypatch.setattr(mod, "run_agent_loop", fake_run_agent_loop)

    import asyncio

    asyncio.run(_run_one("AAPL", use_llm_judge=False, temperature=0.0, enable_compaction=True))

    assert captured_temperature == 0.0


def test_run_one_forwards_enable_compaction_to_agent_loop(monkeypatch):
    """--no-compaction开关（2026-08-16新增）要能真的传到run_agent_loop，
    不能被脚本自己的默认值悄悄覆盖掉——跟temperature那条是同一类回归测试。"""
    import scripts.eval_report_quality as mod

    captured_enable_compaction = None

    class _FakeRunResult:
        completed = True
        stop_reason = "end_turn"
        final_report = "<conclusion>a</conclusion><evidence>b</evidence><flags>c</flags>"
        transcript = []
        reasoning_notes = []

    async def fake_run_agent_loop(ticker, on_tool_result=None, temperature=None, enable_compaction=True):
        nonlocal captured_enable_compaction
        captured_enable_compaction = enable_compaction
        return _FakeRunResult()

    monkeypatch.setattr(mod, "run_agent_loop", fake_run_agent_loop)

    import asyncio

    asyncio.run(_run_one("AAPL", use_llm_judge=False, temperature=0.0, enable_compaction=False))

    assert captured_enable_compaction is False
