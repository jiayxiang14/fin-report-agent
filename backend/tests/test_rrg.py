"""RRG核心数学（compute_rrg/classify_quadrant）的单测。这两个函数之前是
sector_rotation.py的私有实现，完全没有测试覆盖——抽成共享模块rrg.py
（供thematic_flow.py复用）时顺手补上，不是范围扩大。
"""

import numpy as np
import pandas as pd
import pytest

from app.services.rrg import (
    LONG_WINDOW,
    MOMENTUM_NORM_WINDOW,
    MOMENTUM_WINDOW,
    classify_quadrant,
    compute_rrg,
)


def _make_series(values, periods) -> pd.Series:
    dates = pd.bdate_range(start="2024-01-01", periods=periods)
    return pd.Series(values, index=dates)


@pytest.fixture
def min_periods() -> int:
    # rs_ratio需要LONG_WINDOW个点才有第一个非NaN值，rs_momentum还要再等
    # MOMENTUM_WINDOW+MOMENTUM_NORM_WINDOW，多留一点余量
    return LONG_WINDOW + MOMENTUM_WINDOW + MOMENTUM_NORM_WINDOW + 20


def test_compute_rrg_outputs_values_centered_around_100(min_periods):
    rng = np.random.default_rng(42)
    series_closes = _make_series(100 * np.cumprod(1 + rng.normal(0.001, 0.01, min_periods)), min_periods)
    benchmark_closes = _make_series(100 * np.cumprod(1 + rng.normal(0.0, 0.01, min_periods)), min_periods)

    result = compute_rrg(series_closes, benchmark_closes)

    assert not result.empty
    assert set(result.columns) == {"rs_ratio", "rs_momentum"}
    # z-score归一化后应该以100为中枢波动，不会是随机的极端值
    assert 80 < result["rs_ratio"].mean() < 120
    assert 80 < result["rs_momentum"].mean() < 120


def test_compute_rrg_drops_warmup_period_with_nans(min_periods):
    series_closes = _make_series([100.0 + i * 0.1 for i in range(min_periods)], min_periods)
    benchmark_closes = _make_series([100.0] * min_periods, min_periods)

    result = compute_rrg(series_closes, benchmark_closes)

    assert result.isna().sum().sum() == 0  # dropna应该已经清掉warm-up期的NaN
    assert len(result) < min_periods  # 一定比原始输入短（滚动窗口的warm-up期被砍掉）


def test_compute_rrg_empty_when_not_enough_history():
    series_closes = _make_series([100.0, 101.0, 99.0], 3)
    benchmark_closes = _make_series([100.0, 100.0, 100.0], 3)

    result = compute_rrg(series_closes, benchmark_closes)

    assert result.empty


@pytest.mark.parametrize(
    "rs_ratio,rs_momentum,expected",
    [
        (105, 105, "leading"),
        (105, 95, "weakening"),
        (95, 95, "lagging"),
        (95, 105, "improving"),
        (100, 100, "leading"),  # 边界值：>=100 都算强
    ],
)
def test_classify_quadrant(rs_ratio, rs_momentum, expected):
    assert classify_quadrant(rs_ratio, rs_momentum) == expected
