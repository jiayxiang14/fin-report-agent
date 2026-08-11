"""新增工具 get_price_reaction 的回归测试。核心逻辑是"用8-K业绩快报的精确申报时间戳，
判断反应窗口落在申报当天还是下一交易日"——这是最容易出错的部分，用合成数据覆盖收盘前/
收盘后/周末申报三种情况，不依赖真实网络请求。
"""

import asyncio
from unittest.mock import AsyncMock, patch

import pandas as pd
import pytest

from app.services.price_reaction import (
    FOLLOW_THROUGH_WINDOW,
    _avg_volume_baseline,
    _compute_follow_through,
    _resolve_reaction_window,
    find_latest_earnings_8k,
)

# 用 bdate_range 生成纯工作日（周一到周五）的交易日序列，不依赖对具体某一天是星期几的
# 记忆——真实的月份/星期对应关系交给pandas算，测试只关心"收盘前/收盘后/非交易日"这几种
# 相对关系是否被正确处理。
TRADING_DATES = pd.bdate_range(start="2026-07-20", end="2026-08-10")


def test_after_close_filing_pushes_reaction_to_next_trading_day():
    # 真实抓到的AMZN例子：2026-07-30T20:06:23Z，UTC，7月是夏令时(EDT, UTC-4)，
    # 换算成美东本地时间是16:06，晚于16:00收盘
    thursday = _find_weekday(TRADING_DATES, "2026-07-27", target_weekday=3)  # 找到一个周四
    accepted_at = f"{thursday.date().isoformat()}T20:06:23.000Z"

    result = _resolve_reaction_window(accepted_at, TRADING_DATES)

    assert result is not None
    pre_date, post_date = result
    assert pre_date == thursday
    assert post_date == thursday + pd.Timedelta(days=1)  # 次日（周五，还是交易日）


def test_before_close_filing_reacts_same_day():
    thursday = _find_weekday(TRADING_DATES, "2026-07-27", target_weekday=3)
    # UTC 12:00，7月EDT换算成本地是08:00，早于16:00收盘
    accepted_at = f"{thursday.date().isoformat()}T12:00:00.000Z"

    result = _resolve_reaction_window(accepted_at, TRADING_DATES)

    assert result is not None
    pre_date, post_date = result
    assert post_date == thursday
    # pre_date 应该是 thursday 之前最近的一个交易日
    assert post_date - pre_date <= pd.Timedelta(days=3)
    assert pre_date < thursday


def test_weekend_filing_resolves_to_next_trading_day_regardless_of_time():
    saturday = _find_weekday(TRADING_DATES, "2026-07-20", target_weekday=5)
    accepted_at = f"{saturday.date().isoformat()}T09:00:00.000Z"  # 时间是几点不重要

    result = _resolve_reaction_window(accepted_at, TRADING_DATES)

    assert result is not None
    pre_date, post_date = result
    assert post_date.weekday() < 5  # 落在真实交易日上，不是周末
    assert post_date > saturday


def _find_weekday(dates: pd.DatetimeIndex, start_hint: str, target_weekday: int) -> pd.Timestamp:
    """在给定的日期序列附近，找一个恰好是 target_weekday（0=周一...6=周日）的日子。"""
    for offset in range(14):
        candidate = pd.Timestamp(start_hint) + pd.Timedelta(days=offset)
        if candidate.weekday() == target_weekday:
            return candidate
    raise AssertionError("没找到符合条件的日期，测试fixture写错了")


def test_avg_volume_baseline_excludes_reaction_day_and_uses_prior_window():
    dates = pd.bdate_range(start="2026-06-01", end="2026-07-31")
    bars = pd.DataFrame(
        {
            "close": [100.0] * len(dates),
            # 前面全是 1000 成交量，最近几天故意设得很高，模拟反应当天的放量
            "volume": [1000.0] * (len(dates) - 3) + [9000.0, 9500.0, 9800.0],
        },
        index=dates,
    )
    pre_date = dates[-3]  # 反应窗口的"事前"那天

    baseline = _avg_volume_baseline(bars, pre_date, window=20)

    assert baseline == 1000.0  # 只看 pre_date 之前的20天，不应该被后面的放量影响


def test_find_latest_earnings_8k_filters_by_item_code_and_picks_latest():
    submissions = {
        "filings": {
            "recent": {
                "form": ["8-K", "8-K", "10-Q", "8-K"],
                "items": ["8.01,9.01", "2.02,9.01", "", "2.02,9.01"],
                "filingDate": ["2026-06-12", "2026-04-29", "2026-04-30", "2026-07-30"],
                "acceptanceDateTime": [
                    "2026-06-12T12:00:00.000Z",
                    "2026-04-29T20:00:00.000Z",
                    "2026-04-30T21:00:00.000Z",
                    "2026-07-30T20:06:23.000Z",
                ],
            }
        }
    }

    with patch(
        "app.services.price_reaction.fetch_submissions",
        new=AsyncMock(return_value=submissions),
    ):
        result = asyncio.run(find_latest_earnings_8k("0001018724", client=None))

    assert result is not None
    assert result["filing_date"] == "2026-07-30"  # 最新一条带"2.02"的8-K，不是最新的8-K本身
    assert result["acceptance_datetime"] == "2026-07-30T20:06:23.000Z"


def test_compute_follow_through_measures_giveback_after_up_move():
    dates = pd.bdate_range(start="2026-07-01", periods=15)
    post_date = dates[5]
    # post_date 之后5个交易日：108 -> 106 -> 102（回吐最深，且放量）-> 104 -> 105
    closes = [100.0] * 6 + [108.0, 106.0, 102.0, 104.0, 105.0] + [105.0] * 4
    volumes = [1000.0] * 6 + [1000.0, 1000.0, 5000.0, 1000.0, 1000.0] + [1000.0] * 4
    bars = pd.DataFrame({"close": closes, "volume": volumes}, index=dates)

    extreme_close, reversal_pct, volume_ratio = _compute_follow_through(
        bars, pre_close=100.0, post_close=110.0, post_date=post_date, baseline=1000.0
    )

    assert extreme_close == 102.0
    assert reversal_pct == pytest.approx((110.0 - 102.0) / (110.0 - 100.0) * 100)
    assert volume_ratio == 5.0


def test_compute_follow_through_measures_giveback_after_down_move():
    dates = pd.bdate_range(start="2026-07-01", periods=15)
    post_date = dates[5]
    # 跌势看窗口内最高收盘价，95 是反弹最高点
    closes = [100.0] * 6 + [92.0, 93.0, 95.0, 94.0, 93.0] + [93.0] * 4
    volumes = [1000.0] * 6 + [1000.0, 1000.0, 3000.0, 1000.0, 1000.0] + [1000.0] * 4
    bars = pd.DataFrame({"close": closes, "volume": volumes}, index=dates)

    extreme_close, reversal_pct, volume_ratio = _compute_follow_through(
        bars, pre_close=100.0, post_close=90.0, post_date=post_date, baseline=1000.0
    )

    assert extreme_close == 95.0
    assert reversal_pct == pytest.approx((95.0 - 90.0) / (100.0 - 90.0) * 100)
    assert volume_ratio == 3.0


def test_compute_follow_through_returns_none_when_window_data_insufficient():
    dates = pd.bdate_range(start="2026-07-01", periods=8)
    post_date = dates[5]  # 之后只剩2个交易日，不够 FOLLOW_THROUGH_WINDOW=5
    bars = pd.DataFrame(
        {"close": [100.0] * len(dates), "volume": [1000.0] * len(dates)}, index=dates
    )

    result = _compute_follow_through(
        bars, pre_close=100.0, post_close=108.0, post_date=post_date, baseline=1000.0
    )

    assert result == (None, None, None)
    assert FOLLOW_THROUGH_WINDOW == 5  # 断言常量没被悄悄改掉导致测试假设失效


def test_find_latest_earnings_8k_returns_none_when_no_earnings_8k():
    submissions = {
        "filings": {
            "recent": {
                "form": ["8-K", "10-Q"],
                "items": ["8.01,9.01", ""],
                "filingDate": ["2026-06-12", "2026-04-30"],
                "acceptanceDateTime": ["2026-06-12T12:00:00.000Z", "2026-04-30T21:00:00.000Z"],
            }
        }
    }

    with patch(
        "app.services.price_reaction.fetch_submissions",
        new=AsyncMock(return_value=submissions),
    ):
        result = asyncio.run(find_latest_earnings_8k("0001018724", client=None))

    assert result is None
