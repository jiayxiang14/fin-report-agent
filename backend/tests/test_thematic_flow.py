"""细分主题板块轮动的回归测试：等权合成指数、单只股票拉取失败时的容错
（篮子里一只票坏了不该让整个主题报废）、benchmark拉取失败时的错误包装。
Polygon请求全部mock掉，不发真实网络请求。
"""

import asyncio
from unittest.mock import AsyncMock, patch

import numpy as np
import pandas as pd
import pytest

from app.services.polygon_client import PolygonClientError
from app.services.thematic_flow import (
    CHAIN_POSITION,
    THEMATIC_BASKETS,
    TICKER_TO_THEMES,
    ThematicFlowError,
    _build_equal_weighted_index,
    get_thematic_flow,
)


def _bars(closes) -> pd.DataFrame:
    dates = pd.bdate_range(start="2024-01-01", periods=len(closes))
    return pd.DataFrame(
        {"close": closes, "high": closes, "low": closes, "volume": [0] * len(closes)}, index=dates
    )


def test_equal_weighted_index_averages_normalized_series():
    closes_by_ticker = {
        "A": pd.Series([100.0, 110.0, 120.0]),
        "B": pd.Series([50.0, 55.0, 60.0]),  # 涨幅跟A完全一样（都是+10%再+9.09%），归一化后应该重合
    }
    index = _build_equal_weighted_index(closes_by_ticker)
    assert index.iloc[0] == pytest.approx(100.0)
    assert index.iloc[1] == pytest.approx(110.0)
    assert index.iloc[2] == pytest.approx(120.0)


def test_equal_weighted_index_handles_single_ticker_basket():
    closes_by_ticker = {"SOXX": pd.Series([200.0, 220.0, 180.0])}
    index = _build_equal_weighted_index(closes_by_ticker)
    assert index.iloc[0] == pytest.approx(100.0)
    assert index.iloc[2] == pytest.approx(90.0)


def _random_walk_bars(periods: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    closes = 100 * np.cumprod(1 + rng.normal(0.0005, 0.01, periods))
    return _bars(closes)


def test_get_thematic_flow_happy_path():
    periods = 250

    async def fake_fetch_daily_bars(ticker, client):
        return _random_walk_bars(periods, seed=hash(ticker) % 1000)

    with patch("app.services.thematic_flow.fetch_daily_bars", new=fake_fetch_daily_bars):
        result = asyncio.run(get_thematic_flow())

    assert result.benchmark == "SPY"
    assert {t.theme_name for t in result.themes} == set(THEMATIC_BASKETS.keys())
    for theme in result.themes:
        assert theme.constituent_tickers == THEMATIC_BASKETS[theme.theme_name]
        assert theme.quadrant in {"leading", "weakening", "lagging", "improving"}


def test_a_single_bad_ticker_in_basket_does_not_fail_the_whole_theme():
    """存储篮子里4只票，其中SNDK（2025年才分拆上市）假设查不到数据——
    应该跳过它，用剩下3只票照常算出主题位置，不是让整个'存储'主题消失。"""
    periods = 250

    async def fake_fetch_daily_bars(ticker, client):
        if ticker == "SNDK":
            raise PolygonClientError("查不到这个ticker")
        return _random_walk_bars(periods, seed=hash(ticker) % 1000)

    with patch("app.services.thematic_flow.fetch_daily_bars", new=fake_fetch_daily_bars):
        result = asyncio.run(get_thematic_flow())

    storage_theme = next(t for t in result.themes if t.theme_name == "存储")
    assert storage_theme is not None  # 没有因为SNDK失败就整体消失


def test_all_tickers_in_a_theme_failing_marks_it_unavailable_but_others_still_return():
    periods = 250

    async def fake_fetch_daily_bars(ticker, client):
        if ticker in THEMATIC_BASKETS["半导体"]:
            raise PolygonClientError("查不到")
        return _random_walk_bars(periods, seed=hash(ticker) % 1000)

    with patch("app.services.thematic_flow.fetch_daily_bars", new=fake_fetch_daily_bars):
        result = asyncio.run(get_thematic_flow())

    theme_names = {t.theme_name for t in result.themes}
    assert "半导体" not in theme_names
    assert "存储" in theme_names
    assert "半导体" in result.note  # note里如实说明哪个主题因为数据不足展示不了


def test_every_theme_has_a_chain_position_classified():
    # 防止以后加新主题时忘了分类——没有分类的主题会在ThematicFlowPosition
    # 构造时直接KeyError，这里提前用一个显式的测试断言把这个坑说清楚
    assert set(CHAIN_POSITION.keys()) == set(THEMATIC_BASKETS.keys())
    assert all(position in {"上游", "中游", "下游"} for position in CHAIN_POSITION.values())


def test_get_thematic_flow_attaches_chain_position_per_theme():
    periods = 250

    async def fake_fetch_daily_bars(ticker, client):
        return _random_walk_bars(periods, seed=hash(ticker) % 1000)

    with patch("app.services.thematic_flow.fetch_daily_bars", new=fake_fetch_daily_bars):
        result = asyncio.run(get_thematic_flow())

    for theme in result.themes:
        assert theme.chain_position == CHAIN_POSITION[theme.theme_name]


def test_ticker_to_themes_is_a_deterministic_reverse_lookup():
    # NVDA只在"AI芯片/GPU"篮子里出现过一次，不该跟"半导体"(SOXX)混在一起
    assert TICKER_TO_THEMES["NVDA"] == ["AI芯片/GPU"]
    assert "AAPL" not in TICKER_TO_THEMES  # 不在任何篮子里的公司不该凭空出现


def test_matched_themes_populated_when_ticker_is_a_basket_constituent():
    """NVDA既是"AI芯片/GPU"篮子的成分股，SIC也是"SEMICONDUCTORS & RELATED
    DEVICES"（真实核实过的数据），两个来源都命中，matched_themes应该是两者
    的并集，不是只有篮子匹配那一个。"""
    periods = 250

    async def fake_fetch_daily_bars(ticker, client):
        return _random_walk_bars(periods, seed=hash(ticker) % 1000)

    async def fake_fetch_ticker_details(ticker, client):
        return {"sic_description": "SEMICONDUCTORS & RELATED DEVICES"}

    with (
        patch("app.services.thematic_flow.fetch_daily_bars", new=fake_fetch_daily_bars),
        patch("app.services.thematic_flow.fetch_ticker_details", new=fake_fetch_ticker_details),
    ):
        result = asyncio.run(get_thematic_flow("nvda"))  # 故意用小写，验证会被normalize

    assert result.ticker == "NVDA"
    assert set(result.matched_themes) == {"AI芯片/GPU", "半导体"}
    # "AI芯片/GPU"是篮子成分股匹配，"半导体"只有SIC能识别（NVDA不在"半导体"
    # 篮子的手动名单里）——sic_matched_themes要能把这个区分暴露出来，
    # 不能让matched_themes的并集掩盖掉"这条是SIC识别的"这个信息
    assert result.sic_matched_themes == ["半导体"]


def test_matched_themes_empty_when_ticker_not_in_any_basket_or_sic():
    periods = 250

    async def fake_fetch_daily_bars(ticker, client):
        return _random_walk_bars(periods, seed=hash(ticker) % 1000)

    async def fake_fetch_ticker_details(ticker, client):
        return {"sic_description": "BOTTLED & CANNED SOFT DRINKS"}  # KO的真实SIC类描述风格，不匹配任何主题

    with (
        patch("app.services.thematic_flow.fetch_daily_bars", new=fake_fetch_daily_bars),
        patch("app.services.thematic_flow.fetch_ticker_details", new=fake_fetch_ticker_details),
    ):
        result = asyncio.run(get_thematic_flow("KO"))

    assert result.ticker == "KO"
    assert result.matched_themes == []


def test_matched_themes_populated_by_sic_alone_when_not_in_any_basket():
    """核心场景：一家不在任何篮子手动名单里、但SIC分类确实是存储设备的公司
    （比如假设一家没被我列进"存储"篮子的其它硬盘厂商），应该单靠SIC匹配就能
    识别出"存储"这个主题，不需要事先手动把它加进篮子。"""
    periods = 250

    async def fake_fetch_daily_bars(ticker, client):
        return _random_walk_bars(periods, seed=hash(ticker) % 1000)

    async def fake_fetch_ticker_details(ticker, client):
        return {"sic_description": "COMPUTER STORAGE DEVICES"}

    with (
        patch("app.services.thematic_flow.fetch_daily_bars", new=fake_fetch_daily_bars),
        patch("app.services.thematic_flow.fetch_ticker_details", new=fake_fetch_ticker_details),
    ):
        result = asyncio.run(get_thematic_flow("SOMEDRIVE"))

    assert result.matched_themes == ["存储"]
    assert result.sic_matched_themes == ["存储"]  # 全靠SIC识别，篮子里压根没有这只票


def test_matched_themes_falls_back_to_basket_only_when_sic_lookup_fails():
    """SIC查询失败（比如Polygon查不到这个ticker）不该拖累整个匹配结果——
    篮子成分股匹配的部分照常返回，这是软性降级不是硬错误。"""
    from app.services.company_profile import CompanyProfileError

    periods = 250

    async def fake_fetch_daily_bars(ticker, client):
        return _random_walk_bars(periods, seed=hash(ticker) % 1000)

    async def fake_fetch_ticker_details(ticker, client):
        raise CompanyProfileError("Polygon 查不到这个ticker")

    with (
        patch("app.services.thematic_flow.fetch_daily_bars", new=fake_fetch_daily_bars),
        patch("app.services.thematic_flow.fetch_ticker_details", new=fake_fetch_ticker_details),
    ):
        result = asyncio.run(get_thematic_flow("NVDA"))

    assert result.matched_themes == ["AI芯片/GPU"]  # 篮子匹配还在，只是SIC那部分拿不到


def test_matched_themes_empty_and_ticker_none_when_not_provided():
    periods = 250

    async def fake_fetch_daily_bars(ticker, client):
        return _random_walk_bars(periods, seed=hash(ticker) % 1000)

    with patch("app.services.thematic_flow.fetch_daily_bars", new=fake_fetch_daily_bars):
        result = asyncio.run(get_thematic_flow())

    assert result.ticker is None
    assert result.matched_themes == []


def test_benchmark_fetch_failure_raises_thematic_flow_error():
    with patch(
        "app.services.thematic_flow.fetch_daily_bars",
        new=AsyncMock(side_effect=PolygonClientError("上游错误")),
    ):
        with pytest.raises(ThematicFlowError):
            asyncio.run(get_thematic_flow())
