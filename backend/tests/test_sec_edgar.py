"""对第1阶段踩过的两个坑做回归测试，防止以后改代码时悄悄改回去。"""

import pytest

from app.services.sec_edgar import (
    _is_full_year,
    _is_single_quarter,
    extract_key_metrics,
    extract_metrics_history,
)


def _point(end, form, filed, val, start=None, fy=None, fp=None):
    point = {"end": end, "val": val, "form": form, "filed": filed}
    if start is not None:
        point["start"] = start
    if fy is not None:
        point["fy"] = fy
    if fp is not None:
        point["fp"] = fp
    return point


def _facts_json(tag_to_points: dict[str, list[dict]]) -> dict:
    return {
        "facts": {
            "us-gaap": {
                tag: {"units": {"USD": points}} for tag, points in tag_to_points.items()
            }
        }
    }


def test_prefers_most_recently_disclosed_candidate_tag_over_stale_one():
    """踩坑（一）：Apple 2018年后从 Revenues 换成了新标签，旧代码会命中第一个
    "有数据"的候选标签（Revenues），取到停用多年的旧值。"""
    facts_json = _facts_json(
        {
            "Revenues": [
                _point("2018-09-29", "10-K", "2018-11-05", 265_595_000_000, start="2017-10-01", fy=2018, fp="FY"),
            ],
            "RevenueFromContractWithCustomerExcludingAssessedTax": [
                _point("2025-09-27", "10-K", "2025-10-31", 416_161_000_000, start="2024-09-29", fy=2025, fp="FY"),
            ],
        }
    )

    metrics = extract_key_metrics(facts_json)

    assert metrics["revenue"].tag == "RevenueFromContractWithCustomerExcludingAssessedTax"
    assert metrics["revenue"].latest_annual.val == 416_161_000_000


def test_quarterly_value_excludes_year_to_date_cumulative_with_same_end_date():
    """踩坑（二）：同一个 end 日期下，10-Q 里混着单季度值(约91天)和财年累计值(约273天)，
    旧代码只看 form+end，可能捞到累计值当成"最新季度营收"。"""
    facts_json = _facts_json(
        {
            "RevenueFromContractWithCustomerExcludingAssessedTax": [
                # 9个月累计（财年至今），end 日期和单季度值完全一样
                _point("2026-06-27", "10-Q", "2026-07-31", 364_357_000_000, start="2025-09-28", fy=2026, fp="Q3"),
                # 真正的单季度值
                _point("2026-06-27", "10-Q", "2026-07-31", 109_417_000_000, start="2026-03-29", fy=2026, fp="Q3"),
            ],
        }
    )

    metrics = extract_key_metrics(facts_json)

    assert metrics["revenue"].latest_quarterly.val == 109_417_000_000


def test_instantaneous_concepts_without_start_are_unaffected():
    """总资产这类存量指标本来就没有 start，不应该被 duration 过滤误伤。"""
    facts_json = _facts_json(
        {
            "Assets": [
                _point("2026-06-27", "10-Q", "2026-07-31", 383_266_000_000, fy=2026, fp="Q3"),
            ],
        }
    )

    metrics = extract_key_metrics(facts_json)

    assert metrics["total_assets"].latest_quarterly.val == 383_266_000_000


def test_negative_operating_cash_flow_passes_through_unfiltered():
    """净利润为正但经营现金流转负是真实存在的信号，不能被当成异常值滤掉。"""
    facts_json = _facts_json(
        {
            "NetCashProvidedByUsedInOperatingActivities": [
                _point("2025-12-31", "10-K", "2026-02-01", -5_000_000_000, start="2025-01-01", fy=2025, fp="FY"),
            ],
        }
    )

    metrics = extract_key_metrics(facts_json)

    assert metrics["operating_cash_flow"].latest_annual.val == -5_000_000_000


def test_free_cash_flow_is_derived_from_operating_cash_flow_minus_capex():
    facts_json = _facts_json(
        {
            "NetCashProvidedByUsedInOperatingActivities": [
                _point("2025-12-31", "10-K", "2026-02-01", 100_000_000_000, start="2025-01-01", fy=2025, fp="FY"),
            ],
            "PaymentsToAcquirePropertyPlantAndEquipment": [
                _point("2025-12-31", "10-K", "2026-02-01", 30_000_000_000, start="2025-01-01", fy=2025, fp="FY"),
            ],
        }
    )

    metrics = extract_key_metrics(facts_json)

    assert metrics["operating_cash_flow"].latest_annual.val == 100_000_000_000
    assert metrics["capital_expenditures"].latest_annual.val == 30_000_000_000
    assert metrics["free_cash_flow"].latest_annual.val == 70_000_000_000


def test_free_cash_flow_absent_when_capex_tag_missing():
    """只披露了经营现金流、没有资本支出数据时，不该硬凑一个自由现金流出来。"""
    facts_json = _facts_json(
        {
            "NetCashProvidedByUsedInOperatingActivities": [
                _point("2025-12-31", "10-K", "2026-02-01", 100_000_000_000, start="2025-01-01", fy=2025, fp="FY"),
            ],
        }
    )

    metrics = extract_key_metrics(facts_json)

    assert "free_cash_flow" not in metrics
    assert metrics["operating_cash_flow"].latest_annual.val == 100_000_000_000


def test_free_cash_flow_skips_periods_without_matching_end_date_on_both_sides():
    """两个源标签的披露节奏不一定同步，只对齐双方都有数据的期间，不用0去凑缺失的一侧。"""
    facts_json = _facts_json(
        {
            "NetCashProvidedByUsedInOperatingActivities": [
                _point("2025-12-31", "10-K", "2026-02-01", 100_000_000_000, start="2025-01-01", fy=2025, fp="FY"),
                _point("2024-12-31", "10-K", "2025-02-01", 90_000_000_000, start="2024-01-01", fy=2024, fp="FY"),
            ],
            "PaymentsToAcquirePropertyPlantAndEquipment": [
                # 只有2024年这期披露了资本支出，2025年缺失
                _point("2024-12-31", "10-K", "2025-02-01", 20_000_000_000, start="2024-01-01", fy=2024, fp="FY"),
            ],
        }
    )

    metrics = extract_key_metrics(facts_json)

    assert metrics["free_cash_flow"].latest_annual.end == "2024-12-31"
    assert metrics["free_cash_flow"].latest_annual.val == 70_000_000_000


def test_dedupe_by_end_collapses_comparative_year_restatements_in_history():
    """踩坑：同一个财年的数字会在后续1-2份年报里作为对比年份重复披露（比如FY2023
    的营收，会分别出现在FY2023、FY2024、FY2025三份10-K里——美股年报惯例披露近3年
    的对比利润表），取多年历史序列时如果不按 end 去重，会把同一年重复算出好几个点。"""
    facts_json = _facts_json(
        {
            "RevenueFromContractWithCustomerExcludingAssessedTax": [
                # FY2023：自己那年 + 后两年年报里当对比数据，一共出现3次
                _point("2023-09-30", "10-K", "2023-11-03", 383_285_000_000, start="2022-09-25", fy=2023, fp="FY"),
                _point("2023-09-30", "10-K", "2024-11-01", 383_285_000_000, start="2022-09-25", fy=2023, fp="FY"),
                _point("2023-09-30", "10-K", "2025-10-31", 383_285_000_000, start="2022-09-25", fy=2023, fp="FY"),
                # FY2024：自己那年 + 后一年年报里当对比数据
                _point("2024-09-28", "10-K", "2024-11-01", 391_035_000_000, start="2023-10-01", fy=2024, fp="FY"),
                _point("2024-09-28", "10-K", "2025-10-31", 391_035_000_000, start="2023-10-01", fy=2024, fp="FY"),
                # FY2025：只在自己那年年报里出现过
                _point("2025-09-27", "10-K", "2025-10-31", 416_161_000_000, start="2024-09-29", fy=2025, fp="FY"),
            ],
        }
    )

    history = extract_metrics_history(facts_json, years=10)

    ends = [p.end for p in history["revenue"].points]
    assert ends == ["2023-09-30", "2024-09-28", "2025-09-27"]  # 去重后一年一个点，不是6个


def test_dedupe_by_end_prefers_latest_filed_value_over_stale_restated_figure():
    """真实踩过的坑（NBIS/Nebius）：剥离主要业务后，公司把历史可比期间重述成了
    持续经营口径——同一个end日期在不同年份的年报里数值完全不同，不是单纯的重复
    披露。这时候必须取filed最晚的那条（公司当前认可的口径），取最早会拿到剥离
    前、公司自己已经不再使用的旧数字。"""
    facts_json = _facts_json(
        {
            "Revenues": [
                # 原始披露：含已剥离业务的合并口径
                _point("2022-12-31", "20-F", "2023-04-20", 7_417_100_000, start="2022-01-01", fy=2022, fp="FY"),
                # 剥离后重述：持续经营口径，同一个end日期数值完全不同
                _point("2022-12-31", "20-F", "2025-04-30", 13_500_000, start="2022-01-01", fy=2022, fp="FY"),
            ],
        }
    )

    metrics = extract_key_metrics(facts_json)

    assert metrics["revenue"].latest_annual.val == 13_500_000


def test_dedupe_by_end_prefers_amendment_over_original_filing():
    """20-F/A（更正版）明确修改了原始20-F的数值时，更正版按定义应该覆盖原始版本——
    取filed最晚的那条天然就是更正版（更正版申报时间必然晚于原始版本）。"""
    facts_json = _facts_json(
        {
            "Assets": [
                _point("2018-12-31", "20-F", "2019-04-19", 241_698_000_000, fy=2018, fp="FY"),
                _point("2018-12-31", "20-F/A", "2020-04-02", 259_097_000_000, fy=2018, fp="FY"),
            ],
        }
    )

    metrics = extract_key_metrics(facts_json)

    assert metrics["total_assets"].latest_annual.val == 259_097_000_000


def test_extract_metrics_history_orders_chronologically_and_computes_yoy():
    facts_json = _facts_json(
        {
            "NetIncomeLoss": [
                _point("2023-09-30", "10-K", "2023-11-03", 96_995_000_000, start="2022-09-25", fy=2023, fp="FY"),
                _point("2024-09-28", "10-K", "2024-11-01", 93_736_000_000, start="2023-10-01", fy=2024, fp="FY"),
                _point("2025-09-27", "10-K", "2025-10-31", 112_010_000_000, start="2024-09-29", fy=2025, fp="FY"),
            ],
        }
    )

    history = extract_metrics_history(facts_json, years=10)

    points = history["net_income"].points
    assert [p.end for p in points] == ["2023-09-30", "2024-09-28", "2025-09-27"]  # 正序，从早到晚
    assert points[0].yoy_change_pct is None  # 最早一期没有更早的数据算同比
    assert points[1].yoy_change_pct == pytest.approx(-3.36, abs=0.01)
    assert points[2].yoy_change_pct == pytest.approx(19.5, abs=0.01)


def test_extract_metrics_history_respects_years_limit():
    facts_json = _facts_json(
        {
            "Assets": [
                _point(f"{year}-12-31", "10-K", f"{year + 1}-02-01", float(year) * 1_000_000_000)
                for year in range(2015, 2026)
            ],
        }
    )

    history = extract_metrics_history(facts_json, years=5)

    assert len(history["total_assets"].points) == 5
    assert history["total_assets"].points[-1].end == "2025-12-31"  # 最近一年在最后（正序）


def test_extract_metrics_history_derives_free_cash_flow_across_years():
    facts_json = _facts_json(
        {
            "NetCashProvidedByUsedInOperatingActivities": [
                _point("2023-12-31", "10-K", "2024-02-01", 100_000_000_000, start="2023-01-01", fy=2023, fp="FY"),
                _point("2024-12-31", "10-K", "2025-02-01", 120_000_000_000, start="2024-01-01", fy=2024, fp="FY"),
            ],
            "PaymentsToAcquirePropertyPlantAndEquipment": [
                _point("2023-12-31", "10-K", "2024-02-01", 30_000_000_000, start="2023-01-01", fy=2023, fp="FY"),
                _point("2024-12-31", "10-K", "2025-02-01", 40_000_000_000, start="2024-01-01", fy=2024, fp="FY"),
            ],
        }
    )

    history = extract_metrics_history(facts_json, years=10)

    fcf_points = history["free_cash_flow"].points
    assert [p.end for p in fcf_points] == ["2023-12-31", "2024-12-31"]
    assert fcf_points[0].val == 70_000_000_000
    assert fcf_points[1].val == 80_000_000_000


def test_is_single_quarter_boundaries():
    assert _is_single_quarter({"start": "2026-03-29", "end": "2026-06-27"}) is True  # ~90天
    assert _is_single_quarter({"start": "2025-09-28", "end": "2026-06-27"}) is False  # ~272天，9个月累计
    assert _is_single_quarter({"end": "2026-06-27"}) is True  # 无 start，存量指标，不过滤


def test_is_full_year_boundaries():
    assert _is_full_year({"start": "2024-09-29", "end": "2025-09-27"}) is True  # ~363天
    assert _is_full_year({"start": "2026-03-29", "end": "2026-06-27"}) is False  # ~90天，单季度
    assert _is_full_year({"end": "2025-09-27"}) is True  # 无 start，存量指标，不过滤


# --- 境外私人发行人（20-F/20-F-A，NBIS这类没有10-K/10-Q的公司）支持


def test_20f_annual_filings_are_recognized_as_annual_points():
    """NBIS真实场景：只有20-F（年报，一年一次），没有10-K/10-Q——旧代码的_parse_points
    硬编码只认10-K/10-Q，会把这些点全部滤掉，导致get_financials对这类公司返回空数据。"""
    facts_json = _facts_json(
        {
            "Revenues": [
                _point("2024-12-31", "20-F", "2025-04-15", 500_000_000, start="2024-01-01", fy=2024, fp="FY"),
                _point("2023-12-31", "20-F", "2024-04-15", 300_000_000, start="2023-01-01", fy=2023, fp="FY"),
            ],
        }
    )

    metrics = extract_key_metrics(facts_json)

    assert metrics["revenue"].latest_annual.val == 500_000_000
    assert metrics["revenue"].latest_annual.yoy_change_pct == pytest.approx(66.67, abs=0.01)


def test_20f_a_amendment_is_also_recognized_as_annual():
    facts_json = _facts_json(
        {
            "Assets": [
                _point("2024-12-31", "20-F/A", "2025-05-01", 900_000_000, fy=2024, fp="FY"),
            ],
        }
    )

    metrics = extract_key_metrics(facts_json)

    assert metrics["total_assets"].latest_annual.val == 900_000_000


def test_prefers_usd_unit_over_other_currency_when_both_are_disclosed():
    """踩坑：某些有俄罗斯/独联体业务历史的境外发行人（如NBIS）会给同一个标签同时
    披露本币（如RUB）和USD两种单位——旧代码 next(iter(units)) 随手取字典第一个键，
    取到哪个纯粹看JSON里units对象的键顺序，可能拿到本币把营收算错几十倍。"""
    facts_json = {
        "facts": {
            "us-gaap": {
                "Revenues": {
                    "units": {
                        "RUB": [
                            _point(
                                "2024-12-31", "20-F", "2025-04-15", 45_000_000_000,
                                start="2024-01-01", fy=2024, fp="FY",
                            ),
                        ],
                        "USD": [
                            _point(
                                "2024-12-31", "20-F", "2025-04-15", 500_000_000,
                                start="2024-01-01", fy=2024, fp="FY",
                            ),
                        ],
                    }
                }
            }
        }
    }

    metrics = extract_key_metrics(facts_json)

    assert metrics["revenue"].unit == "USD"
    assert metrics["revenue"].latest_annual.val == 500_000_000


def test_falls_back_to_non_usd_unit_when_usd_not_disclosed():
    """纯本国披露、完全没有USD单位的公司，退回随便取一个单位——不能因为找不到USD
    就直接跳过这个标签，那样会让数据完全消失，比"单位是本币"更糟。"""
    facts_json = {
        "facts": {
            "us-gaap": {
                "Revenues": {
                    "units": {
                        "EUR": [
                            _point(
                                "2024-12-31", "20-F", "2025-04-15", 500_000_000,
                                start="2024-01-01", fy=2024, fp="FY",
                            ),
                        ],
                    }
                }
            }
        }
    }

    metrics = extract_key_metrics(facts_json)

    assert metrics["revenue"].unit == "EUR"
    assert metrics["revenue"].latest_annual.val == 500_000_000
