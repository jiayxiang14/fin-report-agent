"""境外私人发行人（Foreign Private Issuer）没有10-K/10-Q，改提交20-F/20-F-A——
`get_filing_text` 请求"10-K"时应该透明兜底到这两种表格，不该因为公司类型就
完全拿不到财报原文。核心逻辑是 `find_latest_filing` 在多个候选表格类型里选出
`recent` 数组中最先出现（即最新申报）的那条，不预设"先试哪个类型"的优先级。
"""

import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from app.services.filing_text import FilingNotFoundError, find_latest_filing, get_filing_text


def _submissions(forms: list[str], filing_dates: list[str]) -> dict:
    n = len(forms)
    return {
        "filings": {
            "recent": {
                "form": forms,
                "accessionNumber": [f"0001-{i:02d}" for i in range(n)],
                "primaryDocument": [f"doc{i}.htm" for i in range(n)],
                "filingDate": filing_dates,
                "reportDate": filing_dates,
            }
        }
    }


def test_finds_single_requested_form_when_present():
    submissions = _submissions(["10-K", "10-Q"], ["2026-02-01", "2025-11-01"])

    result = find_latest_filing(submissions, "10-K")

    assert result["form"] == "10-K"
    assert result["accession_number"] == "0001-00"


def test_raises_not_found_when_form_absent():
    submissions = _submissions(["10-Q"], ["2025-11-01"])

    with pytest.raises(FilingNotFoundError):
        find_latest_filing(submissions, "10-K")


def test_multi_form_tuple_returns_first_array_match_regardless_of_which_form():
    """境外发行人没有10-K，但有20-F——数组本身按申报时间从新到旧排列，
    传入("10-K", "20-F", "20-F/A")这个候选组合时，第一个匹配到的就是最新的，
    不用额外比较日期。"""
    submissions = _submissions(
        ["8-K", "20-F", "6-K"],
        ["2026-05-01", "2026-04-15", "2026-01-10"],
    )

    result = find_latest_filing(submissions, ("10-K", "20-F", "20-F/A"))

    assert result["form"] == "20-F"
    assert result["accession_number"] == "0001-01"


def test_20f_a_amendment_wins_when_filed_after_original_20f():
    """20-F/A（更正版）比原始20-F更晚申报时，应该拿到更正版——不能因为"先试20-F"
    这种固定优先级而漏掉更晚出现的更正版，这里靠数组顺序（更正版排在更前面，
    因为申报更晚）自然处理，不需要额外的"20-F/A优先"特判。"""
    submissions = _submissions(
        ["20-F/A", "20-F"],
        ["2026-05-01", "2026-04-15"],
    )

    result = find_latest_filing(submissions, ("10-K", "20-F", "20-F/A"))

    assert result["form"] == "20-F/A"


def test_raises_not_found_when_none_of_the_candidate_forms_present():
    submissions = _submissions(["8-K"], ["2026-05-01"])

    with pytest.raises(FilingNotFoundError):
        find_latest_filing(submissions, ("10-K", "20-F", "20-F/A"))


def test_get_filing_text_falls_back_to_20f_and_reports_actual_form_retrieved():
    """端到端：请求"10-K"，但这家公司（比如NBIS）的submissions里只有20-F——
    响应里的form字段必须如实反映实际拿到的是20-F，不能保留请求时的"10-K"，
    否则Agent会误以为自己真的读到了年度10-K。"""
    submissions = _submissions(["20-F", "6-K"], ["2026-04-15", "2026-01-10"])

    with (
        patch(
            "app.services.filing_text.resolve_cik",
            new=AsyncMock(return_value=("0001513845", "Nebius Group N.V.")),
        ),
        patch(
            "app.services.filing_text.fetch_submissions",
            new=AsyncMock(return_value=submissions),
        ),
        patch(
            "app.services.filing_text.fetch_filing_document",
            new=AsyncMock(return_value="<html><body>正文</body></html>"),
        ),
    ):
        result = asyncio.run(get_filing_text("NBIS", "10-K"))

    assert result.form == "20-F"
    assert result.accession_number == "0001-00"


def test_get_filing_text_10q_has_no_fallback_and_raises_when_absent():
    """10-Q没有境外发行人的等价替代表格，找不到就应该老实报错，不能悄悄兜底成
    一个语义不对等的表格类型（20-F是年报，不能冒充季报）。"""
    submissions = _submissions(["20-F"], ["2026-04-15"])

    with (
        patch(
            "app.services.filing_text.resolve_cik",
            new=AsyncMock(return_value=("0001513845", "Nebius Group N.V.")),
        ),
        patch(
            "app.services.filing_text.fetch_submissions",
            new=AsyncMock(return_value=submissions),
        ),
    ):
        with pytest.raises(FilingNotFoundError):
            asyncio.run(get_filing_text("NBIS", "10-Q"))
