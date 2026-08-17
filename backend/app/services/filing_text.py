"""SEC EDGAR 财报原文文本获取（10-K/10-Q）。

只做：定位最新一期 10-K/10-Q → 拉取原始文档 → 剥离HTML标签。
不做任何"提取MD&A章节/风险披露章节"之类的语义切分——完整正文
原样交给 Agent，由它自己在阅读时判断哪部分有用（项目书第五节
架构原则3）。

境外私人发行人（Foreign Private Issuer）不提交10-K/10-Q，改提交20-F（年报）/
20-F/A（更正版）——请求"10-K"找不到时会透明兜底到这两种表格（FALLBACK_FORMS），
响应里的`form`字段如实反映实际拿到的类型，不会假装成10-K。"10-Q"没有境外发行人
的等价物，不做兜底。

注：项目书里写的"接入 Full-Text Search API"，实测那个接口
（efts.sec.gov/LATEST/search-index）返回的是关键词搜索命中的
元数据+摘要片段，并不包含完整正文；要拿到某公司最新一期10-K/10-Q
的完整正文，实际可行的路径是 submissions API（拿到最新文件的
accession number + 主文档文件名）+ EDGAR 文档原始地址，这是本文件
真正实现的方式。
"""

import json
import re
from datetime import datetime, timedelta
from pathlib import Path

import httpx
from bs4 import BeautifulSoup

from app.models.filing import FilingTextResponse
from app.services.cache_lock import get_lock
from app.services.sec_client import SecClientError, throttled_get
from app.services.sec_edgar import resolve_cik

SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
DOCUMENT_URL_TEMPLATE = "https://www.sec.gov/Archives/edgar/data/{cik_no_zeros}/{accession_no_dashes}/{primary_doc}"

CACHE_DIR = Path(__file__).resolve().parent.parent.parent / ".cache"
SUBMISSIONS_CACHE_TTL = timedelta(hours=6)  # 和 companyfacts 一样：新文件只在季报/年报时出现

SUPPORTED_FORMS = ("10-K", "10-Q")

# 境外私人发行人（Foreign Private Issuer，比如NBIS这类公司）不提交10-K，改提交
# 20-F（年报，一年一次）；20-F/A是更正版。10-Q没有等价的境外发行人季度报表义务，
# 所以这里没有它的兜底项——找不到10-Q就是真的找不到，不该假装有替代品。
# Agent仍然只能请求"10-K"/"10-Q"（SUPPORTED_FORMS不变），这个兜底是内部实现细节，
# 对Agent透明；但实际拿到的form可能不是它请求的那个，响应里会如实反映。
FALLBACK_FORMS: dict[str, tuple[str, ...]] = {
    "10-K": ("20-F", "20-F/A"),
}


class FilingTextError(SecClientError):
    pass


class FilingNotFoundError(FilingTextError):
    pass


def _submissions_cache_file(cik: str) -> Path:
    return CACHE_DIR / f"submissions_{cik}.json"


async def fetch_submissions(cik: str, client: httpx.AsyncClient) -> dict:
    cache_file = _submissions_cache_file(cik)
    async with get_lock(f"submissions_{cik}"):
        if cache_file.exists():
            age = datetime.now() - datetime.fromtimestamp(cache_file.stat().st_mtime)
            if age < SUBMISSIONS_CACHE_TTL:
                return json.loads(cache_file.read_text())

        url = SUBMISSIONS_URL.format(cik=int(cik))
        try:
            response = await throttled_get(client, url)
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise FilingTextError(f"拉取 submissions 失败：{exc}") from exc
        data = response.json()

        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(json.dumps(data))

        return data


def find_latest_filing(submissions_json: dict, forms: str | tuple[str, ...]) -> dict:
    """在 submissions 的 `recent` 区块里找最新一期指定表格类型的申报。
    只看 `recent`（SEC文档说明里覆盖最近约1000条申报），不翻更早的分页文件——
    对于按季/按年规律申报的公司，最新一期必然落在这个窗口内。

    `forms` 可以传多个类型（比如10-K兜底到20-F/20-F/A的场景）——`recent` 数组
    本身就是按申报时间从新到旧排列的，"第一个匹配任意一种类型的记录"天然就是
    这几种类型里最新的那条，不需要额外按日期比较，也不需要人为规定"先试哪个"
    的优先级（比如20-F/A更正版有可能比原始20-F更晚申报，靠数组顺序自然处理）。
    返回值里带上实际匹配到的 `form`，调用方可能请求的是"10-K"、实际拿到的是
    "20-F"，不能让返回值假装这两者是同一件事。"""
    if isinstance(forms, str):
        forms = (forms,)
    recent = submissions_json.get("filings", {}).get("recent", {})
    all_forms = recent.get("form", [])

    for i, f in enumerate(all_forms):
        if f in forms:
            return {
                "accession_number": recent["accessionNumber"][i],
                "primary_document": recent["primaryDocument"][i],
                "filing_date": recent["filingDate"][i],
                "report_date": recent["reportDate"][i],
                "form": f,
            }

    label = forms[0] if len(forms) == 1 else "/".join(forms)
    raise FilingNotFoundError(f"在最近的申报记录里找不到 {label} 文件")


def _document_cache_file(accession_number: str, primary_document: str) -> Path:
    safe_name = f"{accession_number}_{primary_document}".replace("/", "_")
    return CACHE_DIR / f"filingdoc_{safe_name}"


async def fetch_filing_document(
    cik: str, accession_number: str, primary_document: str, client: httpx.AsyncClient
) -> str:
    """已经正式提交的文件内容不会再变，缓存不设过期时间，命中就直接用。"""
    cache_file = _document_cache_file(accession_number, primary_document)
    async with get_lock(f"filingdoc_{accession_number}_{primary_document}"):
        if cache_file.exists():
            return cache_file.read_text()

        accession_no_dashes = accession_number.replace("-", "")
        url = DOCUMENT_URL_TEMPLATE.format(
            cik_no_zeros=int(cik), accession_no_dashes=accession_no_dashes, primary_doc=primary_document
        )
        try:
            response = await throttled_get(client, url)
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise FilingTextError(f"拉取财报原文失败：{exc}") from exc
        html = response.text

        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(html)

        return html


def strip_html_to_text(html: str) -> str:
    """只做最基础的格式处理：剥离标签、合并多余空行。不做任何语义层面的
    章节提取/内容过滤——保留下来的是完整正文，交给 Agent 自己筛选。

    唯一的例外是去掉 iXBRL 的隐藏元数据（<ix:header> 整块、以及带
    display:none 的元素）：这些不是人类阅读财报时会看到的正文，只是
    机器可读的 XBRL 标签数据在 HTML 里的技术性存放位置，留着只会把
    大量标签命名空间字符串混进正文最前面，跟"筛选内容相关性"无关。"""
    soup = BeautifulSoup(html, "html.parser")

    header = soup.find("ix:header")
    if header is not None:
        header.decompose()
    for hidden in soup.find_all(style=lambda v: v and "display:none" in v.replace(" ", "")):
        hidden.decompose()

    text = soup.get_text(separator="\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


async def get_filing_text(ticker: str, form: str = "10-K") -> FilingTextResponse:
    if form not in SUPPORTED_FORMS:
        raise FilingTextError(f"不支持的表格类型 '{form}'，目前只支持 {SUPPORTED_FORMS}")

    candidate_forms = (form, *FALLBACK_FORMS.get(form, ()))

    async with httpx.AsyncClient(timeout=30.0) as client:
        cik, entity_name = await resolve_cik(ticker, client)
        submissions = await fetch_submissions(cik, client)
        filing = find_latest_filing(submissions, candidate_forms)
        html = await fetch_filing_document(
            cik, filing["accession_number"], filing["primary_document"], client
        )

    text = strip_html_to_text(html)
    accession_no_dashes = filing["accession_number"].replace("-", "")
    source_url = DOCUMENT_URL_TEMPLATE.format(
        cik_no_zeros=int(cik),
        accession_no_dashes=accession_no_dashes,
        primary_doc=filing["primary_document"],
    )

    return FilingTextResponse(
        ticker=ticker.upper(),
        cik=cik,
        entity_name=entity_name,
        form=filing["form"],  # 实际拿到的表格类型；请求10-K的境外发行人可能被兜底成20-F
        filing_date=filing["filing_date"],
        report_date=filing["report_date"],
        accession_number=filing["accession_number"],
        source_url=source_url,
        text=text,
        text_length=len(text),
    )
