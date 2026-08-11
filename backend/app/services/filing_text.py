"""SEC EDGAR 财报原文文本获取（10-K/10-Q）。

只做：定位最新一期 10-K/10-Q → 拉取原始文档 → 剥离HTML标签。
不做任何"提取MD&A章节/风险披露章节"之类的语义切分——完整正文
原样交给 Agent，由它自己在阅读时判断哪部分有用（项目书第五节
架构原则3）。

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


def find_latest_filing(submissions_json: dict, form: str) -> dict:
    """在 submissions 的 `recent` 区块里找最新一期指定表格类型的申报。
    只看 `recent`（SEC文档说明里覆盖最近约1000条申报），不翻更早的分页文件——
    对于按季/按年规律申报10-K/10-Q的公司，最新一期必然落在这个窗口内。"""
    recent = submissions_json.get("filings", {}).get("recent", {})
    forms = recent.get("form", [])

    for i, f in enumerate(forms):
        if f == form:
            return {
                "accession_number": recent["accessionNumber"][i],
                "primary_document": recent["primaryDocument"][i],
                "filing_date": recent["filingDate"][i],
                "report_date": recent["reportDate"][i],
            }

    raise FilingNotFoundError(f"在最近的申报记录里找不到 {form} 文件")


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

    async with httpx.AsyncClient(timeout=30.0) as client:
        cik, entity_name = await resolve_cik(ticker, client)
        submissions = await fetch_submissions(cik, client)
        filing = find_latest_filing(submissions, form)
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
        form=form,
        filing_date=filing["filing_date"],
        report_date=filing["report_date"],
        accession_number=filing["accession_number"],
        source_url=source_url,
        text=text,
        text_length=len(text),
    )
