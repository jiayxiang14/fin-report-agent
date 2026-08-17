from pydantic import BaseModel


class FilingTextResponse(BaseModel):
    ticker: str
    cik: str
    entity_name: str
    form: str  # 实际拿到的表格类型："10-K"/"10-Q"，境外发行人可能被兜底成"20-F"/"20-F/A"
    filing_date: str
    report_date: str  # 财报覆盖期间的期末日期
    accession_number: str
    source_url: str
    text: str  # 剥离HTML标签后的原始正文，不做语义清洗
    text_length: int
