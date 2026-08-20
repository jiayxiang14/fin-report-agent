from fastapi import APIRouter, HTTPException, Query

from app.api.ticker_path import TickerPath
from app.models.filing import FilingTextResponse
from app.services.filing_text import get_filing_text
from app.services.sec_client import SecClientError
from app.services.sec_edgar import TickerNotFoundError

router = APIRouter(prefix="/api/filing-text", tags=["filing"])


@router.get("/{ticker}", response_model=FilingTextResponse)
async def read_filing_text(
    ticker: TickerPath, form: str = Query(default="10-K", pattern="^(10-K|10-Q)$")
) -> FilingTextResponse:
    try:
        return await get_filing_text(ticker, form)
    except TickerNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except SecClientError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
