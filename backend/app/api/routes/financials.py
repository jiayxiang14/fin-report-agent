from fastapi import APIRouter, HTTPException

from app.api.ticker_path import TickerPath
from app.models.financials import FinancialsResponse
from app.services.sec_edgar import SecEdgarError, TickerNotFoundError, get_financials

router = APIRouter(prefix="/api/financials", tags=["financials"])


@router.get("/{ticker}", response_model=FinancialsResponse)
async def read_financials(ticker: TickerPath) -> FinancialsResponse:
    try:
        return await get_financials(ticker)
    except TickerNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except SecEdgarError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
