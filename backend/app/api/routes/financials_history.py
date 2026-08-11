from fastapi import APIRouter, HTTPException

from app.models.financials import FinancialsHistoryResponse
from app.services.sec_edgar import SecEdgarError, TickerNotFoundError, get_financials_history

router = APIRouter(prefix="/api/financials-history", tags=["financials-history"])


@router.get("/{ticker}", response_model=FinancialsHistoryResponse)
async def read_financials_history(ticker: str) -> FinancialsHistoryResponse:
    try:
        return await get_financials_history(ticker)
    except TickerNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except SecEdgarError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
