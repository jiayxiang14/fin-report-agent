from fastapi import APIRouter, HTTPException

from app.api.ticker_path import TickerPath
from app.models.company_profile import CompanyProfileResponse
from app.services.company_profile import CompanyProfileError, get_company_profile

router = APIRouter(prefix="/api/company-profile", tags=["company-profile"])


@router.get("/{ticker}", response_model=CompanyProfileResponse)
async def read_company_profile(ticker: TickerPath) -> CompanyProfileResponse:
    try:
        return await get_company_profile(ticker)
    except CompanyProfileError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
