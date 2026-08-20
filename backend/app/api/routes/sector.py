from fastapi import APIRouter, HTTPException

from app.api.ticker_path import TickerPath
from app.models.sector import SectorRotationResponse
from app.services.sector_rotation import SectorDataError, get_sector_rotation

router = APIRouter(prefix="/api/sector-position", tags=["sector"])


@router.get("/{ticker}", response_model=SectorRotationResponse)
async def read_sector_position(ticker: TickerPath) -> SectorRotationResponse:
    try:
        return await get_sector_rotation(ticker)
    except SectorDataError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
