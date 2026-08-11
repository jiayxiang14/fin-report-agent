from fastapi import APIRouter

from app.models.peer import PeerComparisonResponse
from app.services.peer_comparison import get_peer_comparison

router = APIRouter(prefix="/api/peer-comparison", tags=["peer"])


@router.get("/{ticker}", response_model=PeerComparisonResponse)
async def read_peer_comparison(ticker: str) -> PeerComparisonResponse:
    return await get_peer_comparison(ticker)
