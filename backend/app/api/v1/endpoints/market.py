
from fastapi import APIRouter, HTTPException
from app.engines.market import bubble
from app.domain import schemas
import logging

router = APIRouter()
logger = logging.getLogger(__name__)

@router.get("/risk", response_model=schemas.MarketRiskResponse)
async def get_market_risk():
    """
    Get current market risk / bubble indicators.
    """
    try:
        data = bubble.get_market_risk_data()
        return data
    except Exception as e:
        logger.error(f"Error fetching market risk: {e}")
        raise HTTPException(status_code=500, detail=str(e))
