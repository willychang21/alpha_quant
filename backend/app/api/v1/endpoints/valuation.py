from fastapi import APIRouter, HTTPException, BackgroundTasks
from app.services import valuation_service
from app.domain import schemas
import logging

router = APIRouter()
logger = logging.getLogger(__name__)

@router.post("/analyze", response_model=schemas.ValuationResult)
async def analyze_stock(request: schemas.AnalyzeRequest):
    """
    Analyze a single stock ticker.
    """
    try:
        result = await valuation_service.analyze_stock(request.ticker)
        return result
    except Exception as e:
        logger.error(f"Error analyzing {request.ticker}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/analyze-portfolio", response_model=schemas.AnalysisResponse)
async def analyze_portfolio(request: schemas.AnalysisRequest):
    """
    Analyze a portfolio of stocks (Projection Matrix).
    """
    try:
        # analyze_portfolio is still sync in service, but we can wrap it or leave as is if it's CPU bound
        # For now, we'll keep it as is, but if we made it async in service we'd await it.
        # The service definition I wrote earlier kept it sync.
        # Let's check if I made it async in service... I did NOT make analyze_portfolio async in service.
        # So no await here.
        results = valuation_service.analyze_portfolio(request)
        return results
    except Exception as e:
        logger.error(f"Error analyzing portfolio: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{ticker}", response_model=schemas.ValuationResult)
async def get_valuation(ticker: str):
    """
    Get valuation for a ticker (GET request).
    """
    try:
        return await valuation_service.analyze_stock(ticker)
    except Exception as e:
        logger.error(f"Error getting valuation for {ticker}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/fundamental/{ticker}", response_model=schemas.ValuationResult)
async def get_fundamental_data(ticker: str):
    """
    Get fundamental data for a ticker.
    """
    try:
        # For now, we return the same full valuation result which contains fundamental data
        return await valuation_service.analyze_stock(ticker)
    except Exception as e:
        logger.error(f"Error getting fundamental data for {ticker}: {e}")
        raise HTTPException(status_code=500, detail=str(e))
