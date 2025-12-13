from fastapi import APIRouter
from app.api.v1.endpoints import valuation, portfolios, market, signals, quant, quant_dashboard, data, factors

api_router = APIRouter()
api_router.include_router(valuation.router, prefix="/valuation", tags=["valuation"])
api_router.include_router(valuation.router, prefix="/fundamental", tags=["fundamental"])
api_router.include_router(portfolios.router, tags=["portfolios"])
api_router.include_router(market.router, prefix="/market", tags=["market"])
api_router.include_router(signals.router, prefix="/signals", tags=["signals"])
api_router.include_router(quant.router, prefix="/quant", tags=["quant"])
api_router.include_router(quant_dashboard.router, prefix="/quant", tags=["quant_dashboard"])
api_router.include_router(data.router, prefix="/data", tags=["data"])
api_router.include_router(factors.router, tags=["factors"])

