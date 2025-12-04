from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List
from app.core import database
from app.domain import schemas
from app.services import portfolio_service
from app.data.providers import yfinance_provider
from app.engines.backtest import engine as backtest_engine

router = APIRouter()

@router.get("/portfolios", response_model=List[schemas.Portfolio])
def read_portfolios(db: Session = Depends(database.get_db)):
    return portfolio_service.get_portfolios(db)

@router.post("/portfolios", response_model=schemas.Portfolio)
def create_portfolio(name: str, db: Session = Depends(database.get_db)):
    return portfolio_service.create_portfolio(db, name)

@router.get("/config/{portfolio_id}", response_model=schemas.PortfolioConfig)
def read_portfolio_config(portfolio_id: int, db: Session = Depends(database.get_db)):
    return portfolio_service.get_portfolio_config(db, portfolio_id)

@router.post("/config/{portfolio_id}")
def update_portfolio_config(portfolio_id: int, config: schemas.PortfolioConfig, db: Session = Depends(database.get_db)):
    success = portfolio_service.save_portfolio_config(db, portfolio_id, config)
    if not success:
        raise HTTPException(status_code=404, detail="Portfolio not found")
    return {"status": "success"}

@router.get("/portfolio/{portfolio_id}", response_model=schemas.PortfolioData)
def read_portfolio_data(portfolio_id: int, db: Session = Depends(database.get_db)):
    return portfolio_service.get_portfolio_data(db, portfolio_id)

@router.get("/search", response_model=List[schemas.SearchResult])
def search_assets(q: str):
    return yfinance_provider.search_assets(q)

@router.get("/asset-details/{ticker}", response_model=schemas.AssetDetails)
def read_asset_details(ticker: str):
    return yfinance_provider.get_asset_details(ticker)

@router.post("/analyze", response_model=schemas.AnalysisResponse)
def analyze_portfolio(request: schemas.AnalysisRequest):
    return backtest_engine.analyze_portfolio(request)

@router.post("/backtest", response_model=schemas.BacktestResponse)
def backtest_portfolio(request: schemas.BacktestRequest):
    # Map request to service arguments
    return backtest_engine.get_backtest_data(request.allocation, request.initialAmount, request.monthlyAmount)
