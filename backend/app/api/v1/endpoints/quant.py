"""
Quant API Endpoints

Stock rankings, portfolio targets, and backtest endpoints.
Now powered by Parquet/DuckDB for high-performance queries.
"""

from fastapi import APIRouter, HTTPException, Query
from quant.data.signal_store import get_signal_store
from quant.data.data_provider import create_data_provider
from quant.backtest.engine import BacktestEngine
from app.core.startup import startup_service
from datetime import date, timedelta
import logging
import json

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("/data-status")
def get_data_status():
    """
    Get data lake status and catch-up result.
    
    Returns information about:
    - Last catch-up operation result
    - Data lake health
    """
    try:
        from quant.data.integrity import SmartCatchUpService, MarketCalendar
        from quant.data.parquet_io import ParquetReader, get_data_lake_path
        
        # Get catch-up result from startup
        catchup_result = startup_service.get_catchup_result()
        
        # Get current data status
        reader = ParquetReader(str(get_data_lake_path()))
        df = reader.read_prices(columns=['date', 'ticker'])
        
        if df.empty:
            data_status = {
                "has_data": False,
                "last_date": None,
                "ticker_count": 0
            }
        else:
            data_status = {
                "has_data": True,
                "last_date": str(df['date'].max()),
                "first_date": str(df['date'].min()),
                "ticker_count": df['ticker'].nunique(),
                "total_rows": len(df)
            }
        
        return {
            "data_status": data_status,
            "catchup_result": catchup_result
        }
        
    except Exception as e:
        logger.error(f"Error getting data status: {e}")
        return {
            "data_status": {"error": str(e)},
            "catchup_result": startup_service.get_catchup_result()
        }


@router.get("/rankings")
def get_rankings(
    model: str = Query(default=None, description="Model name (ranking_v3, ranking_v2)"),
    limit: int = Query(default=100, le=2000, description="Max results")
):
    """
    Get latest stock rankings from RankingEngine.
    
    Now uses Parquet/DuckDB backend for 35x faster queries.
    """
    try:
        store = get_signal_store()
        
        # Try v3 first, fallback to v2
        for model_version in ['ranking_v3', 'ranking_v2', 'ranking_v1']:
            if model and model != model_version:
                continue
                
            df = store.get_latest_signals(model_name=model_version, limit=limit)
            
            if not df.empty:
                logger.info(f"Found {len(df)} signals from {model_version}")
                break
        else:
            logger.warning("No ranking signals found")
            return []
        
        # Convert to API response format
        results = []
        for _, row in df.iterrows():
            # Parse metadata
            metadata = {}
            if 'metadata_json' in row and row['metadata_json']:
                try:
                    metadata = json.loads(row['metadata_json']) if isinstance(row['metadata_json'], str) else row['metadata_json']
                except:
                    pass
            
            results.append({
                "rank": int(row['rank']) if 'rank' in row else None,
                "ticker": row['ticker'],
                "score": float(row['score']) if 'score' in row else None,
                "date": str(row['date']) if 'date' in row else None,
                "model": row.get('model_name', model_version),
                "regime": metadata.get('regime', 'Unknown'),
                "pead": metadata.get('pead', 0),
                "sentiment": metadata.get('sentiment', 0),
                "vsm": metadata.get('vsm', 0),
                "bab": metadata.get('bab', 0),
                "qmj": metadata.get('qmj', 0),
                "upside": metadata.get('upside', 0),
                "metadata": row.get('metadata_json')
            })
        
        return results
        
    except Exception as e:
        logger.error(f"Error fetching rankings: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/portfolio")
def get_portfolio(
    model: str = Query(default="mvo_sharpe", description="Optimizer model name")
):
    """
    Get latest portfolio targets.
    
    Now uses Parquet/DuckDB backend.
    """
    try:
        store = get_signal_store()
        df = store.get_latest_targets(model_name=model)
        
        if df.empty:
            logger.warning(f"No portfolio targets found for {model}")
            return []
        
        return [
            {
                "ticker": row['ticker'],
                "weight": float(row['weight']),
                "date": str(row['date']) if 'date' in row else None
            }
            for _, row in df.iterrows()
        ]
        
    except Exception as e:
        logger.error(f"Error fetching portfolio: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/portfolios")
def get_all_portfolios():
    """
    Get all available portfolio models and their latest targets.
    """
    try:
        store = get_signal_store()
        
        # Get all available target dates
        dates = store.list_available_dates('targets')
        
        if not dates:
            return {"models": [], "latest_date": None}
        
        latest_date = dates[0]
        
        # Get targets for different models
        models = {}
        for model_name in ['mvo_sharpe', 'hrp_v1', 'kelly_v1', 'mvo_v1']:
            df = store.get_latest_targets(model_name=model_name)
            if not df.empty:
                models[model_name] = [
                    {"ticker": row['ticker'], "weight": float(row['weight'])}
                    for _, row in df.iterrows()
                ]
        
        return {
            "models": list(models.keys()),
            "latest_date": str(latest_date),
            "portfolios": models
        }
        
    except Exception as e:
        logger.error(f"Error fetching portfolios: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/backtest")
def run_backtest(
    start_year: int = Query(default=None, description="Start year"),
    end_year: int = Query(default=None, description="End year"),
    top_n: int = Query(default=50, description="Number of stocks to hold")
):
    """
    Run a walk-forward factor backtest simulation.
    
    Uses Parquet/DuckDB for fast data access and PointInTimeFactorEngine
    for signal generation (VSM, BAB, Momentum).
    """
    try:
        from quant.backtest.factor_engine import run_factor_backtest, calculate_performance_metrics
        import pandas as pd
        
        # 1. Setup Dates
        if end_year:
            end_date = date(end_year, 12, 31)
        else:
            end_date = date.today()
            
        if start_year:
            start_date = date(start_year, 1, 1)
        else:
            # Default to 2 years
            start_date = date(end_date.year - 2, 1, 1)
            
        # Need extra data for factor lookup (approx 1 year)
        data_start = date(start_date.year - 1, 1, 1)
        
        logger.info(f"Starting backtest {start_date} to {end_date} (Data from {data_start})")
        
        # 2. Fetch Data
        provider = create_data_provider(provider_type='parquet')
        
        # Get universe first (optimization: can we cache this?)
        # For now, get all tickers that existed at end_date
        tickers = provider.get_universe(as_of_date=end_date)
        
        if not tickers:
             raise HTTPException(status_code=404, detail="No data available in data lake")
             
        # Fetch prices
        # We need 'close' for the engine
        logger.info(f"Fetching prices for {len(tickers)} tickers...")
        prices = provider.get_prices(
            tickers=tickers,
            start_date=data_start,
            end_date=end_date,
            fields=['close']  # factor_engine expects 'close'
        )
        
        if prices.empty:
            raise HTTPException(status_code=404, detail="No price data found for selected period")
            
        # 3. Generate Rebalance Dates (Monthly)
        # pd.date_range returns DatetimeIndex, convert to list of dates or timestamps
        # run_factor_backtest expects list of pd.Timestamp or date
        rebalance_dates = pd.date_range(start=start_date, end=end_date, freq='ME')
        
        if len(rebalance_dates) == 0:
             raise HTTPException(status_code=400, detail="Period too short for monthly rebalancing")

        # 4. Run Backtest
        logger.info("Running factor backtest...")
        equity_curve, trade_logs = run_factor_backtest(
            prices_df=prices,
            rebalance_dates=rebalance_dates,
            top_n=top_n
        )
        
        if equity_curve.empty:
            return {
                "metrics": {},
                "equity_curve": [],
                "trades": []
            }
            
        # 5. Calculate Metrics
        metrics = calculate_performance_metrics(equity_curve)
        
        return {
            "metrics": metrics,
            "equity_curve": equity_curve.to_dict(orient='records'),
            "trades": trade_logs
        }
        
    except Exception as e:
        logger.error(f"Backtest failed: {e}")
        # Print stack trace for debugging
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
