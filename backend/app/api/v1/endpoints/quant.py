"""
Quant API Endpoints

Stock rankings, portfolio targets, and backtest endpoints.
Now powered by Parquet/DuckDB for high-performance queries.
"""

from fastapi import APIRouter, HTTPException, Query
from quant.data.signal_store import get_signal_store
from quant.data.data_provider import create_data_provider
from quant.backtest.engine import BacktestEngine
from datetime import date, timedelta
import logging
import json

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("/rankings")
def get_rankings(
    model: str = Query(default=None, description="Model name (ranking_v3, ranking_v2)"),
    limit: int = Query(default=100, le=500, description="Max results")
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
    end_year: int = Query(default=None, description="End year")
):
    """
    Run a backtest simulation.
    
    Uses Parquet/DuckDB DataProvider for fast price lookups.
    """
    try:
        # Create data provider (uses Parquet)
        provider = create_data_provider(provider_type='parquet')
        
        # Create backtest engine with data provider
        engine = BacktestEngine(data_provider=provider)
        
        # Calculate date range
        if end_year:
            end_date = date(end_year, 12, 31)
        else:
            end_date = date.today()
            
        if start_year:
            start_date = date(start_year, 1, 1)
        else:
            start_date = end_date - timedelta(days=365)
        
        results = engine.run_backtest(start_date, end_date)
        
        # Convert DataFrame to dict if needed
        if hasattr(results, 'to_dict'):
            return results.to_dict('records')
        return results
        
    except Exception as e:
        logger.error(f"Backtest failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
