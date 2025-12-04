from dagster import asset, Output, AssetExecutionContext
from app.core.database import SessionLocal
from quant.data.ingestion import IngestionService
import logging

logger = logging.getLogger("dagster")

@asset(group_name="ingestion")
def market_data_daily(context: AssetExecutionContext):
    """
    Fetch and store daily market data for the S&P 500 universe.
    """
    db = SessionLocal()
    try:
        service = IngestionService(db)
        tickers = service.fetch_sp500_tickers()
        context.log.info(f"Fetched {len(tickers)} tickers from service.")
        
        context.log.info(f"Ingesting data for {len(tickers)} tickers...")
        service.ingest_daily_data(tickers, lookback_days=365*2)
        context.log.info("Ingestion complete.")
    finally:
        db.close()

@asset(group_name="ingestion")
def fundamentals_data(context: AssetExecutionContext):
    """
    Fetch and store fundamental data for the S&P 500 universe.
    """
    db = SessionLocal()
    try:
        service = IngestionService(db)
        tickers = service.fetch_sp500_tickers()
        context.log.info(f"Ingesting fundamentals for {len(tickers)} tickers...")
        service.ingest_fundamentals(tickers)
        context.log.info("Fundamentals ingestion complete.")
    finally:
        db.close()

@asset(group_name="factors", deps=[market_data_daily, fundamentals_data])
def factor_scores(context: AssetExecutionContext):
    """
    Calculate factor scores (Momentum, Value) for all securities.
    """
    from quant.factors.engine import FactorEngine
    from quant.data.models import Security
    from datetime import date
    
    db = SessionLocal()
    try:
        engine = FactorEngine(db)
        securities = db.query(Security).filter(Security.active == True).all()
        today = date.today()
        
        context.log.info(f"Calculating factors for {len(securities)} securities...")
        
        for sec in securities:
            # Momentum
            mom = engine.calculate_momentum(sec.sid, today)
            # Value
            val = engine.calculate_value_factors(sec.sid, today)
            
            if mom is not None:
                context.log.debug(f"{sec.ticker} Momentum: {mom:.4f}")
            if val:
                context.log.debug(f"{sec.ticker} Value: {val}")
                
        context.log.info("Factor calculation complete.")
    finally:
        db.close()

@asset(group_name="valuation", deps=[market_data_daily, fundamentals_data])
def valuation_scores(context: AssetExecutionContext):
    """
    Calculate DCF valuation for all securities.
    """
    from quant.valuation.engine import ValuationEngine
    from quant.data.models import Security
    from datetime import date
    
    db = SessionLocal()
    try:
        engine = ValuationEngine(db)
        securities = db.query(Security).filter(Security.active == True).all()
        today = date.today()
        
        context.log.info(f"Running valuation for {len(securities)} securities...")
        
        for sec in securities:
            res = engine.calculate_dcf(sec.sid, today)
            if res:
                context.log.debug(f"{sec.ticker} Fair Value: ${res['fair_value']:.2f} (Upside: {res['upside']:.1%})")
                
        context.log.info("Valuation complete.")
    finally:
        db.close()

@asset(group_name="selection", deps=[valuation_scores])
def stock_ranking(context: AssetExecutionContext):
    """
    Rank stocks based on composite score (Momentum + Value + Upside).
    """
    from quant.selection.ranking import RankingEngine
    from datetime import date
    
    db = SessionLocal()
    try:
        engine = RankingEngine(db)
        today = date.today()
        
        context.log.info("Running stock ranking...")
        top_picks = engine.run_ranking(today)
        
        if top_picks is not None and not top_picks.empty:
            context.log.info("Top 10 Picks:")
            for i, row in top_picks.head(10).iterrows():
                context.log.info(f"#{int(row['rank'])} {row['ticker']} (Score: {row['score']:.2f}) - Mom: {row['momentum']:.1%}, Yield: {row['earnings_yield']:.1%}, Upside: {row['upside']:.1%}")
        else:
            context.log.warning("No ranking generated.")
            
    finally:
        db.close()

@asset(group_name="portfolio", deps=[stock_ranking])
def portfolio_optimization(context: AssetExecutionContext):
    """
    Optimize portfolio weights using MVO.
    """
    from quant.portfolio.optimizer import PortfolioOptimizer
    from datetime import date
    
    db = SessionLocal()
    try:
        optimizer = PortfolioOptimizer(db)
        today = date.today()
        
        context.log.info("Running portfolio optimization...")
        allocations = optimizer.run_optimization(today)
        
        if allocations:
            context.log.info("Optimization Complete. Target Portfolio:")
            for ticker, weight in allocations:
                if weight > 0.01:
                    context.log.info(f"{ticker}: {weight:.1%}")
    finally:
        db.close()
