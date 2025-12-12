"""
Walk-Forward Backtest Runner
Runs a proper point-in-time backtest using historical price data.
"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np
from datetime import datetime
import logging

from app.core.database import SessionLocal
from quant.data.models import Security, MarketDataDaily
from quant.backtest.factor_engine import (
    PointInTimeFactorEngine, 
    run_factor_backtest,
    calculate_performance_metrics
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_price_data(db, start_date: str = '2020-01-01', end_date: str = None) -> pd.DataFrame:
    """Load price data from database."""
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')
    
    query = db.query(
        MarketDataDaily.date,
        Security.ticker,
        MarketDataDaily.close
    ).join(Security, Security.sid == MarketDataDaily.sid)\
     .filter(MarketDataDaily.date >= start_date)\
     .filter(MarketDataDaily.date <= end_date)\
     .filter(MarketDataDaily.close.isnot(None))
    
    data = query.all()
    
    if not data:
        return pd.DataFrame()
    
    df = pd.DataFrame(data, columns=['date', 'ticker', 'close'])
    df['date'] = pd.to_datetime(df['date'])
    
    return df


def generate_monthly_rebalance_dates(start_date: str, end_date: str) -> list:
    """Generate month-end rebalancing dates."""
    dates = pd.date_range(start=start_date, end=end_date, freq='ME')  # Month-End
    return list(dates)


def run_backtest(
    start_year: int = 2020,
    end_year: int = 2024,
    top_n: int = 50
) -> dict:
    """
    Run walk-forward backtest on historical data.
    
    Args:
        start_year: Starting year for backtest
        end_year: Ending year for backtest
        top_n: Number of top stocks to hold
    
    Returns:
        Dict with backtest results
    """
    db = SessionLocal()
    
    logger.info("=" * 60)
    logger.info("  WALK-FORWARD BACKTEST")
    logger.info("=" * 60)
    logger.info(f"Period: {start_year} to {end_year}")
    logger.info(f"Strategy: Top {top_n} stocks by Factor Score")
    logger.info("Factors: VSM + BAB + Momentum (12-1)")
    logger.info("=" * 60)
    
    # 1. Load price data (need 1 year before start for factor calculation)
    data_start = f"{start_year - 1}-01-01"
    data_end = f"{end_year}-12-31"
    
    logger.info(f"Loading price data from {data_start} to {data_end}...")
    prices = load_price_data(db, data_start, data_end)
    
    if prices.empty:
        logger.error("No price data available! Run download_history.py first.")
        db.close()
        return {}
    
    logger.info(f"Loaded {len(prices)} price records for {prices['ticker'].nunique()} tickers")
    
    # 2. Generate rebalancing dates
    rebalance_dates = generate_monthly_rebalance_dates(
        f"{start_year}-01-01",
        f"{end_year}-12-31"
    )
    logger.info(f"Rebalancing dates: {len(rebalance_dates)} months")
    
    # 3. Run backtest
    logger.info("Running backtest...")
    equity_curve, trade_logs = run_factor_backtest(prices, rebalance_dates, top_n=top_n)
    
    if equity_curve.empty:
        logger.error("Backtest failed - no results generated")
        db.close()
        return {}
    
    # 4. Calculate metrics
    metrics = calculate_performance_metrics(equity_curve)
    
    logger.info("=" * 60)
    logger.info("  BACKTEST RESULTS")
    logger.info("=" * 60)
    logger.info(f"Total Return:    {metrics.get('total_return', 0):.2%}")
    logger.info(f"Annual Return:   {metrics.get('annual_return', 0):.2%}")
    logger.info(f"Volatility:      {metrics.get('volatility', 0):.2%}")
    logger.info(f"Sharpe Ratio:    {metrics.get('sharpe_ratio', 0):.2f}")
    logger.info(f"Max Drawdown:    {metrics.get('max_drawdown', 0):.2%}")
    logger.info("=" * 60)
    
    # 5. Save results
    output_path = f"data/exports/backtest_{start_year}_{end_year}.csv"
    equity_curve.to_csv(output_path, index=False)
    logger.info(f"Equity curve saved to: {output_path}")
    
    db.close()
    
    # Save trades
    if trade_logs:
        trades_df = pd.DataFrame(trade_logs)
        trades_path = f"data/exports/backtest_trades_{start_year}_{end_year}.csv"
        trades_df.to_csv(trades_path, index=False)
        logger.info(f"Trade logs saved to: {trades_path}")
    
    return {
        'metrics': metrics,
        'equity_curve': equity_curve.to_dict(orient='records'),
        'trades': trade_logs
    }


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run Walk-Forward Backtest')
    parser.add_argument('--start', type=int, default=2020, help='Start year')
    parser.add_argument('--end', type=int, default=2024, help='End year')
    parser.add_argument('--top', type=int, default=50, help='Top N stocks to hold')
    
    args = parser.parse_args()
    
    results = run_backtest(
        start_year=args.start,
        end_year=args.end,
        top_n=args.top
    )
    
    if results:
        print("\n✅ Backtest completed successfully!")
        print(f"Sharpe Ratio: {results['metrics'].get('sharpe_ratio', 0):.2f}")
    else:
        print("\n❌ Backtest failed - check logs for details")


if __name__ == "__main__":
    main()
