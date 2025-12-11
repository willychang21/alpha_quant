"""Walk Forward Backtester with Benchmark Comparison.

Runs strategy over time with rebalancing and calculates
comprehensive performance metrics including benchmark comparison.
"""

from sqlalchemy.orm import Session
from datetime import date
import pandas as pd
import numpy as np
from quant.backtest.engine import BacktestEngine
from quant.backtest.benchmark import calculate_benchmark_metrics
from quant.backtest.survivorship import apply_survivorship_penalty, validate_against_etf
import logging

logger = logging.getLogger(__name__)


class WalkForwardBacktester:
    """Walk-forward backtester with benchmark and survivorship metrics."""
    
    def __init__(self, db: Session, benchmark: str = 'SPY'):
        self.db = db
        self.benchmark = benchmark
        
    def run(self, start_date: date, end_date: date):
        """Run the strategy over the specified period.
        
        Returns equity curve and comprehensive performance metrics
        including benchmark comparison and survivorship adjustment.
        """
        engine = BacktestEngine(self.db)
        history_df = engine.run_backtest(start_date, end_date)
        
        if history_df.empty:
            return {
                "equity_curve": [],
                "metrics": {}
            }
            
        # Calculate base metrics
        metrics = self._calculate_metrics(history_df)
        
        # Add benchmark comparison metrics
        benchmark_metrics = self._calculate_benchmark_metrics(history_df)
        metrics.update(benchmark_metrics)
        
        # Add survivorship bias adjustment
        if 'cagr' in metrics:
            years = (end_date - start_date).days / 365
            survivorship = apply_survivorship_penalty(
                metrics['cagr'], 
                years, 
                universe='broad'
            )
            metrics['adjusted_cagr'] = survivorship['adjusted_cagr']
            metrics['survivorship_penalty'] = survivorship['annual_penalty']
            metrics['survivorship_note'] = survivorship['assumption']
        
        return {
            "equity_curve": history_df.to_dict(orient='records'),
            "metrics": metrics
        }

    def _calculate_metrics(self, df: pd.DataFrame):
        """Calculate base performance metrics."""
        if df.empty or 'equity' not in df.columns:
            return {}
            
        # Daily Returns
        df = df.copy()
        df['returns'] = df['equity'].pct_change().fillna(0)
        
        total_return = (df['equity'].iloc[-1] / df['equity'].iloc[0]) - 1
        
        # Annualized Return (CAGR)
        days = (df['date'].iloc[-1] - df['date'].iloc[0]).days
        cagr = (1 + total_return) ** (365 / days) - 1 if days > 0 else 0
        
        # Volatility (Annualized)
        volatility = df['returns'].std() * np.sqrt(252)
        
        # Sharpe Ratio (Rf=4%)
        rf = 0.04
        sharpe = (cagr - rf) / volatility if volatility > 0 else 0
        
        # Max Drawdown
        df['cummax'] = df['equity'].cummax()
        df['drawdown'] = (df['equity'] - df['cummax']) / df['cummax']
        max_drawdown = df['drawdown'].min()
        
        return {
            "total_return": total_return,
            "cagr": cagr,
            "volatility": volatility,
            "sharpe_ratio": sharpe,
            "max_drawdown": max_drawdown
        }
    
    def _calculate_benchmark_metrics(self, df: pd.DataFrame):
        """Calculate benchmark comparison metrics (alpha, beta, IR)."""
        if df.empty or 'equity' not in df.columns:
            return {}
        
        try:
            # Create returns series with date index
            df = df.copy()
            df['returns'] = df['equity'].pct_change().fillna(0)
            returns_series = df.set_index('date')['returns']
            
            # Convert to DatetimeIndex if needed
            if not isinstance(returns_series.index, pd.DatetimeIndex):
                returns_series.index = pd.to_datetime(returns_series.index)
            
            # Calculate benchmark metrics
            benchmark_metrics = calculate_benchmark_metrics(
                returns_series,
                benchmark=self.benchmark
            )
            
            return benchmark_metrics
            
        except Exception as e:
            logger.warning(f"Could not calculate benchmark metrics: {e}")
            return {}
