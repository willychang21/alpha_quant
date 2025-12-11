"""Benchmark Metrics Calculator.

Calculates strategy performance relative to a benchmark (SPY by default).
Provides alpha, beta, tracking error, and information ratio.
"""

import logging
from typing import Dict, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def calculate_benchmark_metrics(
    strategy_returns: pd.Series,
    benchmark: str = 'SPY',
    risk_free_rate: float = 0.04
) -> Dict[str, float]:
    """Calculate strategy performance metrics relative to a benchmark.
    
    Args:
        strategy_returns: Strategy daily returns series (with DatetimeIndex)
        benchmark: Benchmark ticker symbol (default: SPY)
        risk_free_rate: Annual risk-free rate (default: 0.04 = 4%)
    
    Returns:
        Dictionary with metrics:
        - alpha: Jensen's Alpha (annualized)
        - beta: Strategy beta relative to benchmark
        - tracking_error: Annualized tracking error
        - information_ratio: Excess return / tracking error
        - benchmark_cagr: Benchmark annualized return
        
        Returns empty dict if calculation fails.
    
    Example:
        >>> returns = equity_curve.pct_change().dropna()
        >>> metrics = calculate_benchmark_metrics(returns)
        >>> print(f"Alpha: {metrics['alpha']:.2%}")
    """
    try:
        if len(strategy_returns) < 20:
            logger.warning("Insufficient strategy data for benchmark metrics")
            return {}
        
        # Get benchmark data
        start_date = strategy_returns.index[0]
        end_date = strategy_returns.index[-1]
        
        benchmark_returns = _fetch_benchmark_returns(benchmark, start_date, end_date)
        if benchmark_returns is None:
            return {}
        
        # Align dates (inner join)
        aligned = pd.concat([strategy_returns, benchmark_returns], axis=1, join='inner')
        aligned.columns = ['strategy', 'benchmark']
        
        if len(aligned) < 20:
            logger.warning(f"Insufficient aligned data: {len(aligned)} < 20")
            return {}
        
        # Calculate Beta
        covariance = aligned['strategy'].cov(aligned['benchmark'])
        market_variance = aligned['benchmark'].var()
        beta = covariance / market_variance if market_variance > 0 else 1.0
        
        # Annualized returns (CAGR)
        n_days = len(aligned)
        strategy_total = (1 + aligned['strategy']).prod()
        benchmark_total = (1 + aligned['benchmark']).prod()
        
        strategy_cagr = strategy_total ** (252 / n_days) - 1
        benchmark_cagr = benchmark_total ** (252 / n_days) - 1
        
        # Jensen's Alpha: (Rp - Rf) - beta * (Rm - Rf)
        rf = risk_free_rate
        alpha = (strategy_cagr - rf) - beta * (benchmark_cagr - rf)
        
        # Tracking Error & Information Ratio
        excess_returns = aligned['strategy'] - aligned['benchmark']
        tracking_error = excess_returns.std() * np.sqrt(252)
        
        if tracking_error > 0:
            information_ratio = excess_returns.mean() * 252 / tracking_error
        else:
            information_ratio = 0.0
        
        return {
            'alpha': float(alpha),
            'beta': float(beta),
            'tracking_error': float(tracking_error),
            'information_ratio': float(information_ratio),
            'benchmark_cagr': float(benchmark_cagr)
        }
        
    except Exception as e:
        logger.warning(f"Failed to calculate benchmark metrics: {e}")
        return {}


def _fetch_benchmark_returns(
    benchmark: str,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp
) -> Optional[pd.Series]:
    """Fetch benchmark daily returns.
    
    Args:
        benchmark: Benchmark ticker
        start_date: Start date
        end_date: End date
    
    Returns:
        Series of daily returns or None if unavailable
    """
    try:
        import yfinance as yf
        
        spy = yf.Ticker(benchmark)
        hist = spy.history(start=start_date, end=end_date)
        
        if hist.empty:
            logger.warning(f"Benchmark {benchmark} data unavailable")
            return None
        
        returns = hist['Close'].pct_change().dropna()
        return returns
        
    except Exception as e:
        logger.warning(f"Failed to fetch benchmark data: {e}")
        return None


def calculate_alpha(
    strategy_cagr: float,
    benchmark_cagr: float,
    beta: float,
    risk_free_rate: float = 0.04
) -> float:
    """Calculate Jensen's Alpha.
    
    Alpha = (Rp - Rf) - beta * (Rm - Rf)
    
    Args:
        strategy_cagr: Strategy annualized return
        benchmark_cagr: Benchmark annualized return
        beta: Strategy beta
        risk_free_rate: Annual risk-free rate
    
    Returns:
        Annualized alpha
    """
    return (strategy_cagr - risk_free_rate) - beta * (benchmark_cagr - risk_free_rate)
