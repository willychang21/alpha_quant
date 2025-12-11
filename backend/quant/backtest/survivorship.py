"""Survivorship Bias Validator.

Validates strategy alpha by comparing against full-market ETFs
and applies survivorship penalties based on universe type.

Delist rates by universe:
- Broad market: 3%/year
- Large cap (S&P 500): 1%/year
- Small cap: 6%/year
"""

import logging
from typing import Dict, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Full-market ETF proxies
MARKET_PROXIES = {
    'VTI': 'US Total Market (3800+ stocks)',
    'IWV': 'Russell 3000 (large+mid+small cap)',
    'SPY': 'S&P 500 (large cap)',
    'IWM': 'Russell 2000 (small cap)',
}

# Annual delist rates by universe type
DELIST_RATES = {
    'broad': 0.03,      # ~3%/year for broad market
    'large_cap': 0.01,  # ~1%/year for S&P 500
    'small_cap': 0.06,  # ~6%/year for small cap
}


def validate_against_etf(
    strategy_returns: pd.Series,
    benchmark: str = 'VTI'
) -> Dict[str, float]:
    """Validate strategy alpha against a full-market ETF.
    
    Compares strategy returns to a broad market ETF to assess
    whether alpha is real or due to survivorship bias.
    
    Args:
        strategy_returns: Strategy daily returns (with DatetimeIndex)
        benchmark: ETF ticker for comparison (default: VTI)
    
    Returns:
        Dictionary with:
        - strategy_cagr: Strategy annualized return
        - etf_cagr: ETF annualized return
        - excess_return: Strategy - ETF return
        - is_outperforming: True if strategy beats ETF
    """
    try:
        if len(strategy_returns) < 20:
            logger.warning("Insufficient data for ETF validation")
            return {}
        
        import yfinance as yf
        
        start_date = strategy_returns.index[0]
        end_date = strategy_returns.index[-1]
        
        etf = yf.Ticker(benchmark)
        etf_hist = etf.history(start=start_date, end=end_date)
        
        if etf_hist.empty:
            logger.warning(f"ETF {benchmark} data unavailable")
            return {}
        
        etf_returns = etf_hist['Close'].pct_change().dropna()
        
        # Align dates
        aligned = pd.concat([strategy_returns, etf_returns], axis=1, join='inner')
        aligned.columns = ['strategy', 'etf']
        
        if len(aligned) < 20:
            logger.warning("Insufficient aligned data for ETF validation")
            return {}
        
        # Calculate CAGRs
        n_days = len(aligned)
        strategy_cagr = (1 + aligned['strategy']).prod() ** (252 / n_days) - 1
        etf_cagr = (1 + aligned['etf']).prod() ** (252 / n_days) - 1
        
        return {
            'strategy_cagr': float(strategy_cagr),
            'etf_cagr': float(etf_cagr),
            'excess_return': float(strategy_cagr - etf_cagr),
            'is_outperforming': strategy_cagr > etf_cagr,
            'benchmark_used': benchmark
        }
        
    except Exception as e:
        logger.warning(f"ETF validation failed: {e}")
        return {}


def apply_survivorship_penalty(
    cagr: float,
    years: float,
    universe: str = 'broad'
) -> Dict[str, float]:
    """Apply survivorship bias penalty to backtest returns.
    
    Estimates the performance drag from using a survivorship-biased
    dataset (only stocks that exist today, missing delisted stocks).
    
    Args:
        cagr: Original annualized return from backtest
        years: Number of years in backtest
        universe: Type of universe - 'broad', 'large_cap', or 'small_cap'
    
    Returns:
        Dictionary with:
        - original_cagr: Input CAGR
        - adjusted_cagr: CAGR after survivorship penalty
        - annual_penalty: Annual performance drag
        - total_penalty: Total penalty over backtest period
        - assumption: Description of assumptions made
    """
    delist_rate = DELIST_RATES.get(universe, DELIST_RATES['broad'])
    
    # Assume delisted stocks average -50% loss
    avg_delist_loss = -0.50
    
    # Annual penalty = probability of delist * average loss
    annual_penalty = delist_rate * avg_delist_loss
    
    # Apply penalty to CAGR
    adjusted_cagr = cagr + annual_penalty
    
    return {
        'original_cagr': cagr,
        'adjusted_cagr': adjusted_cagr,
        'annual_penalty': annual_penalty,
        'total_penalty': annual_penalty * years,
        'delist_rate': delist_rate,
        'assumption': f'Assumes {delist_rate*100:.0f}%/year delist rate with average -50% loss'
    }
