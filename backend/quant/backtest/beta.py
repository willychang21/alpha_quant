"""Historical Beta Calculator.

Calculates rolling historical beta to avoid look-ahead bias in backtesting.
Uses only price data available before the backtest date.

Properties:
- Property 1: Uses only past data (no future data)
- Property 2: Formula: cov(stock, market) / var(market), or 1.0 if insufficient
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def calculate_historical_beta(
    stock_returns: pd.Series,
    market_returns: pd.Series,
    window: int = 252
) -> float:
    """Calculate historical beta avoiding look-ahead bias.
    
    Uses rolling window of past returns to compute beta.
    
    Args:
        stock_returns: Stock daily returns series (with DatetimeIndex)
        market_returns: Market (SPY) daily returns series (with DatetimeIndex)
        window: Rolling window size in trading days (default: 252 = 1 year)
    
    Returns:
        Historical beta value. Returns 1.0 if:
        - Insufficient data (< window days)
        - Zero market variance
        - Insufficient aligned data after join
    
    Example:
        >>> stock_ret = stock_prices.pct_change().dropna()
        >>> spy_ret = spy_prices.pct_change().dropna()
        >>> beta = calculate_historical_beta(stock_ret, spy_ret)
    """
    # Check minimum data requirement
    if len(stock_returns) < window:
        logger.debug(f"Insufficient stock data: {len(stock_returns)} < {window}")
        return 1.0
    
    if len(market_returns) < window:
        logger.debug(f"Insufficient market data: {len(market_returns)} < {window}")
        return 1.0
    
    # Use most recent 'window' days of data
    stock_ret = stock_returns.iloc[-window:]
    market_ret = market_returns.iloc[-window:]
    
    # Align indices (inner join on dates)
    aligned = pd.concat([stock_ret, market_ret], axis=1, join='inner')
    aligned.columns = ['stock', 'market']
    
    # Check if alignment retained sufficient data
    if len(aligned) < window // 2:
        logger.debug(f"Insufficient aligned data: {len(aligned)} < {window // 2}")
        return 1.0
    
    # Calculate covariance and variance
    covariance = aligned['stock'].cov(aligned['market'])
    market_variance = aligned['market'].var()
    
    # Handle zero variance edge case
    if market_variance == 0 or np.isnan(market_variance):
        logger.debug("Zero or NaN market variance, returning default beta 1.0")
        return 1.0
    
    beta = covariance / market_variance
    
    # Handle extreme or NaN values
    if np.isnan(beta) or np.isinf(beta):
        logger.debug(f"Invalid beta value {beta}, returning 1.0")
        return 1.0
    
    return float(beta)


def get_market_returns(
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    market_proxy: str = 'SPY'
) -> Optional[pd.Series]:
    """Fetch market proxy returns for beta calculation.
    
    Args:
        start_date: Start date for historical data
        end_date: End date (typically backtest date)
        market_proxy: Market proxy ticker (default: SPY)
    
    Returns:
        Series of daily returns, or None if data unavailable
    """
    try:
        import yfinance as yf
        
        # Add buffer for lookback
        start_with_buffer = start_date - pd.Timedelta(days=365 + 30)
        
        spy = yf.Ticker(market_proxy)
        hist = spy.history(start=start_with_buffer, end=end_date)
        
        if hist.empty:
            logger.warning(f"No market data for {market_proxy}")
            return None
        
        returns = hist['Close'].pct_change().dropna()
        return returns
        
    except Exception as e:
        logger.warning(f"Failed to fetch market returns: {e}")
        return None


def calculate_stock_beta_at_date(
    ticker: str,
    as_of_date: pd.Timestamp,
    window: int = 252,
    market_proxy: str = 'SPY'
) -> float:
    """Calculate historical beta for a stock as of a specific date.
    
    Convenience function that fetches data and calculates beta.
    Only uses data available before as_of_date (no future data).
    
    Args:
        ticker: Stock ticker symbol
        as_of_date: Calculation date (only data before this date is used)
        window: Lookback window in trading days
        market_proxy: Market proxy ticker
    
    Returns:
        Historical beta value (default 1.0 if calculation fails)
    """
    try:
        import yfinance as yf
        
        # Ensure we have enough history
        start_date = as_of_date - pd.Timedelta(days=window * 2)
        
        # Fetch stock data up to (not including) as_of_date
        stock = yf.Ticker(ticker)
        stock_hist = stock.history(start=start_date, end=as_of_date)
        
        if stock_hist.empty:
            logger.debug(f"No stock history for {ticker}")
            return 1.0
        
        stock_returns = stock_hist['Close'].pct_change().dropna()
        
        # Fetch market data
        market_returns = get_market_returns(start_date, as_of_date, market_proxy)
        if market_returns is None:
            return 1.0
        
        return calculate_historical_beta(stock_returns, market_returns, window)
        
    except Exception as e:
        logger.warning(f"Failed to calculate beta for {ticker}: {e}")
        return 1.0
