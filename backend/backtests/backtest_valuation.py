"""
Valuation Model Backtesting Script
===================================

Tests the accuracy of our DCF/DDM/REIT valuation models by:
1. Fetching historical financial data (quarterly)
2. Running valuation model as-of each quarter
3. Comparing Fair Value vs Actual Price
4. Measuring forward returns (1M, 3M, 6M, 1Y)
5. Calculating hit rate and alpha generation

Author: Quant Engine
Date: 2025-11-29
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import yfinance as yf
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
from app.engines.valuation import core as valuation
from app.domain import schemas

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_historical_financials(ticker: str, periods: int = 12) -> List[Dict]:
    """
    Fetch historical quarterly financial statements.
    
    Args:
        ticker: Stock ticker
        periods: Number of quarters to look back
        
    Returns:
        List of dicts with {date, income, balance, cashflow, info}
    """
    logger.info(f"Fetching historical data for {ticker}...")
    
    stock = yf.Ticker(ticker)
    
    # Get quarterly financials (most recent first)
    income_q = stock.quarterly_income_stmt
    balance_q = stock.quarterly_balance_sheet
    cashflow_q = stock.quarterly_cashflow
    
    if income_q.empty or balance_q.empty or cashflow_q.empty:
        logger.warning(f"No quarterly data for {ticker}")
        return []
    
    # Get available dates (intersection of all statements)
    dates = list(income_q.columns)[:periods]
    
    historical_data = []
    
    for date in dates:
        # Create snapshot as-of this date
        try:
            income_slice = income_q[[date]]
            balance_slice = balance_q[[date]]
            cashflow_slice = cashflow_q[[date]]
            
            # Get info (current info as proxy - limitation of yfinance)
            # In reality, we'd need historical info, but this is best-effort
            info = stock.info
            
            historical_data.append({
                'date': date,
                'income': income_slice,
                'balance': balance_slice,
                'cashflow': cashflow_slice,
                'info': info
            })
            
        except Exception as e:
            logger.error(f"Error processing {ticker} at {date}: {e}")
            continue
    
    logger.info(f"Fetched {len(historical_data)} quarters for {ticker}")
    return historical_data

def get_price_at_date(ticker: str, target_date: pd.Timestamp) -> float:
    """Get stock price as close to target_date as possible."""
    try:
        stock = yf.Ticker(ticker)
        # Fetch a window around the target date
        start = target_date - timedelta(days=7)
        end = target_date + timedelta(days=7)
        
        hist = stock.history(start=start, end=end)
        
        if hist.empty:
            return None
        
        # Get closest date
        # Ensure hist.index is timezone-naive for comparison
        hist_index = hist.index.tz_localize(None)
        diff = hist_index - target_date
        closest_idx = np.abs(diff).argmin()
        return hist.iloc[closest_idx]['Close']
        
    except Exception as e:
        logger.error(f"Error fetching price for {ticker} at {target_date}: {e}")
        return None

def calculate_forward_returns(ticker: str, base_date: pd.Timestamp) -> Dict[str, float]:
    """
    Calculate forward returns from base_date.
    
    Returns:
        {
            '1M': return_1m,
            '3M': return_3m,
            '6M': return_6m,
            '1Y': return_1y
        }
    """
    try:
        stock = yf.Ticker(ticker)
        
        # Get base price
        base_price = get_price_at_date(ticker, base_date)
        
        if not base_price:
            return {}
        
        returns = {}
        
        periods = {
            '1M': 30,
            '3M': 90,
            '6M': 180,
            '1Y': 365
        }
        
        for label, days in periods.items():
            future_date = base_date + timedelta(days=days)
            
            # Don't calculate if future date is beyond today
            if future_date > pd.Timestamp.now():
                continue
            
            future_price = get_price_at_date(ticker, future_date)
            
            if future_price:
                ret = (future_price - base_price) / base_price
                returns[label] = ret
        
        return returns
        
    except Exception as e:
        logger.error(f"Error calculating forward returns: {e}")
        return {}

def run_backtest(ticker: str, periods: int = 8) -> pd.DataFrame:
    """
    Run backtest for a single ticker.
    
    Returns:
        DataFrame with columns:
        - date: Quarter end date
        - actual_price: Stock price at that time
        - fair_value: Model's fair value estimate
        - discount: (Fair Value - Price) / Price
        - rating: Model rating (BUY/HOLD/SELL)
        - return_1M, return_3M, return_6M, return_1Y
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"BACKTEST: {ticker}")
    logger.info(f"{'='*60}\n")
    
    # Get historical financial data
    historical = get_historical_financials(ticker, periods)
    
    if not historical:
        logger.error(f"No historical data for {ticker}")
        return pd.DataFrame()
    
    results = []
    
    for snapshot in historical:
        date = snapshot['date']
        income = snapshot['income']
        balance = snapshot['balance']
        cashflow = snapshot['cashflow']
        info = snapshot['info']
        
        logger.info(f"\nProcessing {ticker} as-of {date.strftime('%Y-%m-%d')}...")
        
        # Get actual price at that time
        actual_price = get_price_at_date(ticker, date)
        
        if not actual_price:
            logger.warning(f"Could not get price for {ticker} at {date}")
            continue
        
        # Get historical price data (for risk metrics)
        try:
            stock = yf.Ticker(ticker)
            hist_start = date - timedelta(days=365)
            history = stock.history(start=hist_start, end=date)
        except:
            history = pd.DataFrame()
        
        # Run valuation model
        try:
            valuation_result = valuation.get_valuation(
                ticker=ticker,
                info=info,
                income=income,
                balance=balance,
                cashflow=cashflow,
                history=history
            )
            
            # Extract fair value
            if valuation_result.dcf:
                fair_value = valuation_result.dcf.sharePrice
            elif valuation_result.ddm:
                fair_value = valuation_result.ddm.fairValue
            elif valuation_result.reit:
                fair_value = valuation_result.reit.fairValue
            else:
                fair_value = None
            
            if not fair_value or fair_value <= 0:
                logger.warning(f"Invalid fair value: {fair_value}")
                continue
            
            # Calculate discount/premium
            discount = (fair_value - actual_price) / actual_price
            
            # Get rating
            rating = valuation_result.rating
            
            # Calculate forward returns
            forward_returns = calculate_forward_returns(ticker, date)
            
            result = {
                'date': date,
                'actual_price': actual_price,
                'fair_value': fair_value,
                'discount': discount,
                'rating': rating,
                **forward_returns
            }
            
            results.append(result)
            
            logger.info(f"  Price: ${actual_price:.2f}")
            logger.info(f"  Fair Value: ${fair_value:.2f}")
            logger.info(f"  Discount: {discount:.1%}")
            logger.info(f"  Rating: {rating}")
            if forward_returns:
                logger.info(f"  Forward Returns: {forward_returns}")
            
        except Exception as e:
            logger.error(f"Error running valuation for {ticker} at {date}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    df = pd.DataFrame(results)
    
    if not df.empty:
        df = df.sort_values('date')
    
    return df

def analyze_backtest_results(df: pd.DataFrame) -> Dict:
    """
    Analyze backtest results and calculate performance metrics.
    
    Returns:
        Dict with:
        - hit_rate: Percentage of correct directional predictions
        - avg_return_when_buy: Average return when model said BUY
        - avg_return_when_sell: Average return when model said SELL
        - correlation: Correlation between discount and forward returns
    """
    if df.empty:
        return {}
    
    metrics = {}
    
    # Hit Rate: Did STRONG BUY/BUY actually outperform?
    if '1Y' in df.columns:
        buy_mask = df['rating'].isin(['STRONG BUY', 'BUY'])
        sell_mask = df['rating'] == 'SELL'
        
        if buy_mask.any():
            avg_return_buy = df[buy_mask]['1Y'].mean()
            metrics['avg_return_when_buy'] = avg_return_buy
        
        if sell_mask.any():
            avg_return_sell = df[sell_mask]['1Y'].mean()
            metrics['avg_return_when_sell'] = avg_return_sell
        
        # Hit Rate: Model said BUY and return > 0
        if buy_mask.any():
            correct_buys = (df[buy_mask]['1Y'] > 0).sum()
            total_buys = buy_mask.sum()
            metrics['buy_hit_rate'] = correct_buys / total_buys if total_buys > 0 else 0
    
    # Correlation between discount and forward returns
    for period in ['1M', '3M', '6M', '1Y']:
        if period in df.columns:
            valid = df[['discount', period]].dropna()
            if len(valid) >= 3:
                corr = valid['discount'].corr(valid[period])
                metrics[f'correlation_{period}'] = corr
    
    # Mean Absolute Error
    df['valuation_error'] = abs(df['fair_value'] - df['actual_price']) / df['actual_price']
    metrics['mean_absolute_error'] = df['valuation_error'].mean()
    
    return metrics

def print_summary(ticker: str, df: pd.DataFrame, metrics: Dict):
    """Print backtest summary."""
    print(f"\n{'='*60}")
    print(f"BACKTEST SUMMARY: {ticker}")
    print(f"{'='*60}\n")
    
    print(f"Total Observations: {len(df)}")
    print(f"Date Range: {df['date'].min().strftime('%Y-%m-%d')} to {df['date'].max().strftime('%Y-%m-%d')}\n")
    
    print("Performance Metrics:")
    print("-" * 60)
    
    if 'mean_absolute_error' in metrics:
        print(f"Mean Absolute Error: {metrics['mean_absolute_error']:.1%}")
    
    if 'buy_hit_rate' in metrics:
        print(f"Buy Hit Rate (1Y): {metrics['buy_hit_rate']:.1%}")
    
    if 'avg_return_when_buy' in metrics:
        print(f"Avg 1Y Return when BUY: {metrics['avg_return_when_buy']:.1%}")
    
    if 'avg_return_when_sell' in metrics:
        print(f"Avg 1Y Return when SELL: {metrics['avg_return_when_sell']:.1%}")
    
    print("\nDiscount vs Forward Return Correlations:")
    for period in ['1M', '3M', '6M', '1Y']:
        key = f'correlation_{period}'
        if key in metrics:
            print(f"  {period}: {metrics[key]:.3f}")
    
    print("\n" + "="*60)

if __name__ == "__main__":
    # Test with multiple tickers
    tickers = ['AAPL', 'MSFT', 'NVDA', 'GOOGL']
    
    all_results = {}
    
    for ticker in tickers:
        try:
            df = run_backtest(ticker, periods=8)
            
            if not df.empty:
                metrics = analyze_backtest_results(df)
                all_results[ticker] = {'df': df, 'metrics': metrics}
                print_summary(ticker, df, metrics)
                
                # Save to CSV
                output_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', f'backtest_{ticker}.csv')
                df.to_csv(output_path, index=False)
                logger.info(f"\nSaved results to {output_path}")
            
        except Exception as e:
            logger.error(f"Failed to backtest {ticker}: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*60)
    print("BACKTEST COMPLETE")
    print("="*60)
