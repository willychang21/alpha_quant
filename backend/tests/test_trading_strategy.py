"""
Trading Strategy Backtest - Based on Valuation Model Ratings
=============================================================

Simulates a trading strategy:
1. STRONG BUY → 100% position
2. BUY → 50% position  
3. HOLD → Keep existing position
4. SELL → Exit position

Compares performance vs S&P 500 (SPY)
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

from app.engines.valuation import core as valuation

logging.basicConfig(level=logging.WARNING, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_historical_valuation(ticker, as_of_date):
    """
    Get valuation as of a specific historical date.
    
    Note: This is approximate - we use current financial data
    but historical price to simulate.
    """
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        # Get financials (current - limitation of yfinance)
        income = stock.quarterly_income_stmt
        balance = stock.quarterly_balance_sheet
        cashflow = stock.quarterly_cashflow
        
        # Get historical price data around the date
        start = as_of_date - timedelta(days=365)
        end = as_of_date + timedelta(days=7)
        history = stock.history(start=start, end=end)
        
        if history.empty:
            return None
        
        # Get price as of date
        closest_idx = (history.index - as_of_date).abs().argmin()
        historical_price = history.iloc[closest_idx]['Close']
        
        # Override current price with historical
        info['currentPrice'] = historical_price
        
        # Run valuation
        result = valuation.get_valuation(
            ticker=ticker,
            info=info,
            income=income,
            balance=balance,
            cashflow=cashflow,
            history=history
        )
        
        return {
            'date': as_of_date,
            'price': historical_price,
            'rating': result.rating,
            'fair_value': (result.dcf.sharePrice if result.dcf 
                          else result.ddm.fairValue if result.ddm 
                          else result.reit.fairValue if result.reit else 0),
        }
        
    except Exception as e:
        logger.error(f"Error getting historical valuation for {ticker}: {e}")
        return None

def simulate_trading_strategy(tickers, start_date, end_date, initial_capital=10000):
    """
    Simulate trading based on model ratings.
    
    Strategy:
    - STRONG BUY: Allocate 100% of available cash
    - BUY: Allocate 50% of available cash
    - HOLD: Do nothing
    - SELL: Exit position
    
    Rebalance monthly based on new ratings.
    """
    # Generate monthly rebalance dates
    dates = pd.date_range(start=start_date, end=end_date, freq='MS')
    
    portfolio_value = []
    transactions = []
    positions = {}  # ticker -> shares
    cash = initial_capital
    
    print(f"\n{'='*80}")
    print(f"TRADING STRATEGY BACKTEST")
    print(f"{'='*80}")
    print(f"Initial Capital: ${initial_capital:,.0f}")
    print(f"Period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    print(f"Universe: {len(tickers)} stocks")
    print(f"Rebalance: Monthly")
    print(f"{'='*80}\n")
    
    for i, date in enumerate(dates):
        print(f"\n[{i+1}/{len(dates)}] Rebalancing on {date.strftime('%Y-%m-%d')}...")
        
        # Get valuations for all tickers
        valuations = {}
        for ticker in tickers:
            val = get_historical_valuation(ticker, date)
            if val:
                valuations[ticker] = val
        
        if not valuations:
            print("  ⚠️  No valuations available")
            continue
        
        # Calculate current portfolio value
        portfolio_val = cash
        for ticker, shares in positions.items():
            if ticker in valuations:
                portfolio_val += shares * valuations[ticker]['price']
        
        print(f"  Portfolio Value: ${portfolio_val:,.0f}")
        print(f"  Cash: ${cash:,.0f}")
        print(f"  Positions: {len(positions)}")
        
        # Trading decisions
        buy_candidates = []
        strong_buy_candidates = []
        
        for ticker, val in valuations.items():
            rating = val['rating']
            
            # SELL: Exit existing positions
            if rating == 'SELL' and ticker in positions:
                shares = positions[ticker]
                proceeds = shares * val['price']
                cash += proceeds
                print(f"  🔴 SELL {ticker}: {shares:.2f} shares @ ${val['price']:.2f} = ${proceeds:,.0f}")
                transactions.append({
                    'date': date,
                    'ticker': ticker,
                    'action': 'SELL',
                    'shares': shares,
                    'price': val['price'],
                    'value': proceeds
                })
                del positions[ticker]
            
            # Queue BUY candidates
            elif rating == 'BUY':
                buy_candidates.append((ticker, val))
            elif rating == 'STRONG BUY':
                strong_buy_candidates.append((ticker, val))
        
        # BUY: Allocate cash
        all_buy_candidates = strong_buy_candidates + buy_candidates
        
        if all_buy_candidates:
            # Equal weight allocation among buy candidates
            per_stock_allocation = cash / len(all_buy_candidates)
            
            for ticker, val in all_buy_candidates:
                weight = 1.0 if val['rating'] == 'STRONG BUY' else 0.5
                allocation = per_stock_allocation * weight
                
                if allocation > 100:  # Minimum $100 per trade
                    shares = allocation / val['price']
                    cost = shares * val['price']
                    
                    if ticker in positions:
                        positions[ticker] += shares
                        action = 'ADD'
                    else:
                        positions[ticker] = shares
                        action = 'BUY'
                    
                    cash -= cost
                    
                    print(f"  🟢 {action} {ticker}: {shares:.2f} shares @ ${val['price']:.2f} = ${cost:,.0f} ({val['rating']})")
                    transactions.append({
                        'date': date,
                        'ticker': ticker,
                        'action': action,
                        'shares': shares,
                        'price': val['price'],
                        'value': cost
                    })
        
        # Record portfolio value
        portfolio_value.append({
            'date': date,
            'value': portfolio_val,
            'cash': cash,
            'positions': len(positions)
        })
    
    # Final liquidation
    final_date = end_date
    print(f"\n\nFinal Liquidation on {final_date.strftime('%Y-%m-%d')}...")
    
    for ticker, shares in list(positions.items()):
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(start=final_date-timedelta(days=7), end=final_date+timedelta(days=1))
            if not hist.empty:
                final_price = hist.iloc[-1]['Close']
                proceeds = shares * final_price
                cash += proceeds
                print(f"  💰 Liquidate {ticker}: {shares:.2f} shares @ ${final_price:.2f} = ${proceeds:,.0f}")
        except:
            pass
    
    final_value = cash
    
    return {
        'portfolio_value': pd.DataFrame(portfolio_value),
        'transactions': pd.DataFrame(transactions),
        'final_value': final_value,
        'initial_capital': initial_capital,
        'return': (final_value - initial_capital) / initial_capital
    }

def benchmark_strategy(start_date, end_date, initial_capital=10000):
    """Buy and hold SPY (S&P 500)"""
    spy = yf.Ticker('SPY')
    hist = spy.history(start=start_date-timedelta(days=7), end=end_date+timedelta(days=1))
    
    if hist.empty:
        return None
    
    start_price = hist.iloc[0]['Close']
    end_price = hist.iloc[-1]['Close']
    
    shares = initial_capital / start_price
    final_value = shares * end_price
    
    return {
        'initial_capital': initial_capital,
        'final_value': final_value,
        'return': (final_value - initial_capital) / initial_capital,
        'start_price': start_price,
        'end_price': end_price
    }

def run_trading_backtest():
    """Run comprehensive trading backtest"""
    
    # Test universe - mix of different stocks
    tickers = [
        # Strong performers (should BUY)
        'JPM', 'WFC', 'GOOGL',
        
        # Tech (mixed signals)
        'AAPL', 'NVDA', 'MSFT',
        
        # Value
        'KO', 'PG', 'JNJ'
    ]
    
    # Backtest period: 1 year ago to now
    end_date = pd.Timestamp.now()
    start_date = end_date - pd.DateOffset(years=1)
    
    initial_capital = 10000
    
    # Run strategy
    print("\n🚀 Running Model-Based Trading Strategy...")
    strategy_results = simulate_trading_strategy(
        tickers=tickers,
        start_date=start_date,
        end_date=end_date,
        initial_capital=initial_capital
    )
    
    # Run benchmark
    print("\n\n📊 Running Benchmark (SPY Buy & Hold)...")
    benchmark_results = benchmark_strategy(start_date, end_date, initial_capital)
    
    # Results
    print(f"\n\n{'='*80}")
    print("RESULTS")
    print(f"{'='*80}\n")
    
    strategy_return = strategy_results['return']
    benchmark_return = benchmark_results['return']
    alpha = strategy_return - benchmark_return
    
    print(f"Model Strategy:")
    print(f"  Initial: ${strategy_results['initial_capital']:,.0f}")
    print(f"  Final:   ${strategy_results['final_value']:,.0f}")
    print(f"  Return:  {strategy_return:+.1%}")
    print()
    print(f"Benchmark (SPY):")
    print(f"  Initial: ${benchmark_results['initial_capital']:,.0f}")
    print(f"  Final:   ${benchmark_results['final_value']:,.0f}")
    print(f"  Return:  {benchmark_return:+.1%}")
    print()
    print(f"Alpha: {alpha:+.1%}")
    
    if alpha > 0.05:  # Outperform by >5%
        print("\n🎉 Model OUTPERFORMED benchmark!")
    elif alpha > -0.05:
        print("\n➡️  Model MATCHED benchmark")
    else:
        print("\n❌ Model UNDERPERFORMED benchmark")
    
    # Save results
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
    perf_path = os.path.join(data_dir, 'strategy_performance.csv')
    trans_path = os.path.join(data_dir, 'strategy_transactions.csv')
    
    strategy_results['portfolio_value'].to_csv(perf_path, index=False)
    strategy_results['transactions'].to_csv(trans_path, index=False)
    
    print(f"\n✅ Results saved to:")
    print(f"  - {perf_path}")
    print(f"  - {trans_path}")
    
    print(f"\n{'='*80}")

if __name__ == "__main__":
    run_trading_backtest()
