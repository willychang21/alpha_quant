"""
Simplified Trading Strategy Validation
=======================================

Uses CURRENT ratings to demonstrate trading strategy concept.
Shows what portfolio performance would be if we followed model ratings.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Load our backtest results
results = pd.read_csv('comprehensive_test_results.csv')

print(f"\n{'='*80}")
print("TRADING STRATEGY SIMULATION (Based on Current Ratings)")
print(f"{'='*80}\n")

# Filter valid results
valid = results[results['status'] == 'OK'].copy()

print(f"Valid stocks: {len(valid)}")
print(f"\nRating Distribution:")
print(valid['rating'].value_counts())

# Strategy allocation
print(f"\n{'-'*80}")
print("STRATEGY RULES:")
print(f"{'-'*80}")
print("STRONG BUY → Allocate 15% of portfolio")
print("BUY → Allocate 10% of portfolio")
print("HOLD → Allocate 5% of portfolio")
print("SELL → 0% allocation")
print()

# Calculate allocations
initial_capital = 10000
allocations = []

for _, row in valid.iterrows():
    ticker = row['ticker']
    rating = row['rating']
    
    if rating == 'STRONG BUY':
        weight = 0.15
    elif rating == 'BUY':
        weight = 0.10
    elif rating == 'HOLD':
        weight = 0.05
    else:  # SELL
        weight = 0.00
    
    if weight > 0:
        allocations.append({
            'ticker': ticker,
            'rating': rating,
            'weight': weight,
            'capital': initial_capital * weight
        })

# Show portfolio
print(f"{'='*80}")
print("RECOMMENDED PORTFOLIO (Based on Current Ratings)")
print(f"{'='*80}\n")

total_allocation = sum(a['weight'] for a in allocations)
cash_reserve = 1 - total_allocation

df_portfolio = pd.DataFrame(allocations)
if not df_portfolio.empty:
    df_portfolio = df_portfolio.sort_values('weight', ascending=False)
    
    for _, row in df_portfolio.iterrows():
        print(f"{row['ticker']:6s} | {row['rating']:12s} | {row['weight']:.0%} | ${row['capital']:,.0f}")
    
    print(f"{'-'*80}")
    print(f"{'CASH':6s} | {'RESERVE':12s} | {cash_reserve:.0%} | ${initial_capital * cash_reserve:,.0f}")
    print(f"{'-'*80}")
    print(f"{'TOTAL':6s} | {'':12s} | 100% | ${initial_capital:,.0f}")

print(f"\n\n{'='*80}")
print("PORTFOLIO SUMMARY")
print(f"{'='*80}\n")

strong_buys = valid[valid['rating'] == 'STRONG BUY']
buys = valid[valid['rating'] == 'BUY']
holds = valid[valid['rating'] == 'HOLD']

print(f"STRONG BUY Stocks ({len(strong_buys)}):")
if not strong_buys.empty:
    for _, row in strong_buys.iterrows():
        print(f"  ✅ {row['ticker']:6s} | Disc: {row['discount']:+.1%} | Model: {row['model']}")
else:
    print("  None")

print(f"\nBUY Stocks ({len(buys)}):")
if not buys.empty:
    for _, row in buys.iterrows():
        print(f"  ✅ {row['ticker']:6s} | Disc: {row['discount']:+.1%} | Model: {row['model']}")
else:
    print("  None")

print(f"\nHOLD Stocks ({len(holds)}):")
if not holds.empty:
    for _, row in holds.iterrows():
        print(f"  ➡️  {row['ticker']:6s} | Disc: {row['discount']:+.1%} | Model: {row['model']}")
else:
    print("  None")

# Calculate expected return based on discounts
print(f"\n\n{'='*80}")
print("EXPECTED RETURNS (If Model is Correct)")
print(f"{'='*80}\n")

if not df_portfolio.empty:
    weighted_discount = 0
    
    for _, row in df_portfolio.iterrows():
        ticker_data = valid[valid['ticker'] == row['ticker']].iloc[0]
        discount = ticker_data['discount']
        weight = row['weight']
        
        weighted_discount += discount * weight
        
        print(f"{row['ticker']:6s} | Weight: {weight:.0%} | Discount: {discount:+.1%} | Contribution: {discount*weight:+.1%}")
    
    print(f"{'-'*80}")
    print(f"Portfolio Weighted Average Discount: {weighted_discount:+.1%}")
    print()
    
    if weighted_discount > 0.10:
        print("🎯 Model suggests portfolio is UNDERVALUED by {:.1%}".format(weighted_discount))
        print("   Expected upside if model is correct!")
    elif weighted_discount < -0.10:
        print("⚠️  Model suggests portfolio is OVERVALUED by {:.1%}".format(abs(weighted_discount)))
        print("   Consider reducing exposure")
    else:
        print("➡️  Portfolio is FAIRLY VALUED")

# Historical performance check (simple)
print(f"\n\n{'='*80}")
print("ACTUAL 1-YEAR PERFORMANCE CHECK")
print(f"{'='*80}\n")

print("Fetching 1-year returns for top picks...")

top_picks = df_portfolio.head(5) if not df_portfolio.empty else pd.DataFrame()

if not top_picks.empty:
    returns_1y = []
    
    for _, row in top_picks.iterrows():
        ticker = row['ticker']
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(period="1y")
            
            if not hist.empty and len(hist) > 1:
                start_price = hist.iloc[0]['Close']
                end_price = hist.iloc[-1]['Close']
                ret = (end_price - start_price) / start_price
                
                returns_1y.append({
                    'ticker': ticker,
                    'rating': row['rating'],
                    'return_1y': ret
                })
                
                print(f"{ticker:6s} | {row['rating']:12s} | 1Y Return: {ret:+.1%}")
        except:
            pass
    
    if returns_1y:
        df_returns = pd.DataFrame(returns_1y)
        avg_return = df_returns['return_1y'].mean()
        
        # Compare to SPY
        try:
            spy = yf.Ticker('SPY')
            spy_hist = spy.history(period="1y")
            spy_return = (spy_hist.iloc[-1]['Close'] - spy_hist.iloc[0]['Close']) / spy_hist.iloc[0]['Close']
            
            print(f"{'-'*80}")
            print(f"Portfolio Avg: {avg_return:+.1%}")
            print(f"SPY (Benchmark): {spy_return:+.1%}")
            print(f"Alpha: {avg_return - spy_return:+.1%}")
            
            if avg_return > spy_return:
                print("\n🎉 Model selections OUTPERFORMED SPY!")
            else:
                print("\n⚠️  Model selections underperformed SPY")
        except:
            pass

print(f"\n\n{'='*80}")
print("CONCLUSION")
print(f"{'='*80}\n")

print("This simulation shows:")
print("1. ✅ Model provides clear BUY/SELL/HOLD signals")
print("2. ✅ Can construct diversified portfolio from ratings")
print("3. ✅ Weighted portfolio discount indicates expected return")
print()
print("⚠️  Note: Past performance does not guarantee future results")
print("⚠️  This is a demonstration - real trading requires more validation")

print(f"\n{'='*80}")
