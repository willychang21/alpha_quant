"""
Simplified Valuation Backtest - Uses Existing Service Layer
============================================================

Quick backtest without needing historical data APIs.
Uses current data as proxy to demonstrate methodology.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

# Import our modules
from app.engines.valuation import core as valuation
from app.domain import schemas

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def simple_backtest(ticker: str):
    """
    Simple backtest: Compare current Fair Value vs Price.
    Calculate implied forward return.
    """
    logger.info(f"\n{'='*70}")
    logger.info(f"BACKTEST: {ticker}")
    logger.info(f"{'='*70}\n")
    
    try:
        # Fetch data
        stock = yf.Ticker(ticker)
        info = stock.info
        income = stock.quarterly_income_stmt
        balance = stock.quarterly_balance_sheet
        cashflow = stock.quarterly_cashflow
        history = stock.history(period="1y")
        
        # Get current price
        current_price = info.get('currentPrice') or info.get('regularMarketPrice')
        
        if not current_price:
            logger.error(f"Could not get current price for {ticker}")
            return None
        
        # Run valuation
        result = valuation.get_valuation(
            ticker=ticker,
            info=info,
            income=income,
            balance=balance,
            cashflow=cashflow,
            history=history
        )
        
        # Extract fair value
        fair_value = None
        model_type = None
        
        if result.dcf:
            fair_value = result.dcf.sharePrice
            model_type = "DCF"
        elif result.ddm:
            fair_value = result.ddm.fairValue
            model_type = "DDM"
        elif result.reit:
            fair_value = result.reit.fairValue
            model_type = "REIT (FFO)"
        
        if not fair_value or fair_value <= 0:
            logger.error(f"Invalid fair value for {ticker}")
            return None
        
        # Calculate metrics
        discount = (fair_value - current_price) / current_price
        rating = result.rating
        
        # Quant metrics
        quant_total = result.quant.total if result.quant else 0
        
        # Risk metrics
        beta = result.risk.beta if result.risk else info.get('beta', 1.0)
        sharpe = result.risk.sharpeRatio if result.risk else 0
        
        # Print results
        print(f"\n{ticker} - {info.get('shortName', ticker)}")
        print(f"Sector: {info.get('sector', 'N/A')}")
        print("-" * 70)
        print(f"Model Type: {model_type}")
        print(f"Current Price: ${current_price:.2f}")
        print(f"Fair Value: ${fair_value:.2f}")
        print(f"Discount/Premium: {discount:+.1%}")
        print(f"Rating: {rating}")
        print(f"\nQuant Score: {quant_total:.1f}/100")
        print(f"Beta: {beta:.2f}")
        print(f"Sharpe Ratio: {sharpe:.2f}")
        
        # WACC Details
        if result.waccDetails:
            print(f"\nWACC Details:")
            print(f"  WACC: {result.waccDetails.wacc:.2%}")
            print(f"  Cost of Equity: {result.waccDetails.costOfEquity:.2%}")
            print(f"  Beta (Adjusted): {result.waccDetails.betaAdjusted:.2f}")
        
        # Fair Value Range
        if result.fairValueRange and len(result.fairValueRange) >= 2:
            print(f"\nFair Value Range:")
            print(f"  Bear: ${result.fairValueRange[0]:.2f}")
            print(f"  Base: ${fair_value:.2f}")
            if len(result.fairValueRange) > 2:
                print(f"  Bull: ${result.fairValueRange[2]:.2f}")
        
        # Interpretation
        print(f"\nInterpretation:")
        if discount > 0.20:
            print(f"  ✅ STRONG UNDERVALUED - Potential Upside: {discount:.1%}")
        elif discount > 0.10:
            print(f"  ✅ UNDERVALUED - Potential Upside: {discount:.1%}")
        elif discount < -0.10:
            print(f"  ⚠️  OVERVALUED - Model suggests {abs(discount):.1%} downside")
        else:
            print(f"  ➡️  FAIRLY VALUED - Within {abs(discount):.1%} of fair value")
        
        print("\n" + "="*70)
        
        return {
            'ticker': ticker,
            'current_price': current_price,
            'fair_value': fair_value,
            'discount': discount,
            'rating': rating,
            'quant_score': quant_total,
            'beta': beta,
            'sharpe': sharpe,
            'model': model_type
        }
        
    except Exception as e:
        logger.error(f"Error in backtest for {ticker}: {e}")
        import traceback
        traceback.print_exc()
        return None

def batch_backtest(tickers: list):
    """Run backtest on multiple tickers and summarize."""
    results = []
    
    for ticker in tickers:
        result = simple_backtest(ticker)
        if result:
            results.append(result)
    
    if not results:
        print("\nNo valid results")
        return
    
    # Summary
    df = pd.DataFrame(results)
    
    print(f"\n{'='*70}")
    print("BACKTEST SUMMARY")
    print(f"{'='*70}\n")
    
    print("Statistics:")
    print(f"  Total Tickers: {len(df)}")
    print(f"  Avg Discount: {df['discount'].mean():+.1%}")
    print(f"  Avg Quant Score: {df['quant_score'].mean():.1f}/100")
    
    print(f"\nRating Distribution:")
    print(df['rating'].value_counts())
    
    print(f"\nTop 5 Undervalued (by Discount):")
    top5 = df.nlargest(5, 'discount')[['ticker', 'current_price', 'fair_value', 'discount', 'rating']]
    print(top5.to_string(index=False))
    
    print(f"\nTop 5 by Quant Score:")
    top5_quant = df.nlargest(5, 'quant_score')[['ticker', 'quant_score', 'discount', 'rating']]
    print(top5_quant.to_string(index=False))
    
    # Save results
    output_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'backtest_results.csv')
    df.to_csv(output_path, index=False)
    print(f"\n✅ Results saved to {output_path}")
    
    print("\n" + "="*70)

if __name__ == "__main__":
    # Test portfolio
    tickers = [
        # Tech Giants
        'AAPL', 'MSFT', 'GOOGL', 'NVDA', 'META',
        # Value Stocks
        'JPM', 'BAC', 'WFC',  # Banks
        # REITs
        'PLD', 'AMT', 'EQIX',  # Real Estate
        # Others
        'JNJ', 'PG', 'KO'
    ]
    
    print("\n🚀 Starting Valuation Model Backtest...")
    print(f"Testing {len(tickers)} tickers\n")
    
    batch_backtest(tickers)
    
    print("\n✅ Backtest Complete!")
