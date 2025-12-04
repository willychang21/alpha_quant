"""
Comprehensive Model Validation Test
====================================

Tests valuation model across diverse scenarios:
- Market Cap: Mega/Large/Mid/Small
- Sectors: All major sectors
- Growth Stages: High/Moderate/Low/Negative
- Special Cases: High debt, no profit, international, etc.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import yfinance as yf
import pandas as pd
import numpy as np
import logging

from app.engines.valuation import core as valuation
from app.domain import schemas

logging.basicConfig(level=logging.WARNING, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_stock(ticker: str, expected_model: str = None):
    """Test single stock and validate results."""
    print(f"\n{'='*70}")
    print(f"Testing: {ticker}")
    print(f"{'='*70}")
    
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        # Quick validation
        if not info or 'currentPrice' not in info:
            print(f"⚠️  {ticker}: No data available")
            return None
        
        income = stock.quarterly_income_stmt
        balance = stock.quarterly_balance_sheet
        cashflow = stock.quarterly_cashflow
        history = stock.history(period="1y")
        
        # Run valuation
        result = valuation.get_valuation(
            ticker=ticker,
            info=info,
            income=income,
            balance=balance,
            cashflow=cashflow,
            history=history
        )
        
        # Extract results
        model_type = None
        fair_value = None
        
        if result.dcf:
            model_type = "DCF"
            fair_value = result.dcf.sharePrice
        elif result.ddm:
            model_type = "DDM"
            fair_value = result.ddm.fairValue
        elif result.reit:
            model_type = "REIT"
            fair_value = result.reit.fairValue
        
        current_price = info.get('currentPrice', 0)
        sector = info.get('sector', 'Unknown')
        market_cap = info.get('marketCap', 0)
        
        # Validation
        discount = (fair_value - current_price) / current_price if current_price > 0 else 0
        
        # Print summary
        print(f"Name: {info.get('shortName', ticker)}")
        print(f"Sector: {sector}")
        print(f"Market Cap: ${market_cap/1e9:.1f}B")
        print(f"Model: {model_type}")
        print(f"Price: ${current_price:.2f}")
        print(f"Fair Value: ${fair_value:.2f}")
        print(f"Discount: {discount:+.1%}")
        print(f"Rating: {result.rating}")
        
        # Validation checks
        issues = []
        
        if fair_value <= 0:
            issues.append("⚠️  Fair value <= 0")
        
        if abs(discount) > 2.0:  # More than 200% off
            issues.append(f"⚠️  Extreme discount: {discount:+.0%}")
        
        if expected_model and model_type != expected_model:
            issues.append(f"⚠️  Expected {expected_model}, got {model_type}")
        
        if result.quant and result.quant.total < 0:
            issues.append("⚠️  Negative quant score")
        
        if issues:
            print("\nIssues:")
            for issue in issues:
                print(f"  {issue}")
            return {'ticker': ticker, 'status': 'WARNING', 'issues': issues}
        else:
            print("✅ Validation passed")
            return {
                'ticker': ticker,
                'status': 'OK',
                'model': model_type,
                'fair_value': fair_value,
                'discount': discount,
                'rating': result.rating
            }
            
    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return {'ticker': ticker, 'status': 'ERROR', 'error': str(e)}

def run_comprehensive_test():
    """Run comprehensive validation across diverse stocks."""
    
    test_cases = {
        # Mega Cap Tech (High Growth)
        "Mega Cap Tech": ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA'],
        
        # Large Cap Value
        "Large Cap Value": ['JPM', 'BAC', 'WFC', 'XOM', 'CVX', 'JNJ', 'PG'],
        
        # Mid Cap Growth
        "Mid Cap": ['SQ', 'SNAP', 'ROKU', 'ETSY'],
        
        # REITs (Different Sub-Sectors)
        "REITs": ['EQIX', 'PLD', 'AMT', 'SPG', 'O', 'VNO'],
        
        # International (ADRs)
        "International": ['TSM', 'BABA', 'NVO', 'SAP'],
        
        # Cyclical
        "Cyclical": ['CAT', 'DE', 'BA', 'GM'],
        
        # Defensive
        "Defensive": ['KO', 'PEP', 'WMT', 'COST'],
        
        # High Yield
        "High Yield": ['T', 'VZ', 'MO', 'PM'],
        
        # Special Cases
        "Special": ['BRK-B', 'GOOG']  # Berkshire (conglomerate), GOOG (vs GOOGL)
    }
    
    results = []
    errors = []
    warnings = []
    
    print("\n" + "="*70)
    print("COMPREHENSIVE VALUATION MODEL TEST")
    print("="*70)
    
    total_tests = sum(len(tickers) for tickers in test_cases.values())
    current = 0
    
    for category, tickers in test_cases.items():
        print(f"\n\n{'#'*70}")
        print(f"# {category}")
        print(f"{'#'*70}")
        
        for ticker in tickers:
            current += 1
            print(f"\n[{current}/{total_tests}]", end=" ")
            
            result = test_stock(ticker)
            
            if result:
                results.append(result)
                if result['status'] == 'ERROR':
                    errors.append(result)
                elif result['status'] == 'WARNING':
                    warnings.append(result)
    
    # Summary
    print(f"\n\n{'='*70}")
    print("TEST SUMMARY")
    print(f"{'='*70}\n")
    
    total = len(results)
    ok = len([r for r in results if r['status'] == 'OK'])
    warn = len(warnings)
    err = len(errors)
    
    print(f"Total Tests: {total}")
    print(f"✅ Passed: {ok} ({ok/total*100:.1f}%)")
    print(f"⚠️  Warnings: {warn} ({warn/total*100:.1f}%)")
    print(f"❌ Errors: {err} ({err/total*100:.1f}%)")
    
    if errors:
        print(f"\n\n{'='*70}")
        print("ERRORS")
        print(f"{'='*70}")
        for e in errors:
            print(f"\n{e['ticker']}: {e.get('error', 'Unknown error')}")
    
    if warnings:
        print(f"\n\n{'='*70}")
        print("WARNINGS")
        print(f"{'='*70}")
        for w in warnings:
            print(f"\n{w['ticker']}:")
            for issue in w.get('issues', []):
                print(f"  {issue}")
    
    # Save results
    df = pd.DataFrame([r for r in results if r['status'] == 'OK'])
    if not df.empty:
        output_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'comprehensive_test_results.csv')
        df.to_csv(output_path, index=False)
        print(f"\n✅ Results saved to {output_path}")
    
    # Final verdict
    print(f"\n\n{'='*70}")
    if err == 0 and warn < total * 0.1:  # Less than 10% warnings
        print("🎉 MODEL VALIDATION: PASSED ✅")
        print("Model is production-ready!")
    elif err == 0:
        print("⚠️  MODEL VALIDATION: PASSED WITH WARNINGS")
        print(f"Review {warn} warnings before production deployment")
    else:
        print("❌ MODEL VALIDATION: FAILED")
        print(f"Fix {err} critical errors before deployment")
    print("="*70)

if __name__ == "__main__":
    run_comprehensive_test()
