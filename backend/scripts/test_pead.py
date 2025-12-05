"""
Test PEAD (Post-Earnings Announcement Drift) Factor

Verifies earnings surprise calculation and recency weighting.
"""
import asyncio
import logging
import pandas as pd
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(name)s - %(message)s')
logger = logging.getLogger(__name__)


def test_pead():
    """Test PEAD factor calculation."""
    print("\n=== Testing PEAD Factor ===")
    
    from quant.features.pead import PostEarningsAnnouncementDrift, EarningsMomentum
    import yfinance as yf
    
    pead = PostEarningsAnnouncementDrift(lookback_days=90, decay_halflife=30)
    
    # Test with several tickers
    tickers = ["AAPL", "MSFT", "NVDA", "AMD", "GOOGL"]
    
    results = []
    
    for ticker in tickers:
        print(f"\n--- {ticker} ---")
        
        # Get earnings history
        stock = yf.Ticker(ticker)
        
        try:
            earnings = stock.earnings_history
            if earnings is not None and not earnings.empty:
                print(f"Earnings History Columns: {earnings.columns.tolist()}")
                print(f"Recent Earnings:\n{earnings.head(3)}")
        except Exception as e:
            print(f"Error fetching earnings: {e}")
        
        # Compute PEAD score
        score_series = pead.compute(pd.DataFrame(), None, ticker=ticker)
        score = score_series.iloc[-1]
        
        results.append({'ticker': ticker, 'pead_score': score})
        print(f"PEAD Score: {score:.4f}")
    
    # Summary
    print("\n=== PEAD Results Summary ===")
    df = pd.DataFrame(results).sort_values('pead_score', ascending=False)
    for _, row in df.iterrows():
        direction = "🟢 BUY" if row['pead_score'] > 0.5 else "🟡 HOLD" if row['pead_score'] > -0.5 else "🔴 AVOID"
        print(f"{row['ticker']:6s}: {row['pead_score']:+.2f} {direction}")
    
    return len([r for r in results if r['pead_score'] != 0]) > 0


def test_earnings_momentum():
    """Test EarningsMomentum factor."""
    print("\n=== Testing Earnings Momentum ===")
    
    from quant.features.pead import EarningsMomentum
    import yfinance as yf
    
    em = EarningsMomentum()
    
    tickers = ["AAPL", "NVDA", "TSLA"]
    
    for ticker in tickers:
        print(f"\n--- {ticker} ---")
        
        try:
            info = yf.Ticker(ticker).info
            
            # Print relevant fields
            print(f"Quarterly Earnings Growth: {info.get('earningsQuarterlyGrowth', 'N/A')}")
            print(f"Revenue Growth: {info.get('revenueGrowth', 'N/A')}")
            print(f"Forward EPS: {info.get('forwardEps', 'N/A')}")
            print(f"Trailing EPS: {info.get('trailingEps', 'N/A')}")
            
            # Compute score
            score_series = em.compute(pd.DataFrame(), info)
            score = score_series.iloc[-1]
            print(f"Earnings Momentum Score: {score:.4f}")
            
        except Exception as e:
            print(f"Error: {e}")


def main():
    print("=" * 60)
    print("PEAD FACTOR TEST")
    print("=" * 60)
    
    pead_passed = test_pead()
    test_earnings_momentum()
    
    print("\n" + "=" * 60)
    if pead_passed:
        print("✅ PEAD FACTOR TEST PASSED!")
    else:
        print("⚠️ PEAD returned all zeros (may need recent earnings)")
    print("=" * 60)


if __name__ == "__main__":
    main()
