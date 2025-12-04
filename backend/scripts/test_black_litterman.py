"""
Test Black-Litterman Integration

Verifies that BlackLittermanModel works correctly with market caps and Z-scores.
"""
import asyncio
import logging
import pandas as pd
import numpy as np
import yfinance as yf

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(name)s - %(message)s')
logger = logging.getLogger(__name__)


def test_black_litterman():
    """Test Black-Litterman optimizer standalone."""
    print("\n=== Testing Black-Litterman Optimizer ===")
    
    from quant.portfolio.advanced_optimizers import BlackLittermanModel
    
    # Sample tickers
    tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "META"]
    
    # 1. Get market caps
    print("Fetching market caps...")
    market_caps = {}
    volatilities = {}
    
    for ticker in tickers:
        try:
            info = yf.Ticker(ticker).info
            mkt_cap = info.get('marketCap', 0)
            if mkt_cap and mkt_cap > 0:
                market_caps[ticker] = mkt_cap
        except Exception as e:
            print(f"Error fetching {ticker}: {e}")
    
    print(f"Market Caps: {market_caps}")
    
    # 2. Get returns and covariance
    print("Fetching price history...")
    data = yf.download(tickers, period="1y", progress=False)
    
    if isinstance(data.columns, pd.MultiIndex):
        prices = data['Close']
    else:
        prices = data
    
    returns = prices.pct_change().dropna()
    cov_matrix = returns.cov() * 252  # Annualized
    
    # Volatilities
    for ticker in tickers:
        if ticker in returns.columns:
            vol = returns[ticker].std() * np.sqrt(252)
            volatilities[ticker] = vol
    
    print(f"Volatilities: {volatilities}")
    
    # 3. Mock Z-scores (from ranking signals)
    z_scores = {
        "AAPL": 1.5,   # Strong buy
        "MSFT": 1.2,   # Buy
        "GOOGL": 0.8,  # Hold/light buy
        "AMZN": -0.3,  # Neutral
        "META": 1.8    # Strong buy
    }
    
    print(f"Z-Scores (alpha signals): {z_scores}")
    
    # 4. Run Black-Litterman
    bl = BlackLittermanModel(tau=0.05, risk_aversion=2.5)
    
    market_caps_series = pd.Series(market_caps)
    weights = bl.optimize(cov_matrix, market_caps_series, z_scores, volatilities, ic=0.05)
    
    print("\n=== Black-Litterman Results ===")
    if weights:
        for ticker, weight in sorted(weights.items(), key=lambda x: x[1], reverse=True):
            print(f"  {ticker}: {weight:.2%}")
        
        total = sum(weights.values())
        print(f"\nTotal Weight: {total:.2%}")
        
        return True
    else:
        print("Black-Litterman optimization failed!")
        return False


def main():
    print("=" * 60)
    print("BLACK-LITTERMAN INTEGRATION TEST")
    print("=" * 60)
    
    passed = test_black_litterman()
    
    print("\n" + "=" * 60)
    if passed:
        print("✅ BLACK-LITTERMAN TEST PASSED!")
    else:
        print("❌ BLACK-LITTERMAN TEST FAILED!")
    print("=" * 60)
    
    return passed


if __name__ == "__main__":
    main()
