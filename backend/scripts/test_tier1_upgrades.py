"""
Test Script for Tier-1 Quant Upgrades

Verifies:
1. Accruals Anomaly Factor
2. IVOL Factor
3. HMM Regime Detection
4. HRP Optimizer
"""
import asyncio
import logging
import pandas as pd
import numpy as np
from datetime import date

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(name)s - %(message)s')
logger = logging.getLogger(__name__)

async def test_accruals():
    """Test Accruals Anomaly factor."""
    print("\n=== Testing Accruals Anomaly ===")
    from quant.features.accruals import AccrualsAnomaly
    from core.adapters.yfinance_provider import YFinanceProvider
    
    provider = YFinanceProvider()
    accruals = AccrualsAnomaly()
    
    # Test with AAPL
    data = await provider.get_ticker_data("AAPL")
    fundamentals = {
        'balance_sheet': data.get('balance')
    }
    
    result = accruals.compute(pd.DataFrame(), fundamentals)
    print(f"AAPL Accruals Score: {result.iloc[-1]:.4f}")
    return result.iloc[-1] != 0.0


async def test_ivol():
    """Test IVOL factor."""
    print("\n=== Testing IVOL ===")
    from quant.features.ivol import IdiosyncraticVolatility
    from core.adapters.yfinance_provider import YFinanceProvider
    
    provider = YFinanceProvider()
    ivol = IdiosyncraticVolatility(lookback=21)
    
    # Get stock history
    history = await provider.get_history("AAPL")
    
    # Test without benchmark (uses total volatility as proxy)
    result = ivol.compute(history, benchmark_history=None)
    print(f"AAPL IVOL Score (no benchmark): {result.iloc[-1]:.4f}")
    return result.iloc[-1] != 0.0


def test_hmm():
    """Test HMM Regime Detection."""
    print("\n=== Testing HMM Regime Detection ===")
    from quant.regime.hmm import RegimeDetector, DynamicFactorWeights
    import yfinance as yf
    
    # Get SPY returns for training
    spy = yf.download("SPY", period="2y", progress=False)
    if isinstance(spy.columns, pd.MultiIndex):
        returns = spy['Close']['SPY'].pct_change().dropna()
    else:
        returns = spy['Close'].pct_change().dropna()
    
    # Fit HMM
    detector = RegimeDetector(n_states=2, lookback=252)
    detector.fit(returns)
    
    # Predict current regime
    regime, probs = detector.predict_regime(returns.iloc[-30:])
    print(f"Current Regime: {regime}, Probabilities: {probs}")
    
    # Test dynamic weights
    weights = DynamicFactorWeights.get_weights(regime)
    print(f"Recommended Weights for {regime}: {weights}")
    
    return regime in ["Bull", "Bear", "Transition", "Unknown"]


def test_hrp():
    """Test HRP Optimizer."""
    print("\n=== Testing HRP Optimizer ===")
    from quant.portfolio.advanced_optimizers import HRPOptimizer
    import yfinance as yf
    
    # Get returns for a few tickers
    tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "META"]
    data = yf.download(tickers, period="1y", progress=False)
    
    # Extract close prices
    if isinstance(data.columns, pd.MultiIndex):
        prices = data['Close']
    else:
        prices = data
    
    returns = prices.pct_change().dropna()
    
    # Run HRP
    hrp = HRPOptimizer()
    weights = hrp.optimize(returns)
    
    print("HRP Weights:")
    for ticker, weight in sorted(weights.items(), key=lambda x: x[1], reverse=True):
        print(f"  {ticker}: {weight:.2%}")
    
    return sum(weights.values()) > 0.99  # Should sum to ~1


async def main():
    print("=" * 60)
    print("TIER-1 QUANT UPGRADE VERIFICATION")
    print("=" * 60)
    
    results = {}
    
    # Test Accruals
    try:
        results['Accruals'] = await test_accruals()
    except Exception as e:
        logger.error(f"Accruals test failed: {e}")
        results['Accruals'] = False
    
    # Test IVOL
    try:
        results['IVOL'] = await test_ivol()
    except Exception as e:
        logger.error(f"IVOL test failed: {e}")
        results['IVOL'] = False
    
    # Test HMM
    try:
        results['HMM'] = test_hmm()
    except Exception as e:
        logger.error(f"HMM test failed: {e}")
        results['HMM'] = False
    
    # Test HRP
    try:
        results['HRP'] = test_hrp()
    except Exception as e:
        logger.error(f"HRP test failed: {e}")
        results['HRP'] = False
    
    # Summary
    print("\n" + "=" * 60)
    print("VERIFICATION RESULTS")
    print("=" * 60)
    for name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{name}: {status}")
    
    all_passed = all(results.values())
    print("\n" + ("ALL TESTS PASSED! ✅" if all_passed else "SOME TESTS FAILED ❌"))
    
    return all_passed


if __name__ == "__main__":
    asyncio.run(main())
