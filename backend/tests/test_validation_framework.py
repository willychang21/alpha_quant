import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add backend to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from quant.backtest.validation import PurgedWalkForwardCV
from quant.backtest.statistics import deflated_sharpe_ratio, sharpe_ratio

def test_purged_cv():
    print("="*50)
    print("Testing PurgedWalkForwardCV")
    print("="*50)
    
    # Generate synthetic data
    dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
    X = pd.DataFrame(np.random.randn(100, 2), index=dates, columns=['f1', 'f2'])
    
    # Prediction times: Label is realized 5 days later
    # t -> t+5
    pred_times = pd.Series(dates + timedelta(days=5), index=dates)
    
    cv = PurgedWalkForwardCV(n_splits=3, embargo_pct=0.01)
    
    fold = 1
    for train_idx, test_idx in cv.split(X, pred_times=pred_times):
        print(f"\nFold {fold}:")
        print(f"  Train: {len(train_idx)} samples. Range: {train_idx.min()} - {train_idx.max()}")
        print(f"  Test:  {len(test_idx)} samples. Range: {test_idx.min()} - {test_idx.max()}")
        
        # Verify Purging
        # Check if any training sample overlaps with test set
        test_start = pred_times.index[test_idx[0]]
        test_end = pred_times.iloc[test_idx[-1]]
        
        train_times = pred_times.index[train_idx]
        train_pred_times = pred_times.iloc[train_idx]
        
        # Overlap condition:
        # Train ends after Test starts AND Train starts before Test ends
        overlaps = (train_pred_times > test_start) & (train_times < test_end)
        
        if overlaps.any():
            print("  ❌ FAILURE: Overlap detected!")
            print(f"     {overlaps.sum()} overlapping samples found.")
        else:
            print("  ✅ SUCCESS: No overlaps (Purging worked)")
            
        fold += 1

def test_dsr():
    print("\n" + "="*50)
    print("Testing Deflated Sharpe Ratio")
    print("="*50)
    
    # Generate returns with SR ~ 0.5 (Borderline)
    np.random.seed(42)
    # Mean 1.5% annualized, Vol 20% annualized -> SR ~ 0.075
    returns = pd.Series(np.random.normal(0.015/252, 0.20/np.sqrt(252), 1000))
    
    sr = sharpe_ratio(returns)
    print(f"Standard Sharpe Ratio: {sr:.4f}")
    
    # Test DSR with 1 trial
    dsr_1 = deflated_sharpe_ratio(returns, trials=1)
    print(f"DSR (1 trial): {dsr_1:.4f} (Should be close to PSR)")
    
    # Test DSR with 100 trials
    dsr_100 = deflated_sharpe_ratio(returns, trials=100)
    print(f"DSR (100 trials): {dsr_100:.4f}")
    
    if dsr_100 < dsr_1:
        print("✅ SUCCESS: DSR decreases with more trials.")
    else:
        print("❌ FAILURE: DSR did not decrease.")

if __name__ == "__main__":
    test_purged_cv()
    test_dsr()
