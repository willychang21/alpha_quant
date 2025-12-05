import sys
import os
import numpy as np
import pandas as pd
import pytest

# Add backend to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from quant.features.labeling import get_daily_vol, apply_triple_barrier, get_bins
try:
    from quant.features.meta_labeling import MetaLabeler, prepare_meta_features
    XGBOOST_AVAILABLE = True
except Exception as e:
    XGBOOST_AVAILABLE = False
    print(f"⚠️ XGBoost not available ({e}). Skipping Meta-Labeling test.")

def test_triple_barrier_labeling():
    print("\nTesting Triple Barrier Labeling...")
    
    # 1. Generate Synthetic Price (Random Walk)
    np.random.seed(42)
    n_days = 200
    returns = np.random.normal(0, 0.01, n_days)
    price = 100 * (1 + returns).cumprod()
    close = pd.Series(price, index=pd.date_range('2023-01-01', periods=n_days))
    
    # 2. Get Volatility
    vol = get_daily_vol(close, span0=20)
    
    # 3. Create Events (Signals)
    # Let's say we have a signal every 10 days
    signal_dates = close.index[::10]
    events = pd.DataFrame(index=signal_dates)
    events['t1'] = events.index + pd.Timedelta(days=10) # Vertical barrier 10 days
    events['trgt'] = vol.loc[events.index] # Dynamic width
    events['side'] = 1 # All Long
    
    # 4. Apply Triple Barrier
    # PT = 1x Vol, SL = 1x Vol
    barriers = apply_triple_barrier(close, events, pt_sl=[1, 1])
    
    print("Barriers Hit:\n", barriers.head())
    
    # 5. Get Labels
    labels = get_bins(barriers, close)
    print("Labels:\n", labels['bin'].value_counts())
    
    # Verify we have some hits
    assert not labels.empty
    assert set(labels['bin'].unique()).issubset({-1, 0, 1})
    
    print("✅ Triple Barrier Labeling Passed")
    return close, events, labels, vol

def test_meta_labeling():
    print("\nTesting Meta-Labeling...")
    
    if not XGBOOST_AVAILABLE:
        print("⚠️ Skipping Meta-Labeling test (XGBoost missing).")
        return

    # Reuse data from previous test
    close, events, labels, vol = test_triple_barrier_labeling()
    
    # 1. Prepare Features
    # We need to align features to the events
    # Signals score: Random for this test
    signals = pd.Series(np.random.randn(len(events)), index=events.index)
    
    X = prepare_meta_features(
        market_data=pd.DataFrame({'Close': close}),
        signals=signals,
        volatility=vol.loc[events.index]
    )
    
    # 2. Prepare Target
    # Meta-label: 1 if bin == 1 (Profit), 0 otherwise
    y = (labels['bin'] == 1).astype(int)
    
    # Align y with X (in case X dropped NaNs)
    y = y.loc[X.index]
    
    # 3. Train Model
    ml = MetaLabeler()
    ml.train(X, y)
    
    # 4. Predict
    probs = ml.predict_proba(X)
    print("Probabilities:\n", probs.head())
    
    assert len(probs) == len(X)
    assert probs.min() >= 0 and probs.max() <= 1
    
    # 5. Feature Importance
    imp = ml.get_feature_importance()
    print("Feature Importance:\n", imp)
    
    assert not imp.empty
    
    print("✅ Meta-Labeling Passed")

if __name__ == "__main__":
    test_triple_barrier_labeling()
    test_meta_labeling()
