import sys
import os
import pandas as pd
import logging

# Add backend to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from quant.model_registry.registry import ModelRegistry
from quant.features.fundamental import QualityFactor, ValueFactor
from quant.features.technical import MomentumFactor

def test_model_registry():
    print("Testing Model Registry...", flush=True)
    registry = ModelRegistry()
    models = registry.list_models()
    
    if len(models) > 0:
        print(f"Found {len(models)} registered models.", flush=True)
        for m in models:
            print(f" - {m.model_id} v{m.version} ({m.type})", flush=True)
        print("Model Registry Test Passed!", flush=True)
    else:
        print("No models found in registry.", flush=True)
        print("Model Registry Test Failed!", flush=True)

def test_features():
    print("\nTesting Feature Engineering...", flush=True)
    
    # Mock Data
    history = pd.DataFrame({
        'Close': [100, 105, 110, 108, 112]
    }, index=pd.date_range(start='2023-01-01', periods=5))
    
    fundamentals = pd.DataFrame({
        'returnOnEquity': [0.20],
        'grossMargins': [0.50],
        'debtToEquity': [80],
        'trailingPE': [20],
        'priceToBook': [3.0],
        'fiftyTwoWeekHigh': [120]
    }, index=[history.index[-1]])
    
    # Test Quality
    quality = QualityFactor()
    q_score = quality.compute(history, fundamentals)
    print(f"Quality Score: {q_score.iloc[0]:.2f}", flush=True)
    assert q_score.iloc[0] > 0
    
    # Test Value
    value = ValueFactor()
    v_score = value.compute(history, fundamentals)
    print(f"Value Score: {v_score.iloc[0]:.2f}", flush=True)
    assert v_score.iloc[0] > 0
    
    # Test Momentum
    momentum = MomentumFactor()
    m_score = momentum.compute(history, fundamentals)
    print(f"Momentum Score: {m_score.iloc[0]:.2f}", flush=True)
    assert m_score.iloc[0] > 0
    
    print("Feature Engineering Test Passed!", flush=True)

if __name__ == "__main__":
    test_model_registry()
    test_features()
