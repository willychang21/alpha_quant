from quant.features.pipeline import get_pipeline, add_feature
import numpy as np

# 1. Define Features
@add_feature("returns", dependencies=["prices"])
def compute_returns(prices):
    print("  -> Executing compute_returns logic")
    return [p2/p1 - 1 for p1, p2 in zip(prices[:-1], prices[1:])]

@add_feature("volatility", dependencies=["returns"])
def compute_volatility(returns):
    print("  -> Executing compute_volatility logic")
    return np.std(returns)

@add_feature("momentum", dependencies=["returns"])
def compute_momentum(returns):
    print("  -> Executing compute_momentum logic")
    return sum(returns)

def test_pipeline():
    pipeline = get_pipeline()
    
    # Context (Raw Data)
    context = {
        "prices": [100, 101, 102, 101, 103, 105]
    }
    
    print("--- Run 1: Cold Cache ---")
    results = pipeline.compute(["volatility", "momentum"], context)
    print("Results:", results)
    
    print("\n--- Run 2: Warm Cache ---")
    # Should not print "Executing..." messages
    results_cached = pipeline.compute(["volatility", "momentum"], context)
    print("Results (Cached):", results_cached)

if __name__ == "__main__":
    test_pipeline()
