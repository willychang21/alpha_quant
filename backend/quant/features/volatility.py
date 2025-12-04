import pandas as pd
import numpy as np
from typing import Optional
from quant.features.base import FeatureGenerator

class VolatilityScaledMomentum(FeatureGenerator):
    @property
    def name(self) -> str:
        return "VolatilityScaledMomentum"
        
    @property
    def description(self) -> str:
        return "12-month cumulative return divided by 1-year realized volatility."
        
    def compute(self, history: pd.DataFrame, fundamentals: Optional[pd.DataFrame] = None) -> pd.Series:
        """
        Compute Volatility Scaled Momentum.
        Signal = (R_12m / Vol_12m)
        """
        if history.empty or len(history) < 252:
            return pd.Series([0.0], index=[pd.Timestamp.now()])
            
        # 1. Calculate Daily Returns
        returns = history['Close'].pct_change().fillna(0)
        
        # 2. Calculate 12-Month Return (approx 252 trading days)
        # We use a rolling window to get the value at the latest date
        # Momentum is often 12-1 (excluding last month), but we'll do simple 12m for now as per plan
        ret_12m = history['Close'].pct_change(periods=252)
        
        # 3. Calculate 1-Year Volatility (Annualized)
        vol_1y = returns.rolling(window=252).std() * np.sqrt(252)
        
        # 4. Calculate Ratio
        # Avoid division by zero
        score = ret_12m / vol_1y.replace(0, np.nan)
        
        # Return the latest value
        latest_score = score.iloc[-1]
        
        if pd.isna(latest_score):
            return pd.Series([0.0], index=[history.index[-1]])
            
        return pd.Series([latest_score], index=[history.index[-1]])
