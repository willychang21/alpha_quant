import pandas as pd
import numpy as np
import yfinance as yf
from typing import Optional
from quant.features.base import FeatureGenerator

class BettingAgainstBeta(FeatureGenerator):
    _spy_returns_cache = None
    _spy_cache_date = None

    @property
    def name(self) -> str:
        return "BettingAgainstBeta"
        
    @property
    def description(self) -> str:
        return "Low-beta anomaly factor. Returns inverse of 1-year Beta vs SPY."
        
    def _get_spy_returns(self) -> pd.Series:
        """
        Fetch and cache SPY returns.
        """
        today = pd.Timestamp.now().date()
        if self._spy_returns_cache is not None and self._spy_cache_date == today:
            return self._spy_returns_cache
            
        # Fetch SPY history (2 years to be safe)
        spy = yf.download("SPY", period="2y", progress=False)
        if isinstance(spy.columns, pd.MultiIndex):
            spy = spy['Close'] # Handle multi-index if needed, but usually 'Close' is top level or 'SPY' is top level
            # Actually yf.download for single ticker might return MultiIndex if threads=True or auto_adjust=False?
            # Let's handle generic case
            if 'Close' in spy.columns:
                 close = spy['Close']
            elif 'SPY' in spy.columns:
                 close = spy['SPY']['Close']
            else:
                 # Fallback, maybe it's just the dataframe
                 close = spy
        else:
            close = spy['Close']
            
        returns = close.pct_change().fillna(0)
        
        # Cache it
        BettingAgainstBeta._spy_returns_cache = returns
        BettingAgainstBeta._spy_cache_date = today
        return returns

    def compute(self, history: pd.DataFrame, fundamentals: Optional[pd.DataFrame] = None, benchmark_history: Optional[pd.DataFrame] = None) -> pd.Series:
        """
        Compute Beta and return inverted score.
        """
        if history.empty or len(history) < 252:
            return pd.Series([0.0], index=[pd.Timestamp.now()])
            
        # 1. Asset Returns
        asset_returns = history['Close'].pct_change().fillna(0)
        
        # 2. Benchmark Returns
        if benchmark_history is not None:
            if benchmark_history.empty:
                 # SPY fetch failed upstream, return neutral score
                 return pd.Series([0.0], index=[history.index[-1]])
            
            # Check if it's a Series or DataFrame
            if isinstance(benchmark_history, pd.Series):
                spy_close = benchmark_history
            elif 'Close' in benchmark_history.columns:
                spy_close = benchmark_history['Close']
            else:
                # Assume the first column is Close if 'Close' is missing
                spy_close = benchmark_history.iloc[:, 0]
                
            spy_returns = spy_close.pct_change().fillna(0)
        else:
            # Fallback (only if benchmark_history was NOT passed at all)
            spy_returns = self._get_spy_returns()
        
        # Ensure spy_returns is a Series (squeeze if DataFrame)
        if isinstance(spy_returns, pd.DataFrame):
            spy_returns = spy_returns.squeeze()

        # Align dates
        common_index = asset_returns.index.intersection(spy_returns.index)
        if len(common_index) < 200: # Need enough overlapping days
             return pd.Series([0.0], index=[history.index[-1]])
             
        asset_aligned = asset_returns.loc[common_index]
        spy_aligned = spy_returns.loc[common_index]
        
        # 3. Calculate Beta (Covariance / Variance)
        if len(asset_aligned) > 252:
            asset_aligned = asset_aligned.iloc[-252:]
            spy_aligned = spy_aligned.iloc[-252:]
            
        # Ensure 1D numpy arrays
        asset_vals = asset_aligned.values.flatten()
        spy_vals = spy_aligned.values.flatten()
            
        covariance = np.cov(asset_vals, spy_vals)[0][1]
        variance = np.var(spy_vals)
        
        if variance == 0:
            beta = 1.0
        else:
            beta = covariance / variance
            
        # 4. Invert Beta (Low Beta = High Score)
        # We can just return -Beta as the raw score. 
        # Ranking/Z-scoring later will handle the rest.
        # Lower beta => Higher -Beta => Higher Rank.
        score = -beta
        
        return pd.Series([score], index=[history.index[-1]])
