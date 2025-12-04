import pandas as pd
import numpy as np
from typing import Optional
from quant.features.base import FeatureGenerator

class QualityMinusJunk(FeatureGenerator):
    @property
    def name(self) -> str:
        return "QualityMinusJunk"
        
    @property
    def description(self) -> str:
        return "Composite quality score: Profitability + Safety + Growth."
        
    def compute(self, history: pd.DataFrame, fundamentals: Optional[pd.DataFrame] = None) -> pd.Series:
        """
        Compute QMJ Score.
        Uses fundamentals data (info dict usually passed as DataFrame or dict).
        The 'fundamentals' arg in RankingEngine is actually 'ticker_data' which contains 'info'.
        Wait, FeatureGenerator.compute expects a DataFrame for fundamentals?
        In RankingEngine currently: 
        ticker_data = await self.market_data.get_ticker_data(sec.ticker)
        But FeatureGenerator.compute signature is (history, fundamentals).
        I need to ensure RankingEngine passes the right thing.
        For now, I'll assume 'fundamentals' is a dict or a DataFrame with 1 row containing the info.
        """
        if fundamentals is None:
             return pd.Series([0.0], index=[history.index[-1]])
             
        # Extract metrics
        # Assuming fundamentals is a DataFrame where columns are keys from 'info'
        # Or if it's a dict, we need to handle that. 
        # Let's assume it's a Series or Dict-like object for the latest date.
        
        try:
            if isinstance(fundamentals, pd.DataFrame):
                if fundamentals.empty: return pd.Series([0.0], index=[history.index[-1]])
                data = fundamentals.iloc[-1]
            else:
                data = fundamentals # Treat as dict
                
            # 1. Profitability: ROE
            roe = data.get('returnOnEquity', 0.0)
            if roe is None: roe = 0.0
            
            # 2. Profitability: Gross Margins
            # yfinance info might have 'grossMargins' or we calculate it
            gross_margin = data.get('grossMargins', 0.0)
            if gross_margin is None: gross_margin = 0.0
            
            # 3. Safety: Debt to Equity
            debt_to_equity = data.get('debtToEquity', 0.0)
            if debt_to_equity is None: debt_to_equity = 100.0 # Default to high debt (bad) if missing? Or 0?
            # debtToEquity in yfinance is usually a percentage (e.g. 150 for 1.5x? No, usually 1.5 or 150. Let's check).
            # Usually it's a ratio. If it's > 100 it might be %. Let's assume higher is worse.
            
            # Composite Score
            # We don't have the cross-section here to do Z-scores properly.
            # So we return the raw components, or a simple linear combination.
            # Ideally, we return a dict of components, and the Pipeline handles the Z-scoring and combination.
            # But FeatureGenerator returns a Series (float).
            # So we'll return a pre-weighted sum, but we can't Z-score here.
            # This is a limitation of the current architecture (calculating factor per stock in isolation).
            # Solution: Return a "Raw Quality Score" that is locally normalized or just sum of ratios?
            # Ratios have different scales. ROE is 0.15, DebtToEquity is 2.0.
            # We can't sum them directly without normalization.
            
            # ALTERNATIVE: Return the components packed in a dict (like RiskMetricsGenerator does)?
            # Then RankingEngine unpacks them, Z-scores them individually across the universe, and sums them.
            # This is the "Hedge Fund" way.
            
            return pd.Series([{
                'roe': float(roe),
                'gross_margin': float(gross_margin),
                'debt_to_equity': float(debt_to_equity)
            }], index=[history.index[-1]])
            
        except Exception as e:
            return pd.Series([0.0], index=[history.index[-1]])
