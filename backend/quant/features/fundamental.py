import pandas as pd
import numpy as np
from typing import Optional
from quant.features.base import FeatureGenerator

class QualityFactor(FeatureGenerator):
    @property
    def name(self) -> str:
        return "Quality"
        
    @property
    def description(self) -> str:
        return "Composite score of ROE, Gross Margin, and Debt/Equity."
        
    def compute(self, history: pd.DataFrame, fundamentals: Optional[pd.DataFrame] = None) -> pd.Series:
        # Note: In a real system, 'fundamentals' would be a DataFrame of historical fundamental data.
        # For this refactor, we are accepting a single row/dict disguised as a DataFrame or just using the 'info' dict passed in context if we were using a context object.
        # However, the interface expects DataFrames.
        # To keep it simple for Stage 2 MVP, we'll assume 'fundamentals' contains the latest snapshot in the last row,
        # or we'll adapt the legacy logic which used a dictionary 'info'.
        
        # ADAPTER HACK: If fundamentals is a Series (single row), convert to DF.
        if isinstance(fundamentals, pd.Series):
            fundamentals = fundamentals.to_frame().T
            
        if fundamentals is None or fundamentals.empty:
            return pd.Series([50.0])
            
        # Extract latest values
        latest = fundamentals.iloc[-1]
        
        roe = latest.get('returnOnEquity', 0)
        gross_margin = latest.get('grossMargins', 0)
        debt_to_equity = latest.get('debtToEquity', 100) / 100.0
        
        # Scoring Logic (Linear Interpolation)
        s_roe = np.interp(roe, [0.05, 0.25], [0, 100])
        s_gm = np.interp(gross_margin, [0.20, 0.60], [0, 100])
        s_de = 100 - np.interp(debt_to_equity, [0.5, 2.5], [0, 100])
        
        score = (s_roe * 0.4) + (s_gm * 0.3) + (s_de * 0.3)
        return pd.Series([score], index=[fundamentals.index[-1]])

class ValueFactor(FeatureGenerator):
    @property
    def name(self) -> str:
        return "Value"
        
    @property
    def description(self) -> str:
        return "Composite score of P/E and P/B ratios."
        
    def compute(self, history: pd.DataFrame, fundamentals: Optional[pd.DataFrame] = None) -> pd.Series:
        if fundamentals is None or fundamentals.empty:
            return pd.Series([50.0])
            
        latest = fundamentals.iloc[-1]
        
        pe = latest.get('trailingPE', 30)
        if not pe or pe <= 0: pe = 50
        
        pb = latest.get('priceToBook', 5)
        if not pb or pb <= 0: pb = 10
        
        s_pe = 100 - np.interp(pe, [10, 50], [0, 100])
        s_pb = 100 - np.interp(pb, [1.0, 10.0], [0, 100])
        
        score = (s_pe * 0.6) + (s_pb * 0.4)
        return pd.Series([score], index=[fundamentals.index[-1]])

class GrowthFactor(FeatureGenerator):
    @property
    def name(self) -> str:
        return "Growth"
        
    @property
    def description(self) -> str:
        return "Composite score of Revenue and Earnings Growth."
        
    def compute(self, history: pd.DataFrame, fundamentals: Optional[pd.DataFrame] = None) -> pd.Series:
        if fundamentals is None or fundamentals.empty:
            return pd.Series([50.0])
            
        latest = fundamentals.iloc[-1]
        
        rev_growth = latest.get('revenueGrowth', 0)
        earn_growth = latest.get('earningsGrowth', 0)
        
        s_rev = np.interp(rev_growth, [0.0, 0.30], [0, 100])
        s_earn = np.interp(earn_growth, [0.0, 0.30], [0, 100])
        
        score = (s_rev * 0.5) + (s_earn * 0.5)
        return pd.Series([score], index=[fundamentals.index[-1]])
