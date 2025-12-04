import pandas as pd
import numpy as np
from typing import List, Dict, Optional

class FactorPipeline:
    """
    Processes raw factor values into clean, tradable signals.
    """
    
    @staticmethod
    def winsorize(series: pd.Series, limits: List[float] = [0.01, 0.01]) -> pd.Series:
        """
        Clip outliers at the given percentiles (e.g., 1% and 99%).
        """
        if series.empty: return series
        return series.clip(lower=series.quantile(limits[0]), upper=series.quantile(1 - limits[1]))
        
    @staticmethod
    def z_score(series: pd.Series) -> pd.Series:
        """
        Normalize to Mean=0, Std=1.
        """
        if series.empty: return series
        std = series.std()
        if std == 0 or pd.isna(std):
            return pd.Series(0.0, index=series.index)
        return (series - series.mean()) / std
        
    @staticmethod
    def neutralize(series: pd.Series, groups: pd.Series) -> pd.Series:
        """
        Subtract group (sector) means from the scores.
        """
        if series.empty or groups.empty: return series
        
        # Create a DataFrame to align series and groups
        df = pd.DataFrame({'score': series, 'group': groups})
        
        # Calculate group means
        group_means = df.groupby('group')['score'].transform('mean')
        
        # Subtract mean
        return df['score'] - group_means

    @classmethod
    def process_factors(cls, df: pd.DataFrame, sector_col: str = 'sector') -> pd.DataFrame:
        """
        Main pipeline execution.
        Expects a DataFrame where columns are raw factors and rows are tickers.
        """
        processed = df.copy()
        
        # 1. Handle QMJ Composite if components exist
        # We assume columns 'roe', 'gross_margin', 'debt_to_equity' exist if QMJ was computed
        if 'roe' in processed.columns and 'debt_to_equity' in processed.columns:
            # Z-score components first
            z_roe = cls.z_score(cls.winsorize(processed['roe']))
            z_gm = cls.z_score(cls.winsorize(processed['gross_margin']))
            z_de = cls.z_score(cls.winsorize(processed['debt_to_equity']))
            
            # Combine: Profitability + Safety (Low Debt)
            # Higher ROE/GM is good (+), Higher Debt is bad (-)
            processed['quality'] = (z_roe + z_gm - z_de) / 3.0
            
        # 2. Process Top-Level Factors
        factors_to_process = ['momentum', 'volatility_scaled_momentum', 'betting_against_beta', 'quality', 'value', 'upside']
        
        for factor in factors_to_process:
            if factor in processed.columns:
                # A. Winsorize
                processed[factor] = cls.winsorize(processed[factor])
                
                # B. Log Transform (optional, for skewed factors like Market Cap or Value)
                # processed[factor] = np.log1p(processed[factor]) 
                
                # C. Z-Score (Global)
                processed[f'z_{factor}'] = cls.z_score(processed[factor])
                
                # D. Neutralize (Sector)
                if sector_col in processed.columns:
                    processed[f'z_{factor}_neutral'] = cls.neutralize(processed[f'z_{factor}'], processed[sector_col])
                    
        return processed
