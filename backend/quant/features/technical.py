import pandas as pd
import numpy as np
from typing import Optional
from quant.features.base import FeatureGenerator

class MomentumFactor(FeatureGenerator):
    @property
    def name(self) -> str:
        return "Momentum"
        
    @property
    def description(self) -> str:
        return "Price proximity to 52-week high."
        
    def compute(self, history: pd.DataFrame, fundamentals: Optional[pd.DataFrame] = None) -> pd.Series:
        if history.empty:
            return pd.Series([50.0])
            
        current_price = history['Close'].iloc[-1]
        
        # Ideally 52w high comes from history, but fallback to fundamentals if provided
        high_52w = current_price
        if len(history) >= 252:
            high_52w = history['Close'].rolling(window=252).max().iloc[-1]
        elif fundamentals is not None and not fundamentals.empty:
            high_52w = fundamentals.iloc[-1].get('fiftyTwoWeekHigh', current_price)
            
        if high_52w <= 0:
            return pd.Series([50.0], index=[history.index[-1]])
            
        dist_to_high = current_price / high_52w
        score = np.interp(dist_to_high, [0.70, 1.00], [0, 100])
        
        return pd.Series([score], index=[history.index[-1]])

class RiskMetricsGenerator(FeatureGenerator):
    @property
    def name(self) -> str:
        return "RiskMetrics"
        
    @property
    def description(self) -> str:
        return "Volatility, Sharpe Ratio, Max Drawdown."
        
    def compute(self, history: pd.DataFrame, fundamentals: Optional[pd.DataFrame] = None) -> pd.Series:
        # Returns a Series where values are not scalar but a dict (or we could split into multiple features)
        # For compatibility with legacy, we'll return a Series of dicts (hacky but works for MVP)
        
        if history.empty or len(history) < 30:
            return pd.Series([{
                'volatility': 0.0,
                'sharpe': 0.0,
                'max_drawdown': 0.0
            }], index=[pd.Timestamp.now()])
            
        returns = history['Close'].pct_change().dropna()
        
        if len(returns) == 0:
            return pd.Series([{
                'volatility': 0.0,
                'sharpe': 0.0,
                'max_drawdown': 0.0
            }], index=[history.index[-1]])
            
        # Volatility
        volatility = returns.std() * np.sqrt(252)
        
        # Sharpe
        rf_daily = 0.042 / 252
        excess_returns = returns - rf_daily
        sharpe = (excess_returns.mean() / returns.std()) * np.sqrt(252) if returns.std() > 0 else 0
        
        # Max Drawdown
        cumulative = (1 + returns).cumprod()
        peak = cumulative.cummax()
        drawdown = (cumulative - peak) / peak
        max_drawdown = drawdown.min()
        
        return pd.Series([{
            'volatility': volatility,
            'sharpe': sharpe,
            'max_drawdown': max_drawdown
        }], index=[history.index[-1]])
