"""
Idiosyncratic Volatility (IVOL) Factor

Based on Ang et al. (2006): Stocks with high idiosyncratic volatility
exhibit significantly lower future returns (the "IVOL Puzzle").

Formula:
1. Regress stock returns against Fama-French 3-Factor model
2. IVOL = std(residuals) over rolling window (21 days)

Signal: LONG Low IVOL, SHORT High IVOL
"""
import pandas as pd
import numpy as np
from typing import Optional
from quant.features.base import FeatureGenerator
import logging

logger = logging.getLogger(__name__)


class IdiosyncraticVolatility(FeatureGenerator):
    """
    Calculates Idiosyncratic Volatility using Fama-French 3-Factor residuals.
    
    Low IVOL = Expected outperformance
    High IVOL = Expected underperformance (overpriced lottery stocks)
    """
    
    def __init__(self, lookback: int = 21):
        """
        Args:
            lookback: Rolling window for IVOL calculation (default: 21 trading days)
        """
        self.lookback = lookback
        self._market_data = None
    
    @property
    def name(self) -> str:
        return "IdiosyncraticVolatility"
    
    @property
    def description(self) -> str:
        return "Standard deviation of Fama-French 3-Factor residuals. Lower is better."
    
    def compute(
        self, 
        history: pd.DataFrame, 
        benchmark_history: Optional[pd.Series] = None
    ) -> pd.Series:
        """
        Compute IVOL from price history.
        
        Args:
            history: OHLCV DataFrame for the stock
            benchmark_history: Market returns (SPY) for beta calculation
            
        Returns:
            pd.Series with IVOL score (inverted: lower IVOL = higher score)
        """
        if history.empty or len(history) < self.lookback + 10:
            return pd.Series([0.0], index=[pd.Timestamp.now()])
        
        try:
            # Calculate daily returns
            if 'Close' in history.columns:
                stock_returns = history['Close'].pct_change().dropna()
            else:
                stock_returns = history.iloc[:, 0].pct_change().dropna()
            
            if len(stock_returns) < self.lookback:
                return pd.Series([0.0], index=[pd.Timestamp.now()])
            
            # Simple IVOL: Use residuals from market model (CAPM approximation)
            # Full FF3 would require SMB/HML data from Kenneth French library
            
            has_benchmark = benchmark_history is not None and (
                isinstance(benchmark_history, pd.Series) and len(benchmark_history) > 0
            )
            
            if has_benchmark:
                # Align dates
                market_returns = benchmark_history.pct_change().dropna()
                
                # Find common dates
                common_idx = stock_returns.index.intersection(market_returns.index)
                
                if len(common_idx) < self.lookback:
                    # Fallback to total volatility
                    ivol = stock_returns.iloc[-self.lookback:].std()
                else:
                    stock_aligned = stock_returns.loc[common_idx]
                    market_aligned = market_returns.loc[common_idx]
                    
                    # Rolling regression for beta and residuals
                    # Use the last `lookback` days
                    recent_stock = stock_aligned.iloc[-self.lookback:]
                    recent_market = market_aligned.iloc[-self.lookback:]
                    
                    # Simple OLS: R_stock = alpha + beta * R_market + epsilon
                    X = np.column_stack([np.ones(len(recent_market)), recent_market.values])
                    y = recent_stock.values
                    
                    try:
                        # Least squares
                        coeffs, residuals, rank, s = np.linalg.lstsq(X, y, rcond=None)
                        
                        # Calculate residuals manually
                        predicted = X @ coeffs
                        epsilon = y - predicted
                        
                        # IVOL = std(residuals)
                        ivol = np.std(epsilon)
                        
                    except np.linalg.LinAlgError:
                        # Fallback to total volatility
                        ivol = recent_stock.std()
            else:
                # No benchmark, use total volatility as proxy
                ivol = stock_returns.iloc[-self.lookback:].std()
            
            # Annualize (optional, for comparability)
            ivol_annual = ivol * np.sqrt(252)
            
            # Invert: Low IVOL should have HIGH score
            # Typical IVOL range: 0.01 to 0.10 (daily)
            # We'll use negative IVOL so that low IVOL = high score
            score = -ivol_annual
            
            logger.debug(f"IVOL: Daily={ivol:.4f}, Annual={ivol_annual:.4f}, Score={score:.4f}")
            
            return pd.Series([score], index=[history.index[-1]])
            
        except Exception as e:
            logger.warning(f"IVOL calculation failed: {e}")
            return pd.Series([0.0], index=[pd.Timestamp.now()])


class AmihudIlliquidity(FeatureGenerator):
    """
    Amihud Illiquidity Ratio: Price impact per dollar of volume.
    
    High illiquidity = Expected premium (compensated risk)
    
    Formula: ILLIQ = (1/D) * Σ(|R_d| / VOL_d)
    """
    
    def __init__(self, lookback: int = 21):
        self.lookback = lookback
    
    @property
    def name(self) -> str:
        return "AmihudIlliquidity"
    
    @property
    def description(self) -> str:
        return "Average price impact per dollar traded. Higher = more illiquid = premium."
    
    def compute(
        self, 
        history: pd.DataFrame, 
        fundamentals: Optional[dict] = None
    ) -> pd.Series:
        """
        Compute Amihud Illiquidity ratio.
        
        Args:
            history: OHLCV DataFrame (must include 'Volume' and 'Close')
            
        Returns:
            pd.Series with illiquidity score
        """
        if history.empty or len(history) < self.lookback:
            return pd.Series([0.0], index=[pd.Timestamp.now()])
        
        try:
            # Need Close and Volume
            if 'Close' not in history.columns or 'Volume' not in history.columns:
                return pd.Series([0.0], index=[pd.Timestamp.now()])
            
            recent = history.iloc[-self.lookback:]
            
            # Daily returns (absolute)
            returns_abs = recent['Close'].pct_change().abs()
            
            # Dollar volume
            dollar_volume = recent['Close'] * recent['Volume']
            
            # Avoid division by zero
            dollar_volume = dollar_volume.replace(0, np.nan)
            
            # Amihud ratio for each day
            daily_illiq = returns_abs / dollar_volume
            
            # Average over the lookback period
            illiq = daily_illiq.mean()
            
            if pd.isna(illiq) or np.isinf(illiq):
                return pd.Series([0.0], index=[pd.Timestamp.now()])
            
            # Scale for readability (multiply by 1e6)
            illiq_scaled = illiq * 1e6
            
            logger.debug(f"Amihud ILLIQ: {illiq_scaled:.4f}")
            
            return pd.Series([illiq_scaled], index=[history.index[-1]])
            
        except Exception as e:
            logger.warning(f"Amihud calculation failed: {e}")
            return pd.Series([0.0], index=[pd.Timestamp.now()])
