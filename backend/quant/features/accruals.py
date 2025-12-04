"""
Accruals Anomaly Factor

Based on Sloan (1996): Stocks with high accruals (low cash quality earnings)
underperform stocks with low accruals (high cash quality earnings).

Formula:
BS_ACC = (ΔCA - ΔCash) - (ΔCL - ΔSTD - ΔITP) - Dep
Scaled_Accruals = BS_ACC / Avg(Total Assets)

Signal: LONG Low Accruals, SHORT High Accruals
"""
import pandas as pd
import numpy as np
from typing import Optional, Dict, Any
from quant.features.base import FeatureGenerator
import logging

logger = logging.getLogger(__name__)


class AccrualsAnomaly(FeatureGenerator):
    """
    Calculates the Balance Sheet Accruals anomaly factor.
    
    Low accruals = High cash quality = Expected outperformance
    High accruals = Low cash quality = Expected underperformance
    """
    
    @property
    def name(self) -> str:
        return "AccrualsAnomaly"
    
    @property
    def description(self) -> str:
        return "Balance Sheet Accruals scaled by average total assets. Lower is better."
    
    def compute(
        self, 
        history: pd.DataFrame, 
        fundamentals: Optional[Dict[str, Any]] = None
    ) -> pd.Series:
        """
        Compute scaled accruals from balance sheet data.
        
        Args:
            history: Price history (not used for this factor)
            fundamentals: Dict containing 'balance_sheet' DataFrame
            
        Returns:
            pd.Series with accruals score (inverted: lower raw = higher score)
        """
        if fundamentals is None:
            return pd.Series([0.0], index=[pd.Timestamp.now()])
        
        balance = fundamentals.get('balance_sheet')
        
        if balance is None or balance.empty:
            return pd.Series([0.0], index=[pd.Timestamp.now()])
        
        try:
            # Get latest two periods for delta calculation
            # yfinance balance sheet columns are dates, most recent first
            if balance.shape[1] < 2:
                return pd.Series([0.0], index=[pd.Timestamp.now()])
            
            # Current period (t) and prior period (t-1)
            current = balance.iloc[:, 0]
            prior = balance.iloc[:, 1]
            
            # Extract values with fallbacks
            def get_val(series, keys, default=0.0):
                for key in keys:
                    if key in series.index:
                        val = series[key]
                        if pd.notna(val):
                            return float(val)
                return default
            
            # Current Assets
            ca_t = get_val(current, ['Current Assets', 'Total Current Assets'])
            ca_t1 = get_val(prior, ['Current Assets', 'Total Current Assets'])
            delta_ca = ca_t - ca_t1
            
            # Cash
            cash_t = get_val(current, ['Cash And Cash Equivalents', 'Cash Cash Equivalents And Short Term Investments'])
            cash_t1 = get_val(prior, ['Cash And Cash Equivalents', 'Cash Cash Equivalents And Short Term Investments'])
            delta_cash = cash_t - cash_t1
            
            # Current Liabilities
            cl_t = get_val(current, ['Current Liabilities', 'Total Current Liabilities'])
            cl_t1 = get_val(prior, ['Current Liabilities', 'Total Current Liabilities'])
            delta_cl = cl_t - cl_t1
            
            # Short-term Debt (in CL)
            std_t = get_val(current, ['Current Debt', 'Current Debt And Capital Lease Obligation'])
            std_t1 = get_val(prior, ['Current Debt', 'Current Debt And Capital Lease Obligation'])
            delta_std = std_t - std_t1
            
            # Income Taxes Payable
            itp_t = get_val(current, ['Income Tax Payable', 'Current Deferred Liabilities'])
            itp_t1 = get_val(prior, ['Income Tax Payable', 'Current Deferred Liabilities'])
            delta_itp = itp_t - itp_t1
            
            # Depreciation (from income statement or cash flow, approximate from balance)
            # If not available, we'll approximate using change in accumulated depreciation
            dep = get_val(current, ['Depreciation And Amortization', 'Depreciation'])
            if dep == 0.0:
                # Fallback: use change in accumulated depreciation
                acc_dep_t = get_val(current, ['Accumulated Depreciation'])
                acc_dep_t1 = get_val(prior, ['Accumulated Depreciation'])
                dep = abs(acc_dep_t - acc_dep_t1)
            
            # Total Assets for scaling
            ta_t = get_val(current, ['Total Assets'])
            ta_t1 = get_val(prior, ['Total Assets'])
            avg_ta = (ta_t + ta_t1) / 2
            
            if avg_ta <= 0:
                return pd.Series([0.0], index=[pd.Timestamp.now()])
            
            # Balance Sheet Accruals Formula
            # BS_ACC = (ΔCA - ΔCash) - (ΔCL - ΔSTD - ΔITP) - Dep
            bs_acc = (delta_ca - delta_cash) - (delta_cl - delta_std - delta_itp) - dep
            
            # Scale by average total assets
            scaled_accruals = bs_acc / avg_ta
            
            # Invert: We want LOW accruals to have HIGH score
            # Typical range is -0.2 to +0.2
            # Multiply by -1 so that low accruals become high scores
            score = -scaled_accruals
            
            logger.debug(f"Accruals: BS_ACC={bs_acc:.2f}, Avg_TA={avg_ta:.2f}, Scaled={scaled_accruals:.4f}, Score={score:.4f}")
            
            return pd.Series([score], index=[pd.Timestamp.now()])
            
        except Exception as e:
            logger.warning(f"Accruals calculation failed: {e}")
            return pd.Series([0.0], index=[pd.Timestamp.now()])
