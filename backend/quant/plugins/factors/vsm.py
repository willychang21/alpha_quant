"""Volatility Scaled Momentum (VSM) Factor Plugin.

Academic reference: Barroso & Santa-Clara (2015)
"Momentum has its moments"

Scales raw momentum by inverse volatility to reduce crash risk.
"""

import logging
from typing import Any, Dict

import numpy as np
import pandas as pd

from quant.core.interfaces import FactorBase, PluginMetadata
from quant.core.registry import register_factor

logger = logging.getLogger(__name__)


@register_factor("VSM")
class VSMFactor(FactorBase):
    """Volatility Scaled Momentum factor.

    Scales raw momentum by inverse volatility to reduce crash risk.
    Higher VSM indicates stronger risk-adjusted momentum.

    Attributes:
        params: Configuration parameters.
        lookback: Momentum lookback period in days.
        vol_window: Volatility estimation window in days.
        target_vol: Target volatility for scaling.
    """

    def __init__(self, params: Dict[str, Any] | None = None) -> None:
        """Initialize VSM factor.

        Args:
            params: Configuration dict with optional keys:
                - lookback: Momentum lookback period (default: 252)
                - vol_window: Volatility window (default: 60)
                - target_vol: Target volatility (default: 0.12)
        """
        self.params = params or {}
        self.lookback = self.params.get("lookback", 252)
        self.vol_window = self.params.get("vol_window", 60)
        self.target_vol = self.params.get("target_vol", 0.12)

    @property
    def metadata(self) -> PluginMetadata:
        """Return plugin metadata."""
        return PluginMetadata(
            name="VSM",
            description="Volatility Scaled Momentum - momentum adjusted for risk",
            version="1.0.0",
            author="DCA Quant",
            category="momentum",
            parameters={
                "lookback": "Momentum lookback period in trading days (default: 252)",
                "vol_window": "Volatility estimation window in days (default: 60)",
                "target_vol": "Target volatility for scaling (default: 0.12)",
            },
        )

    def compute(self, data: pd.DataFrame) -> pd.Series:
        """Compute VSM factor values.

        Args:
            data: DataFrame with 'ticker', 'date', 'close' columns.
                  Should contain historical price data for each ticker.

        Returns:
            Series of VSM values indexed by ticker.
        """
        results: Dict[str, float] = {}

        # Group by ticker and compute for each
        for ticker, group in data.groupby("ticker"):
            group = group.sort_values("date")

            if len(group) < self.lookback:
                results[str(ticker)] = np.nan
                continue

            prices = group["close"].values

            # Compute returns
            returns = np.diff(prices) / prices[:-1]

            if len(returns) < self.lookback:
                results[str(ticker)] = np.nan
                continue

            # Raw momentum (12-month return, skip last month to avoid reversal)
            # Typical: return from t-252 to t-21
            if len(prices) >= self.lookback:
                raw_mom = (prices[-21] / prices[-self.lookback]) - 1
            else:
                raw_mom = (prices[-1] / prices[0]) - 1

            # Realized volatility (recent window)
            recent_returns = returns[-self.vol_window :]
            realized_vol = np.std(recent_returns) * np.sqrt(252)

            # Scale momentum by inverse volatility
            if realized_vol > 0:
                vol_scalar = self.target_vol / realized_vol
                vsm = raw_mom * vol_scalar
            else:
                vsm = raw_mom

            results[str(ticker)] = vsm

        return pd.Series(results, name="VSM")

    def validate_input(self, data: pd.DataFrame) -> bool:
        """Validate input data has required columns."""
        required = {"ticker", "date", "close"}
        return required.issubset(data.columns)
