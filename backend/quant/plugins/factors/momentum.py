"""Momentum Factor Plugin.

Basic price momentum factor: proximity to 52-week high.
Standard momentum signal used in academic factor literature.
"""

import logging
from typing import Any, Dict

import numpy as np
import pandas as pd

from quant.core.interfaces import FactorBase, PluginMetadata
from quant.core.registry import register_factor

logger = logging.getLogger(__name__)


@register_factor("Momentum")
class MomentumFactor(FactorBase):
    """Basic Momentum factor.

    Computes momentum as the ratio of current price to 52-week high.
    Stocks closer to their highs have stronger momentum.

    Attributes:
        params: Configuration parameters.
        lookback: Lookback period for high calculation.
    """

    def __init__(self, params: Dict[str, Any] | None = None) -> None:
        """Initialize Momentum factor.

        Args:
            params: Configuration dict with optional keys:
                - lookback: Days for 52-week high calculation (default: 252)
        """
        self.params = params or {}
        self.lookback = self.params.get("lookback", 252)

    @property
    def metadata(self) -> PluginMetadata:
        """Return plugin metadata."""
        return PluginMetadata(
            name="Momentum",
            description="Price momentum - proximity to 52-week high",
            version="1.0.0",
            author="DCA Quant",
            category="momentum",
            parameters={
                "lookback": "Days for high calculation (default: 252)",
            },
        )

    def compute(self, data: pd.DataFrame) -> pd.Series:
        """Compute Momentum factor values.

        Args:
            data: DataFrame with 'ticker', 'date', 'close' columns.

        Returns:
            Series of momentum scores (0-100) indexed by ticker.
            100 = at 52-week high, lower values = further from high.
        """
        results: Dict[str, float] = {}

        for ticker, group in data.groupby("ticker"):
            group = group.sort_values("date")
            prices = group["close"].values

            if len(prices) == 0:
                results[str(ticker)] = np.nan
                continue

            current_price = prices[-1]

            # Calculate 52-week high
            if len(prices) >= self.lookback:
                high_52w = np.max(prices[-self.lookback :])
            else:
                high_52w = np.max(prices)

            if high_52w <= 0:
                results[str(ticker)] = 50.0
                continue

            # Distance to high as ratio
            dist_to_high = current_price / high_52w

            # Scale to 0-100 (0.70 ratio = 0 score, 1.0 ratio = 100 score)
            score = np.interp(dist_to_high, [0.70, 1.00], [0.0, 100.0])

            results[str(ticker)] = float(score)

        return pd.Series(results, name="Momentum")

    def validate_input(self, data: pd.DataFrame) -> bool:
        """Validate input data has required columns."""
        required = {"ticker", "date", "close"}
        return required.issubset(data.columns)
