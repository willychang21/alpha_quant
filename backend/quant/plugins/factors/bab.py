"""Betting Against Beta (BAB) Factor Plugin.

Academic reference: Frazzini & Pedersen (2014)
"Betting Against Beta"

Low-beta stocks tend to outperform high-beta stocks on a risk-adjusted basis.
"""

import logging
from typing import Any, Dict

import numpy as np
import pandas as pd

from quant.core.interfaces import FactorBase, PluginMetadata
from quant.core.registry import register_factor

logger = logging.getLogger(__name__)


@register_factor("BAB")
class BABFactor(FactorBase):
    """Betting Against Beta factor.

    Computes a factor score based on the inverse of market beta.
    Lower-beta stocks get higher BAB scores.

    Attributes:
        params: Configuration parameters.
        beta_lookback: Lookback period for beta estimation.
    """

    def __init__(self, params: Dict[str, Any] | None = None) -> None:
        """Initialize BAB factor.

        Args:
            params: Configuration dict with optional keys:
                - beta_lookback: Beta estimation lookback (default: 252)
                - market_col: Column name for market returns (default: None, uses mean)
        """
        self.params = params or {}
        self.beta_lookback = self.params.get("beta_lookback", 252)
        self.market_col = self.params.get("market_col", None)

    @property
    def metadata(self) -> PluginMetadata:
        """Return plugin metadata."""
        return PluginMetadata(
            name="BAB",
            description="Betting Against Beta - favors low-beta stocks",
            version="1.0.0",
            author="DCA Quant",
            category="risk",
            parameters={
                "beta_lookback": "Beta estimation lookback in days (default: 252)",
                "market_col": "Column for market returns (default: uses mean)",
            },
        )

    def compute(self, data: pd.DataFrame) -> pd.Series:
        """Compute BAB factor values.

        Args:
            data: DataFrame with 'ticker', 'date', 'close' columns.
                  Optionally 'market_return' for market benchmark.

        Returns:
            Series of BAB values indexed by ticker.
            Higher values indicate lower beta (more desirable for BAB).
        """
        results: Dict[str, float] = {}

        # First, compute returns for all tickers
        ticker_returns: Dict[str, np.ndarray] = {}

        for ticker, group in data.groupby("ticker"):
            group = group.sort_values("date")
            prices = group["close"].values

            if len(prices) < 2:
                continue

            returns = np.diff(prices) / prices[:-1]
            ticker_returns[str(ticker)] = returns

        if not ticker_returns:
            return pd.Series(dtype=float, name="BAB")

        # Compute market return as equal-weighted mean of all stocks
        # (simplified - in practice would use actual market index)
        min_len = min(len(r) for r in ticker_returns.values())
        aligned_returns = np.array(
            [r[-min_len:] for r in ticker_returns.values() if len(r) >= min_len]
        )

        if len(aligned_returns) == 0:
            return pd.Series(dtype=float, name="BAB")

        market_return = np.mean(aligned_returns, axis=0)
        market_var = np.var(market_return)

        # Compute beta for each ticker
        for ticker, returns in ticker_returns.items():
            if len(returns) < self.beta_lookback:
                results[ticker] = np.nan
                continue

            # Use last beta_lookback days
            recent_returns = returns[-self.beta_lookback :]
            recent_market = market_return[-self.beta_lookback :]

            if len(recent_returns) != len(recent_market):
                min_common = min(len(recent_returns), len(recent_market))
                recent_returns = recent_returns[-min_common:]
                recent_market = recent_market[-min_common:]

            # Beta = Cov(r_i, r_m) / Var(r_m)
            cov = np.cov(recent_returns, recent_market)[0, 1]
            var = np.var(recent_market)

            if var > 0:
                beta = cov / var
            else:
                beta = 1.0

            # BAB score is negative of beta (low beta = high score)
            # Centered around 1.0
            results[ticker] = 1.0 - beta

        return pd.Series(results, name="BAB")

    def validate_input(self, data: pd.DataFrame) -> bool:
        """Validate input data has required columns."""
        required = {"ticker", "date", "close"}
        return required.issubset(data.columns)
