"""Quality Minus Junk (QMJ) Factor Plugin.

Academic reference: Asness, Frazzini & Pedersen (2019)
"Quality Minus Junk"

Quality stocks (profitable, growing, safe) outperform junk stocks.
"""

import logging
from typing import Any, Dict

import numpy as np
import pandas as pd

from quant.core.interfaces import FactorBase, PluginMetadata
from quant.core.registry import register_factor

logger = logging.getLogger(__name__)


@register_factor("QMJ")
class QMJFactor(FactorBase):
    """Quality Minus Junk factor.

    Composite quality score based on:
    - Profitability (ROE, gross margin)
    - Growth (earnings growth)
    - Safety (low leverage)

    Attributes:
        params: Configuration parameters.
        profitability_weight: Weight for profitability component.
        growth_weight: Weight for growth component.
        safety_weight: Weight for safety component.
    """

    def __init__(self, params: Dict[str, Any] | None = None) -> None:
        """Initialize QMJ factor.

        Args:
            params: Configuration dict with optional keys:
                - profitability_weight: Weight for profitability (default: 0.5)
                - growth_weight: Weight for growth (default: 0.25)
                - safety_weight: Weight for safety (default: 0.25)
        """
        self.params = params or {}
        self.profitability_weight = self.params.get("profitability_weight", 0.5)
        self.growth_weight = self.params.get("growth_weight", 0.25)
        self.safety_weight = self.params.get("safety_weight", 0.25)

    @property
    def metadata(self) -> PluginMetadata:
        """Return plugin metadata."""
        return PluginMetadata(
            name="QMJ",
            description="Quality Minus Junk - composite quality factor",
            version="1.0.0",
            author="DCA Quant",
            category="quality",
            parameters={
                "profitability_weight": "Weight for profitability (default: 0.5)",
                "growth_weight": "Weight for earnings growth (default: 0.25)",
                "safety_weight": "Weight for low leverage (default: 0.25)",
            },
        )

    def compute(self, data: pd.DataFrame) -> pd.Series:
        """Compute QMJ factor values.

        Args:
            data: DataFrame with 'ticker' and fundamental columns:
                  - 'roe': Return on Equity
                  - 'gross_margin': Gross profit margin
                  - 'earnings_growth': YoY earnings growth
                  - 'debt_to_equity': Total debt / equity ratio

        Returns:
            Series of QMJ values indexed by ticker.
            Higher values indicate higher quality.
        """
        results: Dict[str, float] = {}

        # Get most recent data per ticker
        if "date" in data.columns:
            latest = data.sort_values("date").groupby("ticker").last().reset_index()
        else:
            latest = data

        for _, row in latest.iterrows():
            ticker = str(row.get("ticker", row.name))

            # Extract fundamental metrics (use defaults if missing)
            roe = row.get("roe", np.nan)
            gross_margin = row.get("gross_margin", np.nan)
            earnings_growth = row.get("earnings_growth", np.nan)
            debt_to_equity = row.get("debt_to_equity", np.nan)

            # Compute component scores
            # Profitability: higher ROE and gross margin is better
            profitability_scores = []
            if not pd.isna(roe):
                profitability_scores.append(roe)
            if not pd.isna(gross_margin):
                profitability_scores.append(gross_margin)

            profitability = (
                np.mean(profitability_scores) if profitability_scores else np.nan
            )

            # Growth: higher earnings growth is better
            growth = earnings_growth if not pd.isna(earnings_growth) else np.nan

            # Safety: lower debt-to-equity is better (invert)
            if not pd.isna(debt_to_equity) and debt_to_equity >= 0:
                safety = 1.0 / (1.0 + debt_to_equity)  # Transform to 0-1 range
            else:
                safety = np.nan

            # Compute weighted composite
            components = []
            weights = []

            if not pd.isna(profitability):
                components.append(profitability)
                weights.append(self.profitability_weight)

            if not pd.isna(growth):
                components.append(growth)
                weights.append(self.growth_weight)

            if not pd.isna(safety):
                components.append(safety)
                weights.append(self.safety_weight)

            if components and sum(weights) > 0:
                # Normalize weights
                total_weight = sum(weights)
                qmj = sum(c * w for c, w in zip(components, weights)) / total_weight
            else:
                qmj = np.nan

            results[ticker] = qmj

        return pd.Series(results, name="QMJ")

    def validate_input(self, data: pd.DataFrame) -> bool:
        """Validate input data has at least ticker column."""
        return "ticker" in data.columns or data.index.name == "ticker"
