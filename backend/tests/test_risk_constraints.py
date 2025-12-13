"""Property-based tests for Risk Constraints.

Tests constraint checking correctness (Properties 15-19).
Uses Hypothesis for property-based testing.

Properties tested:
- Property 15: MaxWeight Constraint Correctness
- Property 16: Sector Constraint Correctness
- Property 17: Beta Constraint Correctness
- Property 18: Multiple Constraint Application
- Property 19: Constraint Violation Detection
"""

import sys
from pathlib import Path
from typing import Dict, List

import pandas as pd
import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from quant.core.registry import registry
from quant.plugins.risk_models.max_weight import MaxWeightConstraint
from quant.plugins.risk_models.sector import SectorConstraint
from quant.plugins.risk_models.beta import BetaConstraint


# =============================================================================
# Hypothesis Configuration
# =============================================================================

settings.register_profile("ci", max_examples=100, deadline=None)
settings.register_profile("dev", max_examples=30, deadline=None)
settings.load_profile("dev")


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture(autouse=True)
def ensure_plugins_registered():
    """Ensure plugins are registered for tests."""
    # Import plugins to trigger registration
    from quant.plugins.risk_models import max_weight, sector, beta
    yield


# =============================================================================
# Property 15: MaxWeight Constraint Correctness
# =============================================================================


class TestMaxWeightConstraint:
    """Tests for MaxWeight constraint."""

    @given(
        max_weight=st.floats(min_value=0.05, max_value=0.50),
        weights=st.lists(
            st.floats(min_value=0.0, max_value=1.0),
            min_size=2,
            max_size=10
        )
    )
    @settings(max_examples=50)
    def test_property15_maxweight_detects_violations(
        self, max_weight: float, weights: List[float]
    ) -> None:
        """
        **Feature: registry-pattern-refactoring, Property 15: MaxWeight Constraint Correctness**

        For any portfolio weights and max_weight threshold, MaxWeightConstraint.check_constraints()
        SHALL return False if and only if any weight exceeds the threshold.
        """
        # Normalize weights to sum to 1
        total = sum(weights)
        if total <= 0:
            assume(False)
        
        normalized = [w / total for w in weights]
        tickers = [f"STOCK{i}" for i in range(len(normalized))]
        weight_series = pd.Series(normalized, index=tickers)

        constraint = MaxWeightConstraint(params={"max_weight": max_weight})
        is_valid, msg = constraint.check_constraints(weight_series)

        # Check if any weight exceeds threshold
        actual_max = max(normalized)
        expected_valid = actual_max <= max_weight

        assert is_valid == expected_valid, (
            f"Max weight {actual_max:.4f}, threshold {max_weight:.4f}, "
            f"expected valid={expected_valid}, got {is_valid}"
        )

    def test_property15_exact_threshold_is_valid(self) -> None:
        """Weights exactly at threshold should be valid."""
        weights = pd.Series([0.10, 0.10, 0.80], index=["A", "B", "C"])
        constraint = MaxWeightConstraint(params={"max_weight": 0.80})
        is_valid, _ = constraint.check_constraints(weights)
        assert is_valid

    def test_property15_just_over_threshold_is_invalid(self) -> None:
        """Weights just over threshold should be invalid."""
        weights = pd.Series([0.10, 0.10, 0.81], index=["A", "B", "C"])
        constraint = MaxWeightConstraint(params={"max_weight": 0.80})
        is_valid, msg = constraint.check_constraints(weights)
        assert not is_valid
        assert msg is not None
        assert "exceeded" in msg.lower() or "C" in msg


# =============================================================================
# Property 16: Sector Constraint Correctness
# =============================================================================


class TestSectorConstraint:
    """Tests for Sector constraint."""

    @given(
        max_sector=st.floats(min_value=0.1, max_value=0.8),
        n_stocks=st.integers(min_value=3, max_value=10)
    )
    @settings(max_examples=30)
    def test_property16_sector_detects_violations(
        self, max_sector: float, n_stocks: int
    ) -> None:
        """
        **Feature: registry-pattern-refactoring, Property 16: Sector Constraint Correctness**

        For any portfolio weights, sector mapping, and sector limit, SectorConstraint.check_constraints()
        SHALL return False if and only if any sector's total weight exceeds the limit.
        """
        # Create weights summing to 1
        tickers = [f"STOCK{i}" for i in range(n_stocks)]
        weights = pd.Series(1.0 / n_stocks, index=tickers)

        # Assign all to same sector (should violate if n_stocks * (1/n_stocks) = 1.0 > max)
        sectors = {t: "Technology" for t in tickers}

        constraint = SectorConstraint(params={"max_sector_weight": max_sector})
        is_valid, msg = constraint.check_constraints(weights, sectors=sectors)

        # All in one sector = 1.0 weight
        expected_valid = 1.0 <= max_sector

        assert is_valid == expected_valid

    def test_property16_diversified_sectors_is_valid(self) -> None:
        """Properly diversified sectors should be valid."""
        weights = pd.Series([0.25, 0.25, 0.25, 0.25], index=["A", "B", "C", "D"])
        sectors = {
            "A": "Tech",
            "B": "Finance",
            "C": "Healthcare",
            "D": "Energy"
        }
        constraint = SectorConstraint(params={"max_sector_weight": 0.30})
        is_valid, _ = constraint.check_constraints(weights, sectors=sectors)
        assert is_valid

    def test_property16_concentrated_sector_is_invalid(self) -> None:
        """Concentrated sector should be invalid."""
        weights = pd.Series([0.20, 0.20, 0.20, 0.40], index=["A", "B", "C", "D"])
        sectors = {
            "A": "Tech",
            "B": "Tech",
            "C": "Finance",
            "D": "Tech"
        }
        # Tech = 0.20 + 0.20 + 0.40 = 0.80
        constraint = SectorConstraint(params={"max_sector_weight": 0.50})
        is_valid, msg = constraint.check_constraints(weights, sectors=sectors)
        assert not is_valid
        assert msg is not None


# =============================================================================
# Property 17: Beta Constraint Correctness
# =============================================================================


class TestBetaConstraint:
    """Tests for Beta constraint."""

    @given(
        min_beta=st.floats(min_value=0.5, max_value=0.9),
        max_beta=st.floats(min_value=1.1, max_value=1.5),
        stock_betas=st.lists(
            st.floats(min_value=0.3, max_value=2.0),
            min_size=2,
            max_size=5
        )
    )
    @settings(max_examples=30)
    def test_property17_beta_detects_violations(
        self, min_beta: float, max_beta: float, stock_betas: List[float]
    ) -> None:
        """
        **Feature: registry-pattern-refactoring, Property 17: Beta Constraint Correctness**

        For any portfolio weights, asset betas, and beta bounds, BetaConstraint.check_constraints()
        SHALL return False if and only if portfolio beta is outside bounds.
        """
        n = len(stock_betas)
        tickers = [f"STOCK{i}" for i in range(n)]
        weights = pd.Series(1.0 / n, index=tickers)  # Equal weight
        betas = dict(zip(tickers, stock_betas))

        constraint = BetaConstraint(params={"min_beta": min_beta, "max_beta": max_beta})
        is_valid, msg = constraint.check_constraints(weights, betas=betas)

        # Calculate expected portfolio beta
        portfolio_beta = sum(stock_betas) / n
        expected_valid = min_beta <= portfolio_beta <= max_beta

        assert is_valid == expected_valid, (
            f"Portfolio beta {portfolio_beta:.3f}, bounds [{min_beta:.3f}, {max_beta:.3f}], "
            f"expected valid={expected_valid}, got {is_valid}"
        )

    def test_property17_neutral_portfolio_in_bounds(self) -> None:
        """Market-neutral portfolio should be in typical bounds."""
        weights = pd.Series([0.5, 0.5], index=["HIGH", "LOW"])
        betas = {"HIGH": 1.5, "LOW": 0.5}  # Average = 1.0
        constraint = BetaConstraint(params={"min_beta": 0.8, "max_beta": 1.2})
        is_valid, _ = constraint.check_constraints(weights, betas=betas)
        assert is_valid


# =============================================================================
# Property 18: Multiple Constraint Application
# =============================================================================


class TestMultipleConstraints:
    """Tests for multiple constraint application."""

    def test_property18_all_constraints_checked(self) -> None:
        """
        **Feature: registry-pattern-refactoring, Property 18: Multiple Constraint Application**

        For any set of N configured risk models, the optimizer SHALL check all N constraints,
        and optimization SHALL fail if any constraint returns False.
        """
        weights = pd.Series([0.50, 0.50], index=["A", "B"])
        sectors = {"A": "Tech", "B": "Tech"}
        betas = {"A": 2.0, "B": 2.0}  # High beta portfolio

        constraints = [
            MaxWeightConstraint(params={"max_weight": 0.60}),  # Should pass
            SectorConstraint(params={"max_sector_weight": 0.40}),  # Should fail (1.0 > 0.4)
            BetaConstraint(params={"min_beta": 0.8, "max_beta": 1.2}),  # Should fail (2.0 > 1.2)
        ]

        results = []
        for c in constraints:
            is_valid, msg = c.check_constraints(weights, sectors=sectors, betas=betas)
            results.append(is_valid)

        # At least one should fail
        assert not all(results), "Expected at least one constraint to fail"
        # Specifically sector and beta should fail
        assert results[0] is True  # MaxWeight passes
        assert results[1] is False  # Sector fails
        assert results[2] is False  # Beta fails


# =============================================================================
# Property 19: Constraint Violation Detection
# =============================================================================


class TestConstraintViolationDetection:
    """Tests for constraint violation error messages."""

    def test_property19_maxweight_provides_details(self) -> None:
        """
        **Feature: registry-pattern-refactoring, Property 19: Constraint Violation Detection**

        For any weights that violate a constraint, check_constraints() SHALL return
        (False, error_message) where error_message describes the violation.
        """
        weights = pd.Series([0.15, 0.85], index=["SMALL", "LARGE"])
        constraint = MaxWeightConstraint(params={"max_weight": 0.50})
        is_valid, msg = constraint.check_constraints(weights)

        assert not is_valid
        assert msg is not None
        assert "LARGE" in msg or "85" in msg or "exceeded" in msg.lower()

    def test_property19_sector_provides_details(self) -> None:
        """Sector violation message includes sector name."""
        weights = pd.Series([0.6, 0.4], index=["A", "B"])
        sectors = {"A": "Technology", "B": "Technology"}
        constraint = SectorConstraint(params={"max_sector_weight": 0.50})
        is_valid, msg = constraint.check_constraints(weights, sectors=sectors)

        assert not is_valid
        assert msg is not None
        assert "Technology" in msg or "exceeded" in msg.lower()

    def test_property19_beta_provides_details(self) -> None:
        """Beta violation message includes actual beta value."""
        weights = pd.Series([1.0], index=["HIGH_BETA"])
        betas = {"HIGH_BETA": 2.5}
        constraint = BetaConstraint(params={"min_beta": 0.8, "max_beta": 1.2})
        is_valid, msg = constraint.check_constraints(weights, betas=betas)

        assert not is_valid
        assert msg is not None
        assert "2.5" in msg or "high" in msg.lower()
