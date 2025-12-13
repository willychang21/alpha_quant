"""Property-based tests for Registry Core functionality.

Tests registration/retrieval (Properties 2-4) and metadata (Property 21).
Uses Hypothesis for property-based testing.

Properties tested:
- Property 2: Decorator Registration Round-Trip (factors, optimizers, risk models)
- Property 3: Registry Lookup Consistency
- Property 4: Registry Listing Completeness
- Property 21: Plugin Metadata Retrieval
"""

import sys
from pathlib import Path
from typing import Optional

import pandas as pd
import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from quant.core.interfaces import (
    FactorBase,
    OptimizerBase,
    PluginMetadata,
    RiskModelBase,
)
from quant.core.registry import (
    PluginRegistry,
    register_factor,
    register_optimizer,
    register_risk_model,
    registry,
)


# =============================================================================
# Hypothesis Configuration
# =============================================================================

settings.register_profile("ci", max_examples=100, deadline=None)
settings.register_profile("dev", max_examples=20, deadline=None)
settings.load_profile("dev")


# =============================================================================
# Test Fixtures
# =============================================================================


def create_test_factor(name: str, description: str = "Test") -> type:
    """Create a valid FactorBase subclass dynamically."""
    class TestFactor(FactorBase):
        def __init__(self, params: dict = None):
            self.params = params or {}

        @property
        def metadata(self) -> PluginMetadata:
            return PluginMetadata(name=name, description=description)

        def compute(self, data: pd.DataFrame) -> pd.Series:
            return pd.Series(dtype=float)

    TestFactor.__name__ = f"Factor_{name}"
    return TestFactor


def create_test_optimizer(name: str, description: str = "Test") -> type:
    """Create a valid OptimizerBase subclass dynamically."""
    class TestOptimizer(OptimizerBase):
        def __init__(self, params: dict = None):
            self.params = params or {}

        @property
        def metadata(self) -> PluginMetadata:
            return PluginMetadata(name=name, description=description)

        def optimize(
            self, returns: pd.DataFrame, cov: pd.DataFrame, **kwargs
        ) -> pd.Series:
            return pd.Series(dtype=float)

    TestOptimizer.__name__ = f"Optimizer_{name}"
    return TestOptimizer


def create_test_risk_model(name: str, description: str = "Test") -> type:
    """Create a valid RiskModelBase subclass dynamically."""
    class TestRiskModel(RiskModelBase):
        def __init__(self, params: dict = None):
            self.params = params or {}

        @property
        def metadata(self) -> PluginMetadata:
            return PluginMetadata(name=name, description=description)

        def check_constraints(
            self, weights: pd.Series, **context
        ) -> tuple[bool, Optional[str]]:
            return True, None

    TestRiskModel.__name__ = f"RiskModel_{name}"
    return TestRiskModel


@pytest.fixture(autouse=True)
def clean_registry():
    """Reset registry before each test."""
    registry.clear()
    yield
    registry.clear()


# =============================================================================
# Property 2: Decorator Registration Round-Trip
# =============================================================================


class TestFactorRegistrationRoundTrip:
    """Test factor registration round-trip."""

    @given(name=st.text(min_size=1, max_size=30, alphabet=st.characters(
        whitelist_categories=("Lu", "Ll", "Nd"),
        whitelist_characters="_"
    )))
    @settings(max_examples=30)
    def test_property2_factor_registration_round_trip(self, name: str) -> None:
        """
        **Feature: registry-pattern-refactoring, Property 2: Decorator Registration Round-Trip**

        For any valid plugin class and registration name, decorating with
        @register_factor(name) SHALL make the class retrievable via
        registry.get_factor(name).
        """
        # Skip empty names after filtering
        assume(len(name.strip()) > 0)
        name = name.strip()

        factor_cls = create_test_factor(name)
        decorated = registry.register_factor(name)(factor_cls)

        retrieved = registry.get_factor(name)
        assert retrieved is decorated
        assert retrieved is factor_cls


class TestOptimizerRegistrationRoundTrip:
    """Test optimizer registration round-trip."""

    @given(name=st.text(min_size=1, max_size=30, alphabet=st.characters(
        whitelist_categories=("Lu", "Ll", "Nd"),
        whitelist_characters="_"
    )))
    @settings(max_examples=30)
    def test_property2_optimizer_registration_round_trip(self, name: str) -> None:
        """
        **Feature: registry-pattern-refactoring, Property 2: Decorator Registration Round-Trip**

        For any valid plugin class and registration name, decorating with
        @register_optimizer(name) SHALL make the class retrievable via
        registry.get_optimizer(name).
        """
        assume(len(name.strip()) > 0)
        name = name.strip()

        optimizer_cls = create_test_optimizer(name)
        decorated = registry.register_optimizer(name)(optimizer_cls)

        retrieved = registry.get_optimizer(name)
        assert retrieved is decorated


class TestRiskModelRegistrationRoundTrip:
    """Test risk model registration round-trip."""

    @given(name=st.text(min_size=1, max_size=30, alphabet=st.characters(
        whitelist_categories=("Lu", "Ll", "Nd"),
        whitelist_characters="_"
    )))
    @settings(max_examples=30)
    def test_property2_risk_model_registration_round_trip(self, name: str) -> None:
        """
        **Feature: registry-pattern-refactoring, Property 2: Decorator Registration Round-Trip**

        For any valid plugin class and registration name, decorating with
        @register_risk_model(name) SHALL make the class retrievable via
        registry.get_risk_model(name).
        """
        assume(len(name.strip()) > 0)
        name = name.strip()

        risk_model_cls = create_test_risk_model(name)
        decorated = registry.register_risk_model(name)(risk_model_cls)

        retrieved = registry.get_risk_model(name)
        assert retrieved is decorated


# =============================================================================
# Property 3: Registry Lookup Consistency
# =============================================================================


class TestRegistryLookupConsistency:
    """Test registry lookup consistency."""

    def test_property3_get_factor_returns_exact_class(self) -> None:
        """
        **Feature: registry-pattern-refactoring, Property 3: Registry Lookup Consistency**

        For any registered plugin name, registry.get_factor(name) SHALL return
        the exact class that was registered.
        """
        factor_cls = create_test_factor("Momentum")
        registry.register_factor("Momentum")(factor_cls)

        retrieved = registry.get_factor("Momentum")
        assert retrieved is factor_cls

    def test_property3_unregistered_raises_keyerror_with_available_names(self) -> None:
        """
        **Feature: registry-pattern-refactoring, Property 3: Registry Lookup Consistency**

        For any unregistered name, it SHALL raise KeyError with available names listed.
        """
        # Register some factors
        registry.register_factor("Alpha")(create_test_factor("Alpha"))
        registry.register_factor("Beta")(create_test_factor("Beta"))

        with pytest.raises(KeyError) as exc_info:
            registry.get_factor("NonExistent")

        error_message = str(exc_info.value)
        assert "NonExistent" in error_message
        assert "Alpha" in error_message or "available" in error_message.lower()

    def test_property3_optimizer_lookup_consistency(self) -> None:
        """Optimizer lookup returns exact registered class."""
        optimizer_cls = create_test_optimizer("HRP")
        registry.register_optimizer("HRP")(optimizer_cls)

        assert registry.get_optimizer("HRP") is optimizer_cls

    def test_property3_risk_model_lookup_consistency(self) -> None:
        """Risk model lookup returns exact registered class."""
        risk_cls = create_test_risk_model("MaxWeight")
        registry.register_risk_model("MaxWeight")(risk_cls)

        assert registry.get_risk_model("MaxWeight") is risk_cls


# =============================================================================
# Property 4: Registry Listing Completeness
# =============================================================================


class TestRegistryListingCompleteness:
    """Test registry listing completeness."""

    @given(names=st.lists(
        st.text(min_size=1, max_size=20, alphabet=st.characters(
            whitelist_categories=("Lu", "Ll", "Nd"),
            whitelist_characters="_"
        )),
        min_size=1,
        max_size=10,
        unique=True
    ))
    @settings(max_examples=30)
    def test_property4_list_factors_returns_all_registered(self, names: list) -> None:
        """
        **Feature: registry-pattern-refactoring, Property 4: Registry Listing Completeness**

        For any set of N registered factors, registry.list_factors() SHALL return
        exactly N names matching all registered factors.
        """
        # Filter and clean names
        clean_names = [n.strip() for n in names if n.strip()]
        assume(len(clean_names) > 0)
        unique_names = list(set(clean_names))

        # Clear registry at start of each Hypothesis example
        registry.clear()

        for name in unique_names:
            registry.register_factor(name)(create_test_factor(name))

        listed = registry.list_factors()

        assert len(listed) == len(unique_names)
        assert set(listed) == set(unique_names)

    def test_property4_list_optimizers_completeness(self) -> None:
        """list_optimizers returns all registered optimizer names."""
        names = ["HRP", "MVO", "Kelly"]
        for name in names:
            registry.register_optimizer(name)(create_test_optimizer(name))

        listed = registry.list_optimizers()

        assert len(listed) == 3
        assert set(listed) == set(names)

    def test_property4_list_risk_models_completeness(self) -> None:
        """list_risk_models returns all registered risk model names."""
        names = ["MaxWeight", "Sector", "Beta"]
        for name in names:
            registry.register_risk_model(name)(create_test_risk_model(name))

        listed = registry.list_risk_models()

        assert len(listed) == 3
        assert set(listed) == set(names)


# =============================================================================
# Property 21: Plugin Metadata Retrieval
# =============================================================================


class TestPluginMetadataRetrieval:
    """Test plugin metadata retrieval."""

    def test_property21_get_all_metadata_includes_factors(self) -> None:
        """
        **Feature: registry-pattern-refactoring, Property 21: Plugin Metadata Retrieval**

        For any registered plugin, registry.get_all_metadata() SHALL return
        metadata including name, description, version, and parameters.
        """
        # Register a factor with specific metadata
        @register_factor("TestFactor")
        class MetadataFactor(FactorBase):
            def __init__(self, params: dict = None):
                self.params = params or {}

            @property
            def metadata(self) -> PluginMetadata:
                return PluginMetadata(
                    name="TestFactor",
                    description="A test factor",
                    version="2.0.0",
                    author="Test Author",
                    parameters={"lookback": "Lookback period"},
                )

            def compute(self, data: pd.DataFrame) -> pd.Series:
                return pd.Series(dtype=float)

        all_metadata = registry.get_all_metadata()

        assert "factors" in all_metadata
        assert "TestFactor" in all_metadata["factors"]

        factor_meta = all_metadata["factors"]["TestFactor"]
        assert factor_meta["name"] == "TestFactor"
        assert factor_meta["description"] == "A test factor"
        assert factor_meta["version"] == "2.0.0"
        assert "lookback" in factor_meta["parameters"]

    def test_property21_get_all_metadata_includes_all_types(self) -> None:
        """Metadata includes factors, optimizers, and risk models."""
        registry.register_factor("F1")(create_test_factor("F1"))
        registry.register_optimizer("O1")(create_test_optimizer("O1"))
        registry.register_risk_model("R1")(create_test_risk_model("R1"))

        metadata = registry.get_all_metadata()

        assert "F1" in metadata["factors"]
        assert "O1" in metadata["optimizers"]
        assert "R1" in metadata["risk_models"]


# =============================================================================
# Singleton Pattern Tests
# =============================================================================


class TestSingletonPattern:
    """Test that registry is a singleton."""

    def test_singleton_returns_same_instance(self) -> None:
        """Multiple PluginRegistry() calls return same instance."""
        r1 = PluginRegistry()
        r2 = PluginRegistry()

        assert r1 is r2

    def test_module_level_registry_is_singleton(self) -> None:
        """Module-level registry is the singleton instance."""
        new_instance = PluginRegistry()
        assert registry is new_instance
