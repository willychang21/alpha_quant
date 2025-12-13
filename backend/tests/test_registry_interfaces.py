"""Property-based tests for Core Interfaces and Registry System.

Tests ABC enforcement (Properties 1) and interface validation (Property 6).
Uses Hypothesis for property-based testing.

Properties tested:
- Property 1: ABC Interface Enforcement (FactorBase, OptimizerBase, RiskModelBase)
- Property 6: Factor Interface Validation
"""

import sys
from pathlib import Path
from typing import Optional

import pandas as pd
import pytest
from hypothesis import given, settings
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
# Property 1: ABC Interface Enforcement
# =============================================================================


class TestFactorBaseABC:
    """Test that FactorBase enforces abstract method implementation."""

    def test_cannot_instantiate_abstract_factor(self) -> None:
        """
        **Feature: registry-pattern-refactoring, Property 1: ABC Interface Enforcement**

        For any class inheriting from FactorBase without implementing required
        abstract methods, instantiation SHALL raise TypeError.
        """
        # Attempt to create a class without implementing abstract methods
        class IncompleteFactor(FactorBase):
            pass

        with pytest.raises(TypeError) as exc_info:
            IncompleteFactor()

        assert "abstract" in str(exc_info.value).lower()

    def test_cannot_instantiate_without_metadata(self) -> None:
        """
        **Feature: registry-pattern-refactoring, Property 1: ABC Interface Enforcement**

        Missing metadata property raises TypeError.
        """
        class MissingMetadata(FactorBase):
            def compute(self, data: pd.DataFrame) -> pd.Series:
                return pd.Series(dtype=float)

        with pytest.raises(TypeError):
            MissingMetadata()

    def test_cannot_instantiate_without_compute(self) -> None:
        """
        **Feature: registry-pattern-refactoring, Property 1: ABC Interface Enforcement**

        Missing compute method raises TypeError.
        """
        class MissingCompute(FactorBase):
            @property
            def metadata(self) -> PluginMetadata:
                return PluginMetadata(name="Test", description="Test")

        with pytest.raises(TypeError):
            MissingCompute()

    def test_complete_implementation_succeeds(self) -> None:
        """Valid implementation can be instantiated."""
        class CompleteFactor(FactorBase):
            @property
            def metadata(self) -> PluginMetadata:
                return PluginMetadata(name="Complete", description="Test")

            def compute(self, data: pd.DataFrame) -> pd.Series:
                return pd.Series(dtype=float)

        # Should not raise
        factor = CompleteFactor()
        assert factor.metadata.name == "Complete"


class TestOptimizerBaseABC:
    """Test that OptimizerBase enforces abstract method implementation."""

    def test_cannot_instantiate_abstract_optimizer(self) -> None:
        """
        **Feature: registry-pattern-refactoring, Property 1: ABC Interface Enforcement**

        For any class inheriting from OptimizerBase without implementing required
        abstract methods, instantiation SHALL raise TypeError.
        """
        class IncompleteOptimizer(OptimizerBase):
            pass

        with pytest.raises(TypeError) as exc_info:
            IncompleteOptimizer()

        assert "abstract" in str(exc_info.value).lower()

    def test_cannot_instantiate_without_metadata(self) -> None:
        """Missing metadata raises TypeError."""
        class MissingMetadata(OptimizerBase):
            def optimize(
                self, returns: pd.DataFrame, cov: pd.DataFrame, **kwargs
            ) -> pd.Series:
                return pd.Series(dtype=float)

        with pytest.raises(TypeError):
            MissingMetadata()

    def test_cannot_instantiate_without_optimize(self) -> None:
        """Missing optimize method raises TypeError."""
        class MissingOptimize(OptimizerBase):
            @property
            def metadata(self) -> PluginMetadata:
                return PluginMetadata(name="Test", description="Test")

        with pytest.raises(TypeError):
            MissingOptimize()

    def test_complete_implementation_succeeds(self) -> None:
        """Valid implementation can be instantiated."""
        class CompleteOptimizer(OptimizerBase):
            @property
            def metadata(self) -> PluginMetadata:
                return PluginMetadata(name="Complete", description="Test")

            def optimize(
                self, returns: pd.DataFrame, cov: pd.DataFrame, **kwargs
            ) -> pd.Series:
                return pd.Series(dtype=float)

        optimizer = CompleteOptimizer()
        assert optimizer.metadata.name == "Complete"


class TestRiskModelBaseABC:
    """Test that RiskModelBase enforces abstract method implementation."""

    def test_cannot_instantiate_abstract_risk_model(self) -> None:
        """
        **Feature: registry-pattern-refactoring, Property 1: ABC Interface Enforcement**

        For any class inheriting from RiskModelBase without implementing required
        abstract methods, instantiation SHALL raise TypeError.
        """
        class IncompleteRiskModel(RiskModelBase):
            pass

        with pytest.raises(TypeError) as exc_info:
            IncompleteRiskModel()

        assert "abstract" in str(exc_info.value).lower()

    def test_cannot_instantiate_without_metadata(self) -> None:
        """Missing metadata raises TypeError."""
        class MissingMetadata(RiskModelBase):
            def check_constraints(
                self, weights: pd.Series, **context
            ) -> tuple[bool, Optional[str]]:
                return True, None

        with pytest.raises(TypeError):
            MissingMetadata()

    def test_cannot_instantiate_without_check_constraints(self) -> None:
        """Missing check_constraints method raises TypeError."""
        class MissingCheck(RiskModelBase):
            @property
            def metadata(self) -> PluginMetadata:
                return PluginMetadata(name="Test", description="Test")

        with pytest.raises(TypeError):
            MissingCheck()

    def test_complete_implementation_succeeds(self) -> None:
        """Valid implementation can be instantiated."""
        class CompleteRiskModel(RiskModelBase):
            @property
            def metadata(self) -> PluginMetadata:
                return PluginMetadata(name="Complete", description="Test")

            def check_constraints(
                self, weights: pd.Series, **context
            ) -> tuple[bool, Optional[str]]:
                return True, None

        risk_model = CompleteRiskModel()
        assert risk_model.metadata.name == "Complete"


# =============================================================================
# Property 6: Factor Interface Validation
# =============================================================================


class TestFactorInterfaceValidation:
    """Test that registered factors implement FactorBase."""

    def test_registered_factor_is_instance_of_factorbase(self) -> None:
        """
        **Feature: registry-pattern-refactoring, Property 6: Factor Interface Validation**

        For any class registered via @register_factor, isinstance(cls(), FactorBase)
        SHALL return True.
        """
        # Clear any previous test registrations
        registry.clear()

        @register_factor("TestFactor")
        class ValidFactor(FactorBase):
            def __init__(self, params: dict = None):
                self.params = params or {}

            @property
            def metadata(self) -> PluginMetadata:
                return PluginMetadata(name="TestFactor", description="Test")

            def compute(self, data: pd.DataFrame) -> pd.Series:
                return pd.Series(dtype=float)

        # Retrieve and instantiate
        factor_cls = registry.get_factor("TestFactor")
        factor = factor_cls()

        assert isinstance(factor, FactorBase)

    def test_cannot_register_non_factorbase(self) -> None:
        """
        **Feature: registry-pattern-refactoring, Property 6: Factor Interface Validation**

        Attempting to register a class not inheriting from FactorBase SHALL raise TypeError.
        """
        registry.clear()

        with pytest.raises(TypeError) as exc_info:
            @register_factor("Invalid")
            class NotAFactor:
                pass

        assert "FactorBase" in str(exc_info.value)


# =============================================================================
# PluginMetadata Tests
# =============================================================================


class TestPluginMetadata:
    """Tests for PluginMetadata dataclass."""

    @given(
        name=st.text(min_size=1, max_size=50),
        description=st.text(max_size=200),
        version=st.from_regex(r"[0-9]+\.[0-9]+\.[0-9]+", fullmatch=True),
    )
    @settings(max_examples=30)
    def test_metadata_serialization_round_trip(
        self, name: str, description: str, version: str
    ) -> None:
        """Metadata can be serialized to dict and back."""
        metadata = PluginMetadata(
            name=name,
            description=description,
            version=version,
            author="Test",
            parameters={"param1": "value1"},
        )

        serialized = metadata.to_dict()

        assert serialized["name"] == name
        assert serialized["description"] == description
        assert serialized["version"] == version
        assert serialized["parameters"] == {"param1": "value1"}

    def test_metadata_default_values(self) -> None:
        """Metadata has correct defaults."""
        metadata = PluginMetadata(name="Test", description="Desc")

        assert metadata.version == "1.0.0"
        assert metadata.author == ""
        assert metadata.parameters == {}
        assert metadata.category == ""

    def test_metadata_post_init_handles_none_parameters(self) -> None:
        """__post_init__ converts None parameters to empty dict."""
        metadata = PluginMetadata(
            name="Test", description="Desc", parameters=None  # type: ignore
        )
        assert metadata.parameters == {}
