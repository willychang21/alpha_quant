"""Factor API Endpoints.

Provides REST API for factor discovery, configuration, and performance monitoring.

Endpoints:
- GET /factors/ - List all factors with metadata
- GET /factors/performance - Factor performance metrics
- GET /factors/correlations - Pairwise correlation matrix
- PUT /factors/weights/{factor_name} - Update factor weight
- POST /factors/reset - Reset to defaults
- PUT /factors/{factor_name}/toggle - Enable/disable factor
"""

from typing import Dict, List, Optional
import logging

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from quant.features.factor_registry import FactorRegistry, FactorMetadata

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/factors", tags=["factors"])

# Singleton registry
_factor_registry: Optional[FactorRegistry] = None


def get_factor_registry() -> FactorRegistry:
    """Get or create factor registry singleton."""
    global _factor_registry
    if _factor_registry is None:
        _factor_registry = FactorRegistry()
    return _factor_registry


# Request/Response Models
class FactorResponse(BaseModel):
    """Response model for factor metadata."""
    name: str
    description: str
    category: str
    default_weight: float
    user_weight: Optional[float] = None
    enabled: bool
    effective_weight: float
    ic_5d: Optional[float] = None
    turnover: Optional[float] = None


class WeightUpdateRequest(BaseModel):
    """Request model for weight update."""
    weight: float = Field(ge=0.0, description="New weight (must be >= 0)")


class ToggleRequest(BaseModel):
    """Request model for toggling factor."""
    enabled: bool


class FactorListResponse(BaseModel):
    """Response model for factor list."""
    factors: List[FactorResponse]
    total_count: int


class PerformanceResponse(BaseModel):
    """Response model for factor performance."""
    factor_name: str
    ic_5d: Optional[float] = None
    ic_21d: Optional[float] = None
    sharpe: Optional[float] = None
    turnover: Optional[float] = None


class CorrelationResponse(BaseModel):
    """Response model for factor correlations."""
    matrix: Dict[str, Dict[str, float]]
    factor_names: List[str]


# Endpoints
@router.get("/", response_model=FactorListResponse)
async def list_factors():
    """List all registered factors with metadata."""
    registry = get_factor_registry()
    factors = registry.get_all_factors()
    
    return FactorListResponse(
        factors=[
            FactorResponse(
                name=f.name,
                description=f.description,
                category=f.category.value,
                default_weight=f.default_weight,
                user_weight=f.user_weight,
                enabled=f.enabled,
                effective_weight=f.effective_weight,
                ic_5d=f.ic_5d,
                turnover=f.turnover
            )
            for f in factors
        ],
        total_count=len(factors)
    )


@router.get("/performance", response_model=List[PerformanceResponse])
async def get_performance():
    """Get performance metrics for all factors.
    
    Returns rolling IC (information coefficient), Sharpe ratio, and turnover
    for each factor.
    """
    registry = get_factor_registry()
    factors = registry.get_all_factors()
    
    # Note: In production, these would be computed from actual data
    return [
        PerformanceResponse(
            factor_name=f.name,
            ic_5d=f.ic_5d,
            ic_21d=None,  # Would be computed
            sharpe=None,  # Would be computed
            turnover=f.turnover
        )
        for f in factors
    ]


@router.get("/correlations", response_model=CorrelationResponse)
async def get_correlations():
    """Get pairwise correlation matrix between factors.
    
    Returns correlation coefficients as a nested dictionary.
    """
    registry = get_factor_registry()
    factor_names = [f.name for f in registry.get_all_factors()]
    
    # Note: In production, compute actual correlations
    # For now, return identity matrix as placeholder
    matrix = {
        name: {other: (1.0 if name == other else 0.0) for other in factor_names}
        for name in factor_names
    }
    
    return CorrelationResponse(
        matrix=matrix,
        factor_names=factor_names
    )


@router.put("/weights/{factor_name}")
async def update_weight(factor_name: str, request: WeightUpdateRequest):
    """Update user weight for a specific factor.
    
    Args:
        factor_name: Name of the factor.
        request: Weight update request.
        
    Returns:
        Updated factor metadata.
    """
    registry = get_factor_registry()
    
    success = registry.set_user_weight(factor_name, request.weight)
    if not success:
        raise HTTPException(
            status_code=404,
            detail=f"Factor not found or invalid weight: {factor_name}"
        )
    
    factor = registry.get_factor(factor_name)
    return FactorResponse(
        name=factor.name,
        description=factor.description,
        category=factor.category.value,
        default_weight=factor.default_weight,
        user_weight=factor.user_weight,
        enabled=factor.enabled,
        effective_weight=factor.effective_weight,
        ic_5d=factor.ic_5d,
        turnover=factor.turnover
    )


@router.post("/reset")
async def reset_weights():
    """Reset all user weights to defaults."""
    registry = get_factor_registry()
    registry.reset_user_weights()
    
    return {"message": "All weights reset to defaults"}


@router.put("/{factor_name}/toggle")
async def toggle_factor(factor_name: str, request: ToggleRequest):
    """Enable or disable a factor.
    
    Args:
        factor_name: Name of the factor.
        request: Toggle request.
        
    Returns:
        Updated factor metadata.
    """
    registry = get_factor_registry()
    
    success = registry.toggle_factor(factor_name, request.enabled)
    if not success:
        raise HTTPException(
            status_code=404,
            detail=f"Factor not found: {factor_name}"
        )
    
    factor = registry.get_factor(factor_name)
    return FactorResponse(
        name=factor.name,
        description=factor.description,
        category=factor.category.value,
        default_weight=factor.default_weight,
        user_weight=factor.user_weight,
        enabled=factor.enabled,
        effective_weight=factor.effective_weight,
        ic_5d=factor.ic_5d,
        turnover=factor.turnover
    )
