from fastapi import APIRouter, HTTPException, Query
from typing import Dict, Any, List
import pandas as pd
import numpy as np
from datetime import datetime

from quant.mlops.registry import ModelRegistry
from quant.risk.var import calculate_component_var
from quant.risk.hedging import calculate_tail_hedge_cost
from quant.execution.algo import VWAPExecution

router = APIRouter()

@router.get("/ml/signals", response_model=Dict[str, Any])
def get_ml_signals():
    """
    Returns the latest ML signal metrics from MLflow.
    """
    try:
        registry = ModelRegistry(experiment_name="test_genetic_algo") # Using the one we created
        best_run = registry.get_best_run(metric_name="best_fitness", mode="max")
        
        if best_run is None:
            return {
                "status": "no_data",
                "metrics": {},
                "params": {}
            }
            
        return {
            "status": "success",
            "run_id": best_run.run_id,
            "metrics": best_run.filter(regex="metrics.").to_dict(),
            "params": best_run.filter(regex="params.").to_dict()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/risk/metrics", response_model=Dict[str, Any])
def get_risk_metrics(
    portfolio_value: float = 1_000_000,
    volatility: float = 0.15
):
    """
    Returns current risk metrics (VaR, Hedge Cost).
    """
    try:
        # 1. Component VaR (Demo Portfolio)
        weights = np.array([0.6, 0.4]) # 60/40
        cov_matrix = np.array([[0.04, 0.01], [0.01, 0.01]]) # Dummy cov
        
        p_var, m_var, c_var = calculate_component_var(
            weights, cov_matrix, portfolio_value=portfolio_value
        )
        
        # 2. Tail Hedge Cost
        hedge_cost = calculate_tail_hedge_cost(
            portfolio_value=portfolio_value,
            spot_price=400, # Dummy SPY
            volatility=volatility
        )
        
        return {
            "var": {
                "portfolio_var": p_var,
                "component_var": c_var.tolist(),
                "weights": weights.tolist()
            },
            "hedge": hedge_cost
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/execution/vwap", response_model=Dict[str, Any])
def get_vwap_schedule(
    shares: int = 10000,
    volume: int = 1_000_000,
    volatility: float = 0.02
):
    """
    Returns a VWAP execution schedule and impact cost.
    """
    try:
        algo = VWAPExecution()
        schedule = algo.generate_schedule(total_shares=shares)
        
        impact_bps = algo.estimate_impact_cost(shares, volume, volatility)
        
        return {
            "schedule": schedule.to_dict(orient="records"),
            "impact_cost_bps": impact_bps,
            "total_shares": shares
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
