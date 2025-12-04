import pandas as pd
import numpy as np
from typing import Dict, List, Any
import logging

logger = logging.getLogger(__name__)

class PortfolioSimulator:
    """
    Fast vectorized simulator for standard portfolio strategies (DCA, Lump Sum).
    Replaces the legacy 'get_backtest_data' function.
    """
    
    @staticmethod
    def simulate_dca(
        history: pd.DataFrame, 
        allocation: Dict[str, float], 
        monthly_amount: float,
        initial_amount: float = 0,
        transaction_cost_rate: float = 0.0 # e.g. 0.001 for 10bps
    ) -> Dict[str, Any]:
        """
        Simulate Dollar Cost Averaging.
        """
        if history.empty:
            return {}
            
        # Normalize weights
        total_alloc = sum(allocation.values())
        if total_alloc == 0:
            return {}
            
        weights = pd.Series(allocation) / total_alloc
        
        # Filter history to tickers in allocation
        available_tickers = [t for t in weights.index if t in history.columns]
        if not available_tickers:
            return {}
            
        prices = history[available_tickers].dropna()
        if prices.empty:
            return {}
            
        # Calculate daily/monthly returns
        # Assuming history is monthly for this specific legacy compatibility, 
        # but robust enough to handle daily if resampled.
        returns = prices.pct_change().fillna(0)
        
        # Portfolio Returns (Weighted)
        # aligned weights
        w = weights[available_tickers].values
        portfolio_returns = returns.dot(w)
        
        # Apply Transaction Costs to Returns
        # For DCA, we pay cost on the *new* contribution every period.
        # For rebalancing (not explicitly modeled here yet), we'd pay on turnover.
        # Here we approximate cost by reducing the contribution amount effectively.
        
        # Simulation
        dates = portfolio_returns.index
        
        # DCA Logic
        dca_value = initial_amount * (1 - transaction_cost_rate) if initial_amount > 0 else 0
        total_invested = initial_amount
        dca_curve = []
        
        # Lump Sum Logic
        curr_lump = max(initial_amount, monthly_amount) * (1 - transaction_cost_rate)
        lump_curve = []
        
        for date, ret in portfolio_returns.items():
            # DCA: Add contribution (minus cost), then grow
            net_contribution = monthly_amount * (1 - transaction_cost_rate)
            dca_value = (dca_value + net_contribution) * (1 + ret)
            total_invested += monthly_amount
            
            dca_curve.append({
                "date": date.strftime('%Y-%m-%d'),
                "value": dca_value,
                "invested": total_invested
            })
            
            # Lump Sum: Just grow
            curr_lump = curr_lump * (1 + ret)
            lump_curve.append({
                "date": date.strftime('%Y-%m-%d'),
                "value": curr_lump
            })
            
        return {
            "dca": dca_curve,
            "lump_sum": lump_curve,
            "metrics": {
                "total_return_dca": (dca_value / total_invested - 1) if total_invested > 0 else 0,
                "final_value_dca": dca_value,
                "transaction_costs_paid": total_invested * transaction_cost_rate # Approx
            }
        }

    @staticmethod
    def run_cost_sensitivity_analysis(
        history: pd.DataFrame,
        allocation: Dict[str, float],
        monthly_amount: float,
        initial_amount: float = 0,
        cost_scenarios: List[float] = [0.0, 0.001, 0.002] # 0, 10bps, 20bps
    ) -> pd.DataFrame:
        """
        Run simulation under multiple cost scenarios and return comparative metrics.
        """
        results = []
        for cost in cost_scenarios:
            res = PortfolioSimulator.simulate_dca(
                history, allocation, monthly_amount, initial_amount, transaction_cost_rate=cost
            )
            metrics = res.get('metrics', {})
            results.append({
                "cost_bps": cost * 10000,
                "final_value": metrics.get('final_value_dca', 0),
                "total_return": metrics.get('total_return_dca', 0)
            })
            
        return pd.DataFrame(results)
