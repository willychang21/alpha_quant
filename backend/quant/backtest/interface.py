"""Backtest Interface Definitions.

Provides abstract base classes for:
- Strategy: Trading strategy interface
- BacktestResult: Standardized result container
- BacktestEngine: Backtest execution interface
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
import pandas as pd
from datetime import datetime


class Strategy(ABC):
    """Abstract base class for trading strategies."""
    
    @abstractmethod
    def on_data(self, data: Dict[str, pd.DataFrame], portfolio: Dict[str, Any]):
        """Called on every data step.
        
        Args:
            data: Dictionary of DataFrames (history) for each ticker.
            portfolio: Current portfolio state.
        
        Returns:
            List of orders (dict) to execute.
        """
        pass


class BacktestResult:
    """Standardized backtest result container.
    
    Attributes:
        equity_curve: Series of portfolio values over time
        metrics: Dictionary of performance metrics (cagr, sharpe, max_drawdown, etc.)
        trades: List of executed trades
    """
    
    def __init__(
        self,
        equity_curve: Optional[pd.Series] = None,
        metrics: Optional[Dict[str, float]] = None,
        trades: Optional[List[Dict[str, Any]]] = None
    ):
        self.equity_curve = equity_curve if equity_curve is not None else pd.Series(dtype=float)
        self.metrics = metrics if metrics is not None else {}
        self.trades = trades if trades is not None else []


class BacktestEngine(ABC):
    """Abstract base class for backtest engines."""
    
    @abstractmethod
    def run(
        self, 
        strategy: Strategy, 
        start_date: datetime, 
        end_date: datetime, 
        initial_capital: float
    ) -> BacktestResult:
        """Run the backtest.
        
        Args:
            strategy: Trading strategy instance
            start_date: Backtest start date
            end_date: Backtest end date
            initial_capital: Starting capital
        
        Returns:
            BacktestResult with equity curve, metrics, and trades
        """
        pass
