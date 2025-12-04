from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
import pandas as pd
from datetime import datetime

class Strategy(ABC):
    """
    Abstract base class for trading strategies.
    """
    @abstractmethod
    def on_data(self, data: Dict[str, pd.DataFrame], portfolio: Dict[str, Any]):
        """
        Called on every data step.
        :param data: Dictionary of DataFrames (history) for each ticker.
        :param portfolio: Current portfolio state.
        :return: List of orders (dict) to execute.
        """
        pass

class BacktestResult(ABC):
    """
    Standardized backtest result.
    """
    metrics: Dict[str, float]
    equity_curve: pd.Series
    trades: List[Dict[str, Any]]

class BacktestEngine(ABC):
    """
    Abstract base class for backtest engines.
    """
    @abstractmethod
    def run(self, strategy: Strategy, start_date: datetime, end_date: datetime, initial_capital: float) -> BacktestResult:
        """
        Run the backtest.
        """
        pass
