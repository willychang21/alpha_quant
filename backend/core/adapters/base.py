from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
import pandas as pd

class MarketDataProvider(ABC):
    """
    Abstract base class for market data providers.
    """
    
    @abstractmethod
    async def get_ticker_data(self, ticker: str) -> Dict[str, Any]:
        """
        Fetch all necessary data for a ticker (info, financials, history).
        """
        pass
        
    @abstractmethod
    async def get_exchange_rate(self, from_currency: str, to_currency: str = "USD") -> float:
        """
        Fetch exchange rate.
        """
        pass
        
    @abstractmethod
    async def get_history(self, ticker: str, period: str = "2y") -> pd.DataFrame:
        """
        Fetch historical price data.
        """
        pass
