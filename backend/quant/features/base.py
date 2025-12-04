from abc import ABC, abstractmethod
import pandas as pd
from typing import List, Optional

class FeatureGenerator(ABC):
    """
    Abstract base class for feature (factor) generators.
    """
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Unique name of the feature/factor."""
        pass
        
    @property
    @abstractmethod
    def description(self) -> str:
        """Description of what this feature measures."""
        pass
        
    @abstractmethod
    def compute(self, history: pd.DataFrame, fundamentals: Optional[pd.DataFrame] = None) -> pd.Series:
        """
        Compute the feature values.
        :param history: DataFrame with OHLCV data (DatetimeIndex).
        :param fundamentals: Optional DataFrame with fundamental data.
        :return: Series with computed feature values (DatetimeIndex).
        """
        pass
