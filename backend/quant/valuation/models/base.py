from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from pydantic import BaseModel
from datetime import date

class ValuationResult(BaseModel):
    fair_value: float
    upside: float
    details: Dict[str, Any]
    model_name: str

class ValuationContext(BaseModel):
    ticker: str
    valuation_date: date
    financials: Dict[str, Any]  # Abstracted financials input
    market_data: Dict[str, Any] # Abstracted market data input
    params: Optional[Dict[str, Any]] = {}

class ValuationModel(ABC):
    @abstractmethod
    def calculate(self, context: ValuationContext) -> Optional[ValuationResult]:
        """
        Calculate fair value based on the provided context.
        Returns None if calculation cannot be performed (missing data).
        """
        pass
