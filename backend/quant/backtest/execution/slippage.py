from abc import ABC, abstractmethod
import numpy as np

class SlippageModel(ABC):
    @abstractmethod
    def calculate_price(self, signal_price: float, quantity: float, volume: float = None, volatility: float = None) -> float:
        """
        Calculate the executed price including slippage.
        """
        pass

class FixedSlippage(SlippageModel):
    def __init__(self, spread_bps: float = 10.0):
        self.spread = spread_bps / 10000.0

    def calculate_price(self, signal_price: float, quantity: float, volume: float = None, volatility: float = None) -> float:
        # Buy: Price + Spread, Sell: Price - Spread (Wait, usually you pay spread on both)
        # Actually: Buy executes at Ask (Price + half_spread), Sell at Bid (Price - half_spread)
        # Simplified: You pay a cost relative to mid-price.
        
        slippage = signal_price * self.spread
        if quantity > 0:
            return signal_price + slippage
        else:
            return signal_price - slippage

class VolumeShareSlippage(SlippageModel):
    """
    Models market impact using a square-root law or similar.
    Impact ~ Volatility * sqrt(OrderSize / DailyVolume)
    """
    def __init__(self, price_impact_coeff: float = 0.1):
        self.coeff = price_impact_coeff

    def calculate_price(self, signal_price: float, quantity: float, volume: float = None, volatility: float = None) -> float:
        if volume is None or volume == 0:
            return signal_price # Fallback
            
        if volatility is None:
            volatility = 0.01 # Default 1% daily vol
            
        participation_rate = abs(quantity) / volume
        impact = self.coeff * volatility * np.sqrt(participation_rate)
        
        if quantity > 0:
            return signal_price * (1 + impact)
        else:
            return signal_price * (1 - impact)
