from abc import ABC, abstractmethod

class FillModel(ABC):
    @abstractmethod
    def get_fill_quantity(self, desired_quantity: float, volume: float = None) -> float:
        pass

class ImmediateFill(FillModel):
    def get_fill_quantity(self, desired_quantity: float, volume: float = None) -> float:
        return desired_quantity

class LiquidityConstrainedFill(FillModel):
    def __init__(self, max_participation: float = 0.1):
        self.max_participation = max_participation

    def get_fill_quantity(self, desired_quantity: float, volume: float = None) -> float:
        if volume is None:
            return desired_quantity
            
        max_fill = volume * self.max_participation
        if abs(desired_quantity) > max_fill:
            return max_fill if desired_quantity > 0 else -max_fill
        return desired_quantity
