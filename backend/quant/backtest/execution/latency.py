from abc import ABC, abstractmethod
import random

class LatencyModel(ABC):
    @abstractmethod
    def get_delay_seconds(self) -> float:
        pass

class FixedLatency(LatencyModel):
    def __init__(self, delay_seconds: float = 0.1):
        self.delay = delay_seconds

    def get_delay_seconds(self) -> float:
        return self.delay

class StochasticLatency(LatencyModel):
    def __init__(self, min_delay: float = 0.05, max_delay: float = 0.5):
        self.min_delay = min_delay
        self.max_delay = max_delay

    def get_delay_seconds(self) -> float:
        return random.uniform(self.min_delay, self.max_delay)
