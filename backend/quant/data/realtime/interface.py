from abc import ABC, abstractmethod
from typing import List, Callable, Dict, Any
import asyncio

class StreamClient(ABC):
    """
    Abstract base class for real-time data stream clients.
    """
    
    def __init__(self):
        self.callbacks: List[Callable[[Dict[str, Any]], None]] = []
        
    def add_callback(self, callback: Callable[[Dict[str, Any]], None]):
        """Register a callback function to handle incoming data."""
        self.callbacks.append(callback)
        
    @abstractmethod
    async def connect(self):
        """Establish connection to the data source."""
        pass
        
    @abstractmethod
    async def disconnect(self):
        """Close connection."""
        pass
        
    @abstractmethod
    async def subscribe(self, tickers: List[str]):
        """Subscribe to real-time updates for tickers."""
        pass
        
    @abstractmethod
    async def unsubscribe(self, tickers: List[str]):
        """Unsubscribe from tickers."""
        pass
        
    async def _emit(self, data: Dict[str, Any]):
        """Internal method to notify callbacks."""
        for callback in self.callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(data)
                else:
                    callback(data)
            except Exception as e:
                # Log error but don't crash the stream
                print(f"Error in stream callback: {e}")
