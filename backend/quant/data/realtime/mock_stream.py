import asyncio
import random
from typing import List, Dict, Any
from quant.data.realtime.interface import StreamClient

class MockStreamClient(StreamClient):
    """
    Simulates real-time market data updates.
    """
    
    def __init__(self, update_interval: float = 1.0):
        super().__init__()
        self.update_interval = update_interval
        self.subscribed_tickers = set()
        self.running = False
        self._task = None
        
    async def connect(self):
        self.running = True
        self._task = asyncio.create_task(self._stream_loop())
        print("MockStreamClient connected.")
        
    async def disconnect(self):
        self.running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        print("MockStreamClient disconnected.")
        
    async def subscribe(self, tickers: List[str]):
        for t in tickers:
            self.subscribed_tickers.add(t)
        print(f"Subscribed to: {tickers}")
        
    async def unsubscribe(self, tickers: List[str]):
        for t in tickers:
            if t in self.subscribed_tickers:
                self.subscribed_tickers.remove(t)
        print(f"Unsubscribed from: {tickers}")
        
    async def _stream_loop(self):
        """Generates random price updates."""
        while self.running:
            if not self.subscribed_tickers:
                await asyncio.sleep(self.update_interval)
                continue
                
            # Simulate updates for a subset of tickers
            tickers_to_update = list(self.subscribed_tickers)
            
            for ticker in tickers_to_update:
                # Generate random price movement
                price_change = random.uniform(-0.005, 0.005) # +/- 0.5%
                
                update = {
                    "type": "trade",
                    "ticker": ticker,
                    "price": 100.0 * (1 + price_change), # Mock price around 100
                    "volume": random.randint(10, 1000),
                    "timestamp": asyncio.get_event_loop().time()
                }
                
                await self._emit(update)
                
            await asyncio.sleep(self.update_interval)
