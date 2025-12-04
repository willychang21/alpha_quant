import asyncio
import logging
import concurrent.futures
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
import yfinance as yf
import pandas as pd
from fastapi import WebSocket

from quant.data.realtime.interface import StreamClient
from quant.data.realtime.mock_stream import MockStreamClient
from quant.data.realtime.connection_manager import ConnectionManager

logger = logging.getLogger(__name__)

class MarketDataService:
    """
    Service for retrieving market data asynchronously with caching.
    Designed to meet high-performance standards by avoiding blocking I/O
    and reducing redundant network requests.
    """
    
    _instance = None
    _cache: Dict[str, Any] = {}
    _cache_expiry: Dict[str, datetime] = {}
    _executor = concurrent.futures.ThreadPoolExecutor(max_workers=10)
    
    # Streaming components
    stream_client: Optional[StreamClient] = None
    connection_manager: Optional[ConnectionManager] = None
    
    # Cache TTL settings
    TTL_INFO = timedelta(minutes=60)
    TTL_PRICE = timedelta(minutes=1)
    TTL_FINANCIALS = timedelta(days=1)

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(MarketDataService, cls).__new__(cls)
            cls._instance._initialize_streaming()
        return cls._instance

    def _initialize_streaming(self):
        """Initialize streaming components."""
        self.connection_manager = ConnectionManager()
        # In production, this would be a real client (e.g., Polygon, Alpaca)
        self.stream_client = MockStreamClient()
        self.stream_client.add_callback(self._handle_stream_update)

    async def start_stream(self):
        """Start the stream client."""
        if self.stream_client:
            await self.stream_client.connect()

    async def stop_stream(self):
        """Stop the stream client."""
        if self.stream_client:
            await self.stream_client.disconnect()

    async def connect_websocket(self, websocket: WebSocket, tickers: List[str] = None):
        """Handle new WebSocket connection."""
        if self.connection_manager:
            await self.connection_manager.connect(websocket)
            if tickers and self.stream_client:
                await self.connection_manager.subscribe(websocket, tickers)
                await self.stream_client.subscribe(tickers)

    def disconnect_websocket(self, websocket: WebSocket):
        """Handle WebSocket disconnection."""
        if self.connection_manager:
            self.connection_manager.disconnect(websocket)

    async def _handle_stream_update(self, data: Dict[str, Any]):
        """Callback for stream updates."""
        if self.connection_manager:
            await self.connection_manager.broadcast(data)

    async def get_ticker_data(self, ticker: str) -> Dict[str, Any]:
        """
        Orchestrates fetching all necessary data for a ticker in parallel.
        Returns a dictionary containing info, financials, and history.
        """
        logger.info(f"Fetching data for {ticker}...")
        
        # Define tasks
        loop = asyncio.get_running_loop()
        
        # We fetch these in parallel
        # 1. Info (includes current price, profile)
        # 2. Financials (Income, Balance, Cashflow)
        # 3. History (Price action for quant/risk)
        
        t = yf.Ticker(ticker)
        
        try:
            results = await asyncio.gather(
                self._fetch_info(loop, t, ticker),
                self._fetch_financials(loop, t, ticker),
                self._fetch_history(loop, t, ticker)
            )
            
            info, financials, history = results
            
            return {
                "info": info,
                "income": financials.get("income"),
                "balance": financials.get("balance"),
                "cashflow": financials.get("cashflow"),
                "history": history
            }
            
        except Exception as e:
            logger.error(f"Error fetching data for {ticker}: {e}")
            raise e

    async def get_exchange_rate(self, from_currency: str, to_currency: str = "USD") -> float:
        """
        Fetches exchange rate asynchronously with caching.
        """
        if from_currency == to_currency:
            return 1.0
            
        pair = f"{from_currency}{to_currency}=X"
        cache_key = f"fx_{pair}"
        
        cached = self._get_from_cache(cache_key)
        if cached:
            return cached
            
        try:
            loop = asyncio.get_running_loop()
            rate = await loop.run_in_executor(self._executor, self._fetch_fx_rate_sync, pair)
            self._set_cache(cache_key, rate, self.TTL_INFO) # FX rates don't change that fast for valuation purposes
            return rate
        except Exception as e:
            logger.error(f"Error fetching exchange rate {pair}: {e}")
            return 1.0 # Fallback

    def _fetch_fx_rate_sync(self, pair: str) -> float:
        ticker = yf.Ticker(pair)
        rate = ticker.info.get('regularMarketPrice') or ticker.info.get('previousClose')
        if not rate:
            hist = ticker.history(period="1d")
            if not hist.empty:
                rate = hist['Close'].iloc[-1]
        return rate if rate else 1.0

    async def _fetch_info(self, loop, ticker_obj, ticker_symbol) -> Dict:
        cache_key = f"info_{ticker_symbol}"
        cached = self._get_from_cache(cache_key)
        if cached:
            return cached
            
        def fetch():
            data = ticker_obj.info
            # Ensure it's a dict (yfinance sometimes behaves oddly)
            if isinstance(data, pd.DataFrame):
                # If it's a DataFrame, try to convert to dict (maybe it's a single row?)
                if not data.empty:
                    return data.iloc[0].to_dict()
                return {}
            return data
            
        info = await loop.run_in_executor(self._executor, fetch)
        if not isinstance(info, dict):
             logger.warning(f"Info for {ticker_symbol} is not a dict: {type(info)}")
             info = {}
             
        self._set_cache(cache_key, info, self.TTL_INFO)
        return info

    async def _fetch_financials(self, loop, ticker_obj, ticker_symbol) -> Dict[str, pd.DataFrame]:
        cache_key = f"fin_{ticker_symbol}"
        cached = self._get_from_cache(cache_key)
        if cached:
            return cached
            
        def fetch():
            return {
                "income": ticker_obj.income_stmt,
                "balance": ticker_obj.balance_sheet,
                "cashflow": ticker_obj.cashflow
            }
            
        financials = await loop.run_in_executor(self._executor, fetch)
        self._set_cache(cache_key, financials, self.TTL_FINANCIALS)
        return financials

    async def _fetch_history(self, loop, ticker_obj, ticker_symbol) -> pd.DataFrame:
        cache_key = f"hist_{ticker_symbol}"
        cached = self._get_from_cache(cache_key)
        if cached:
            return cached
            
        def fetch():
            return ticker_obj.history(period="2y")
            
        history = await loop.run_in_executor(self._executor, fetch)
        self._set_cache(cache_key, history, self.TTL_PRICE)
        return history

    def _get_from_cache(self, key: str) -> Optional[Any]:
        if key in self._cache:
            if datetime.now() < self._cache_expiry[key]:
                # logger.debug(f"Cache hit for {key}")
                return self._cache[key]
            else:
                del self._cache[key]
                del self._cache_expiry[key]
        return None

    def _set_cache(self, key: str, value: Any, ttl: timedelta):
        self._cache[key] = value
        self._cache_expiry[key] = datetime.now() + ttl
