import asyncio
import concurrent.futures
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
import yfinance as yf
import pandas as pd
from core.adapters.base import MarketDataProvider

# Import infrastructure
from core.structured_logger import get_structured_logger
from core.error_handler import with_retry, handle_gracefully
from core.rate_limiter import get_yfinance_rate_limiter

logger = get_structured_logger("CoreYFinanceProvider")

class YFinanceProvider(MarketDataProvider):
    """
    YFinance implementation of MarketDataProvider.
    """
    
    _instance = None
    _cache: Dict[str, Any] = {}
    _cache_expiry: Dict[str, datetime] = {}
    _executor = concurrent.futures.ThreadPoolExecutor(max_workers=10)
    _rate_limiter = None
    
    # Cache TTL settings
    TTL_INFO = timedelta(minutes=60)
    TTL_PRICE = timedelta(minutes=1)
    TTL_FINANCIALS = timedelta(days=1)

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(YFinanceProvider, cls).__new__(cls)
            cls._instance._rate_limiter = get_yfinance_rate_limiter()
        return cls._instance

    async def get_ticker_data(self, ticker: str) -> Dict[str, Any]:
        logger.info(f"Fetching data for {ticker} via YFinance...")
        
        loop = asyncio.get_running_loop()
        await self._rate_limiter.acquire()
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
            self._set_cache(cache_key, rate, self.TTL_INFO)
            return rate
        except Exception as e:
            logger.error(f"Error fetching exchange rate {pair}: {e}")
            return 1.0

    async def get_history(self, ticker: str, period: str = "2y") -> pd.DataFrame:
        loop = asyncio.get_running_loop()
        t = yf.Ticker(ticker)
        return await self._fetch_history(loop, t, ticker)

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
            if isinstance(data, pd.DataFrame):
                if not data.empty:
                    return data.iloc[0].to_dict()
                return {}
            return data
            
        info = await loop.run_in_executor(self._executor, fetch)
        if not isinstance(info, dict):
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
        if cached is not None:
            return cached
            
        @with_retry(max_retries=2, base_delay=0.5, on_data_error=pd.DataFrame())
        def fetch():
            return ticker_obj.history(period="2y")
            
        history = await loop.run_in_executor(self._executor, fetch)
        self._set_cache(cache_key, history, self.TTL_PRICE)
        return history

    async def get_estimates(self, ticker: str) -> pd.DataFrame:
        """
        Fetches earnings estimates for the ticker.
        """
        loop = asyncio.get_running_loop()
        t = yf.Ticker(ticker)
        return await self._fetch_estimates(loop, t, ticker)

    async def _fetch_estimates(self, loop, ticker_obj, ticker_symbol) -> pd.DataFrame:
        cache_key = f"est_{ticker_symbol}"
        cached = self._get_from_cache(cache_key)
        if cached is not None:
            return cached
            
        @with_retry(max_retries=2, base_delay=0.5, on_data_error=pd.DataFrame())
        def fetch():
            # yfinance property is earnings_estimate
            try:
                return ticker_obj.earnings_estimate
            except Exception:
                return pd.DataFrame()
            
        estimates = await loop.run_in_executor(self._executor, fetch)
        if estimates is None:
            estimates = pd.DataFrame()
            
        self._set_cache(cache_key, estimates, self.TTL_FINANCIALS)
        return estimates

    def _get_from_cache(self, key: str) -> Optional[Any]:
        if key in self._cache:
            if datetime.now() < self._cache_expiry[key]:
                return self._cache[key]
            else:
                del self._cache[key]
                del self._cache_expiry[key]
        return None

    def _set_cache(self, key: str, value: Any, ttl: timedelta):
        self._cache[key] = value
        self._cache_expiry[key] = datetime.now() + ttl
