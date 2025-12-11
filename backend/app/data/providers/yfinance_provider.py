"""YFinance Data Provider Module.

Provides functions for fetching stock data from Yahoo Finance:
- Asset search
- Holdings for ETFs
- Price data
- Fundamental data
"""

import yfinance as yf
import pandas as pd
import requests
import json
import os
import math
from typing import Dict, List, Optional, Any

from app.domain import schemas

# Import infrastructure
from core.structured_logger import get_structured_logger
from core.error_handler import with_retry, handle_gracefully
from core.rate_limiter import get_yfinance_rate_limiter

logger = get_structured_logger("YFinanceProvider")

# Get rate limiter instance
_rate_limiter = get_yfinance_rate_limiter()



@with_retry(max_retries=2, base_delay=0.5, on_data_error=[])
def search_assets(query: str) -> List[Dict[str, Any]]:
    """Search for assets on Yahoo Finance.
    
    Args:
        query: Search query string
        
    Returns:
        List of matching assets with symbol, name, exchange info
    """
    # Rate limit
    _rate_limiter.acquire_sync()
    
    url = f"https://query2.finance.yahoo.com/v1/finance/search?q={query}"
    headers = {'User-Agent': 'Mozilla/5.0'}
    r = requests.get(url, headers=headers)
    data = r.json()
    
    results = []
    if 'quotes' in data:
        for q in data['quotes']:
            if q.get('isYahooFinance', False):
                results.append({
                    "symbol": q.get('symbol'),
                    "shortname": q.get('shortname') or q.get('longname') or q.get('symbol'),
                    "exchange": q.get('exchange'),
                    "typeDisp": q.get('typeDisp')
                })
    return results

@handle_gracefully(default=[])
def fetch_holdings(ticker: str) -> List[Dict[str, Any]]:
    """
    Fetch ETF/Fund holdings directly from yfinance.
    
    NOTE: This is real-time data only. Do NOT use for backtesting
    as it would cause Look-Ahead Bias. For backtesting, use ETF
    prices directly instead of decomposing into holdings.
    
    Args:
        ticker: ETF or fund ticker symbol
        
    Returns:
        List of holdings with ticker, name, percent
    """
    _rate_limiter.acquire_sync()
    
    holdings = []
    t = yf.Ticker(ticker)
    
    # Try to get holdings from funds_data
    if hasattr(t, 'funds_data') and t.funds_data and hasattr(t.funds_data, 'top_holdings'):
        top_holdings = t.funds_data.top_holdings
        if isinstance(top_holdings, pd.DataFrame):
            for symbol, row in top_holdings.iterrows():
                percent = 0
                if 'Holding Percent' in row:
                    percent = float(row['Holding Percent'])
                elif 'Holding %' in row:
                    percent = float(row['Holding %'])
                
                holdings.append({
                    "ticker": str(symbol),
                    "name": str(row.get('Name', symbol)),
                    "percent": percent * 100
                })
    
    return holdings


@handle_gracefully(default={"ticker": "", "type": "EQUITY", "holdings": []})
def get_asset_details(ticker: str) -> Dict[str, Any]:
    """Get detailed asset information including holdings.
    
    Args:
        ticker: Stock or ETF ticker symbol
        
    Returns:
        Dict with ticker, type, and holdings
    """
    _rate_limiter.acquire_sync()
    
    t = yf.Ticker(ticker)
    info = t.info
    holdings = fetch_holdings(ticker)
    
    return {
        "ticker": ticker,
        "type": info.get('quoteType', 'EQUITY'),
        "holdings": holdings
    }

def get_price_data(ticker: str):
    try:
        t = yf.Ticker(ticker)
        price = t.fast_info.last_price
        prev_close = t.fast_info.previous_close
        change = price - prev_close
        change_percent = (change / prev_close) * 100 if prev_close else 0
        return {
            "price": price,
            "change": change,
            "change_percent": change_percent,
            "name": ticker, 
            "obj": t
        }
    except Exception as e:
        logger.error(f"Error fetching {ticker}: {e}")
        return None

def get_fundamental_data(ticker: str) -> schemas.FundamentalResponse:
    try:
        t = yf.Ticker(ticker)
        info = t.info
        
        # Extract Data with Fallbacks
        price = info.get('currentPrice') or info.get('regularMarketPrice') or info.get('previousClose') or 0.0
        
        # EPS: Try trailing, then forward, then basic eps
        eps = info.get('trailingEps') or info.get('forwardEps') or info.get('epsTrailingTwelveMonths')
        
        # Book Value
        book_value = info.get('bookValue')
        
        # P/E: Try trailing, then calculate from price/eps
        pe_ratio = info.get('trailingPE')
        if not pe_ratio and price and eps and eps > 0:
            pe_ratio = price / eps
            
        # Dividend Yield
        div_yield = info.get('dividendYield')
        
        # Profit Margins
        profit_margin = info.get('profitMargins')
        
        # ROE
        roe = info.get('returnOnEquity')
        
        # FCF Yield = Free Cash Flow / Market Cap
        fcf = info.get('freeCashflow')
        market_cap = info.get('marketCap')
        fcf_yield = None
        if fcf and market_cap and market_cap > 0:
            fcf_yield = fcf / market_cap
            
        data = schemas.FundamentalData(
            ticker=ticker,
            name=info.get('shortName') or info.get('longName') or ticker,
            sector=info.get('sector', 'Unknown'),
            industry=info.get('industry', 'Unknown'),
            currency=info.get('currency', 'USD'),
            price=price,
            
            peRatio=pe_ratio,
            forwardPE=info.get('forwardPE'),
            pegRatio=info.get('pegRatio') or info.get('trailingPegRatio'),
            pbRatio=info.get('priceToBook'),
            dividendYield=div_yield,
            evToEbitda=info.get('enterpriseToEbitda'),
            
            roe=roe,
            profitMargin=profit_margin,
            eps=eps,
            revenueGrowth=info.get('revenueGrowth'),
            
            debtToEquity=info.get('debtToEquity'),
            currentRatio=info.get('currentRatio'),
            freeCashflow=fcf,
            fcfYield=fcf_yield,
            
            targetMeanPrice=info.get('targetMeanPrice'),
            recommendationKey=info.get('recommendationKey')
        )
        
        # Value Analysis
        # Graham Number = Sqrt(22.5 * EPS * BookValue)
        graham_number = None
        if eps and book_value and eps > 0 and book_value > 0:
            try:
                graham_number = math.sqrt(22.5 * eps * book_value)
            except:
                pass
                
        # Simple Undervalued Logic
        is_undervalued = False
        details = {}
        
        if graham_number and price < graham_number:
            is_undervalued = True
            details['graham_check'] = "Price below Graham Number"
            
        if info.get('pegRatio') and info.get('pegRatio') < 1:
            details['peg_check'] = "PEG < 1 (Potential Value)"
            
        if pe_ratio and pe_ratio < 15:
             details['pe_check'] = "P/E < 15 (Historic Value)"
             
        if roe and roe > 0.15:
             details['roe_check'] = "ROE > 15% (Quality)"

        analysis = schemas.ValueAnalysis(
            grahamNumber=graham_number,
            isUndervalued=is_undervalued,
            details=details
        )
        
        return schemas.FundamentalResponse(data=data, analysis=analysis)
        
    except Exception as e:
        logger.error(f"Fundamental analysis error for {ticker}: {e}")
        # Return empty/default response on error
        return schemas.FundamentalResponse(
            data=schemas.FundamentalData(ticker=ticker, name=ticker, price=0),
            analysis=schemas.ValueAnalysis()
        )
