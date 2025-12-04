import yfinance as yf
import pandas as pd
from sqlalchemy.orm import Session
from datetime import datetime, timedelta
import logging
from quant.data.db_ops import get_or_create_security, bulk_upsert_market_data

logger = logging.getLogger(__name__)

class IngestionService:
    def __init__(self, db: Session):
        self.db = db

    def fetch_sp500_tickers(self) -> list[str]:
        """
        Fetch S&P 500 tickers.
        For MVP, we return a static list of top 50 to save time/bandwidth, 
        or scrape Wikipedia if needed. Let's use a robust static list for now.
        """
        # Top 50 by weight approx
        return [
            "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "TSLA", "BRK-B", "LLY", "AVGO",
            "JPM", "XOM", "UNH", "V", "PG", "MA", "COST", "JNJ", "HD", "MRK",
            "ABBV", "CVX", "BAC", "WMT", "CRM", "AMD", "PEP", "KO", "NFLX", "TMO",
            "ADBE", "WFC", "LIN", "ACN", "MCD", "DIS", "CSCO", "ABT", "INTC", "VZ",
            "CMCSA", "INTU", "QCOM", "IBM", "TXN", "AMGN", "NOW", "GE", "SPGI", "CAT"
        ]

    def ingest_daily_data(self, tickers: list[str], lookback_days: int = 365 * 2):
        """
        Fetch and store daily data for tickers.
        """
        logger.info(f"Starting ingestion for {len(tickers)} tickers...")
        
        # Batch processing to avoid yfinance limits/timeouts
        batch_size = 10
        for i in range(0, len(tickers), batch_size):
            batch = tickers[i:i+batch_size]
            self._process_batch(batch, lookback_days)
            
    def _process_batch(self, tickers: list[str], lookback_days: int):
        try:
            # Download data
            start_date = (datetime.now() - timedelta(days=lookback_days)).strftime('%Y-%m-%d')
            data = yf.download(tickers, start=start_date, group_by='ticker', progress=False, threads=True)
            
            records = []
            
            for ticker in tickers:
                # Handle single ticker vs multi-ticker structure in yfinance
                if len(tickers) == 1:
                    df = data
                else:
                    if ticker not in data.columns.levels[0]:
                        logger.warning(f"No data for {ticker}")
                        continue
                    df = data[ticker]
                
                if df.empty:
                    continue
                
                # Get Security ID
                security = get_or_create_security(self.db, ticker)
                
                # Reset index to get Date column
                df = df.reset_index()
                
                for _, row in df.iterrows():
                    # Ensure we have valid data
                    if pd.isna(row['Close']):
                        continue
                        
                    records.append({
                        'sid': security.sid,
                        'date': row['Date'].date(),
                        'open': row['Open'],
                        'high': row['High'],
                        'low': row['Low'],
                        'close': row['Close'],
                        'adj_close': row['Close'], # yfinance 'Close' is often adj close depending on settings, but let's assume Close for now. 
                        # actually yfinance auto-adjusts. 'Adj Close' column exists usually.
                        # Let's check columns.
                        'volume': int(row['Volume'])
                    })
            
            # Bulk upsert
            if records:
                bulk_upsert_market_data(self.db, records)
                
        except Exception as e:
            logger.error(f"Error processing batch {tickers}: {e}")

    def ingest_fundamentals(self, tickers: list[str]):
        """
        Fetch and store fundamental data (Balance Sheet, Income Statement, Cash Flow).
        """
        logger.info(f"Starting fundamental ingestion for {len(tickers)} tickers...")
        
        from quant.data.db_ops import bulk_upsert_fundamentals
        
        for ticker in tickers:
            try:
                t = yf.Ticker(ticker)
                security = get_or_create_security(self.db, ticker)
                
                records = []
                
                # We'll fetch annual financials for simplicity in MVP
                # yfinance returns DataFrame with columns as Dates
                
                financials = t.financials
                balance_sheet = t.balance_sheet
                cashflow = t.cashflow
                
                # Helper to process dataframe
                def process_df(df, statement_type):
                    if df is None or df.empty:
                        return
                    
                    # Transpose so dates are rows
                    df_T = df.T
                    for date, row in df_T.iterrows():
                        # date is Timestamp
                        date_val = date.date()
                        
                        for metric, value in row.items():
                            if pd.isna(value):
                                continue
                                
                            records.append({
                                'sid': security.sid,
                                'date': date_val,
                                'metric': metric, # e.g. 'Total Revenue'
                                'value': float(value),
                                'period': '12M'
                            })

                process_df(financials, 'income')
                process_df(balance_sheet, 'balance')
                process_df(cashflow, 'cashflow')
                
                if records:
                    bulk_upsert_fundamentals(self.db, records)
                    logger.info(f"Ingested {len(records)} fundamental records for {ticker}")
                    
            except Exception as e:
                logger.error(f"Error fetching fundamentals for {ticker}: {e}")
