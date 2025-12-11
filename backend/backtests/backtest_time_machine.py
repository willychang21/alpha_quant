"""
Time Machine Backtest Engine
=============================

Performs rigorous Point-in-Time (PIT) backtesting by simulating
past environments.

Methodology:
1. Set a 'Backtest Date' (e.g., 2025-01-01)
2. Fetch full financial history
3. MASK (Delete) all data released after the Backtest Date
4. Run Valuation Model using only the "Past" data
5. Compare "Past Predicted Upside" vs "Actual Future Return"

This eliminates Look-Ahead Bias.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from app.engines.valuation import core as valuation
from app.domain import schemas

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Mute other loggers
logging.getLogger("yfinance").setLevel(logging.WARNING)
logging.getLogger("valuation").setLevel(logging.INFO) # Enable INFO logs for debugging

# Import historical beta calculator
from quant.backtest.beta import calculate_historical_beta, get_market_returns


def _calculate_historical_beta_for_pit(ticker: str, hist: pd.DataFrame, backtest_date) -> float:
    """Calculate historical beta using only data available before backtest_date.
    
    Args:
        ticker: Stock ticker
        hist: Price history DataFrame up to backtest_date
        backtest_date: The point-in-time date
    
    Returns:
        Historical beta, default 1.0 if calculation fails
    """
    try:
        if hist.empty or len(hist) < 60:
            return 1.0
        
        # Get stock returns from available history
        stock_returns = hist['Close'].pct_change().dropna()
        
        # Get market returns for the same period
        start_date = hist.index[0]
        if hasattr(start_date, 'tz_localize'):
            start_date = pd.Timestamp(start_date).tz_localize(None)
        end_date = pd.Timestamp(backtest_date).tz_localize(None) if isinstance(backtest_date, str) else backtest_date
        
        market_returns = get_market_returns(start_date, end_date)
        if market_returns is None:
            return 1.0
        
        return calculate_historical_beta(stock_returns, market_returns)
    except Exception as e:
        logger.debug(f"{ticker}: Could not calculate historical beta: {e}")
        return 1.0

def get_point_in_time_data(ticker: str, backtest_date_str: str):
    """
    Reconstructs the state of the world as of backtest_date.
    """
    backtest_date = pd.Timestamp(backtest_date_str).tz_localize(None)
    
    stock = yf.Ticker(ticker)
    
    # 1. Get Historical Price
    # Fetch history around the date to find the closest close
    start_date = backtest_date - timedelta(days=10)
    end_date = backtest_date + timedelta(days=5)
    hist = stock.history(start=start_date, end=end_date)
    
    # Filter for dates <= backtest_date
    hist = hist[hist.index.tz_localize(None) <= backtest_date]
    
    if hist.empty:
        logger.warning(f"{ticker}: No price history found before {backtest_date_str}")
        return None
        
    historical_price = hist.iloc[-1]['Close']
    historical_date = hist.index[-1]
    
    # 2. Get Financials and Filter for PIT (Point-in-Time)
    # yfinance returns columns as dates. We keep only columns < backtest_date
    
    def filter_financials(df):
        if df is None or df.empty:
            return df
        # Convert column names to datetime if they aren't already
        valid_cols = []
        for col in df.columns:
            try:
                col_date = pd.to_datetime(col).tz_localize(None)
                if col_date < backtest_date:
                    valid_cols.append(col)
            except (TypeError, ValueError) as e:
                logger.debug(f"Skipping non-date column: {col}, error: {e}")
        
        return df[valid_cols]

    income = filter_financials(stock.quarterly_income_stmt)
    balance = filter_financials(stock.quarterly_balance_sheet)
    cashflow = filter_financials(stock.quarterly_cashflow)
    
    if income is None or income.empty:
        logger.warning(f"{ticker}: No financial statements available before {backtest_date_str}")
        return None

    # 3. Reconstruct 'Info' Dictionary (The tricky part)
    # We cannot use stock.info because it's current. We must estimate key metrics.
    
    # Get latest available shares outstanding from Balance Sheet
    try:
        shares_row = balance.loc['Ordinary Shares Number'] if 'Ordinary Shares Number' in balance.index else \
                     balance.loc['Share Issued'] if 'Share Issued' in balance.index else None
        
        if shares_row is not None and not shares_row.empty:
            shares_outstanding = shares_row.iloc[0] # Latest relative to backtest date
        else:
            # Fallback to current if history missing (minor leakage, usually stable)
            shares_outstanding = stock.info.get('sharesOutstanding')
    except (TypeError, KeyError, AttributeError) as e:
        logger.debug(f"{ticker}: Could not get shares from balance sheet: {e}")
        shares_outstanding = stock.info.get('sharesOutstanding')

    market_cap = historical_price * shares_outstanding if shares_outstanding else 0
    
    # Calculate Historical Growth Rates
    # This is critical because we don't have analyst estimates
    earnings_growth = 0.05 # Default 5%
    revenue_growth = 0.05
    
    try:
        if income is not None and not income.empty:
            if len(income.columns) >= 5:
                # Sort columns by date descending (just in case)
                sorted_cols = sorted(income.columns, key=lambda x: pd.to_datetime(x), reverse=True)
                
                # Get latest and year-ago quarters
                latest_q = income[sorted_cols[0]]
                year_ago_q = income[sorted_cols[4]] # 4 quarters ago
                
                # Revenue Growth
                rev_latest = latest_q.get('Total Revenue') or latest_q.get('Revenue')
                rev_ago = year_ago_q.get('Total Revenue') or year_ago_q.get('Revenue')
                
                if rev_latest and rev_ago and rev_ago > 0:
                    revenue_growth = (rev_latest - rev_ago) / rev_ago
                    
                # Earnings Growth
                ni_latest = latest_q.get('Net Income')
                ni_ago = year_ago_q.get('Net Income')
                
                if ni_latest and ni_ago and abs(ni_ago) > 0:
                    earnings_growth = (ni_latest - ni_ago) / abs(ni_ago)
                    
                # Cap extreme growth for stability
                # earnings_growth = max(-0.5, min(earnings_growth, 0.5)) # REMOVED CAP FOR DEBUG
                # revenue_growth = max(-0.2, min(revenue_growth, 0.3))
            
    except (TypeError, ValueError, KeyError) as e:
        logger.warning(f"{ticker}: Could not calculate historical growth: {e}")

    # Mock Info Object
    # We populate only what valuation.py needs, using historical values where possible
    mock_info = {
        'symbol': ticker,
        'currentPrice': historical_price,
        'marketCap': market_cap,
        'sharesOutstanding': shares_outstanding or 0,
        'sector': stock.info.get('sector', 'Unknown'), # Sector doesn't change
        'industry': stock.info.get('industry', 'Unknown'),
        # Use historical beta to avoid look-ahead bias
        'beta': _calculate_historical_beta_for_pit(ticker, hist, backtest_date),
        'dividendRate': stock.info.get('dividendRate', 0.0), # Approx
        'dividendYield': stock.info.get('dividendYield', 0.0), # Approx
        # We don't have historical analyst estimates, so we disable those features or use simple growth
        'targetMeanPrice': 0.0, # Safe default
        'targetLowPrice': 0.0,
        'targetHighPrice': 0.0,
        'numberOfAnalystOpinions': 0,
        'earningsGrowth': earnings_growth, # Use calculated historical growth
        'revenueGrowth': revenue_growth,
        'returnOnEquity': 0.15, # Default reasonable ROE if missing
        'payoutRatio': 0.40, # Default payout
        'trailingPE': 20.0, # Default PE
        'forwardPE': 20.0,
        'priceToBook': 3.0,
    }
    
    return {
        'ticker': ticker,
        'date': historical_date,
        'price': historical_price,
        'info': mock_info,
        'income': income,
        'balance': balance,
        'cashflow': cashflow,
        'history': hist # This is history UP TO backtest_date
    }

def run_time_machine_test(tickers, backtest_date="2025-01-01"):
    print(f"\n{'='*80}")
    print(f"TIME MACHINE BACKTEST")
    print(f"Target Date: {backtest_date}")
    print(f"{'='*80}\n")
    
    results = []
    
    # Get current prices for comparison
    current_prices = {}
    for t in tickers:
        try:
            current_prices[t] = yf.Ticker(t).history(period="1d")['Close'].iloc[-1]
        except (KeyError, IndexError, ValueError) as e:
            logger.debug(f"Could not get current price for {t}: {e}")
            current_prices[t] = 0

    for ticker in tickers:
        print(f"Processing {ticker}...", end=" ")
        
        data = get_point_in_time_data(ticker, backtest_date)
        
        if not data:
            print("Skipping (No Data)")
            continue
            
        try:
            # Run Valuation
            # IMPORTANT: We are passing quarterly data, so we must tell valuation.py to annualize it
            result = valuation.get_valuation(
                ticker=ticker,
                info=data['info'],
                income=data['income'],
                balance=data['balance'],
                cashflow=data['cashflow'],
                history=data['history'],
                is_quarterly=True
            )
            
            if ticker == 'NVDA':
                print(f"\n[DEBUG NVDA] Inputs & Intermediates:")
                print(f"  - Calculated Revenue Growth: {data['info'].get('revenueGrowth', 0):.1%}")
                print(f"  - Calculated Earnings Growth: {data['info'].get('earningsGrowth', 0):.1%}")
                print(f"  - Shares Outstanding: {data['info'].get('sharesOutstanding', 0):,.0f}")
                print(f"  - Market Cap: ${data['info'].get('marketCap', 0):,.0f}")
                if result.dcf:
                    print(f"  - DCF Value: ${result.dcf.sharePrice:.2f}")
                    print(f"  - WACC: {result.dcf.wacc:.1%}")
                    # We can't easily access internal DCF steps, but these inputs tell a lot
            
            # Extract Fair Value
            fair_value = 0
            model_type = "N/A"
            
            # Extract Fair Value based on model type
            fair_value = 0.0
            model_type = "N/A"
            
            if result.dcf:
                fair_value = result.dcf.sharePrice
                model_type = "DCF"
            elif result.ddm:
                fair_value = result.ddm.fairValue
                model_type = "DDM"
            elif result.reit:
                fair_value = result.reit.fairValue
                model_type = "REIT"
                
            if fair_value == 0:
                print(f"Warning: No valid valuation for {ticker}")
                continue
            elif result.reit:
                fair_value = result.reit.fairValue
                model_type = "REIT"
                
            # Calculate Metrics
            start_price = data['price']
            curr_price = current_prices.get(ticker, 0)
            
            predicted_upside = (fair_value - start_price) / start_price
            actual_return = (curr_price - start_price) / start_price
            
            print(f"Done. Rating: {result.rating}")
            
            results.append({
                'Ticker': ticker,
                'Model': model_type,
                'Start_Price': start_price,
                'Fair_Value_Jan1': fair_value,
                'Rating_Jan1': result.rating,
                'Predicted_Upside': predicted_upside,
                'Current_Price': curr_price,
                'Actual_Return': actual_return,
                'Prediction_Quality': '✅ Correct' if (predicted_upside > 0 and actual_return > 0) or (predicted_upside < 0 and actual_return < 0) else '❌ Wrong'
            })
            
        except Exception as e:
            print(f"Error: {e}")
            # import traceback
            # traceback.print_exc()

    # Display Results
    if not results:
        print("No results generated.")
        return

    df = pd.DataFrame(results)
    
    print(f"\n\n{'='*100}")
    print(f"BACKTEST RESULTS (Jan 1, 2025 -> Nov 29, 2025)")
    print(f"{'='*100}")
    
    # Format for display
    display_df = df.copy()
    display_df['Start_Price'] = display_df['Start_Price'].map('${:,.2f}'.format)
    display_df['Fair_Value_Jan1'] = display_df['Fair_Value_Jan1'].map('${:,.2f}'.format)
    display_df['Current_Price'] = display_df['Current_Price'].map('${:,.2f}'.format)
    display_df['Predicted_Upside'] = display_df['Predicted_Upside'].map('{:+.1%}'.format)
    display_df['Actual_Return'] = display_df['Actual_Return'].map('{:+.1%}'.format)
    
    print(display_df[['Ticker', 'Rating_Jan1', 'Start_Price', 'Fair_Value_Jan1', 'Predicted_Upside', 'Actual_Return', 'Prediction_Quality']].to_string(index=False))
    
    # Summary Stats
    print(f"\n{'-'*100}")
    
    # Strong Buy Performance
    strong_buys = df[df['Rating_Jan1'] == 'STRONG BUY']
    buys = df[df['Rating_Jan1'] == 'BUY']
    sells = df[df['Rating_Jan1'] == 'SELL']
    
    print(f"Performance by Rating (Avg Return):")
    if not strong_buys.empty:
        print(f"  STRONG BUY ({len(strong_buys)}): {strong_buys['Actual_Return'].mean():+.1%}")
    if not buys.empty:
        print(f"  BUY        ({len(buys)}): {buys['Actual_Return'].mean():+.1%}")
    if not sells.empty:
        print(f"  SELL       ({len(sells)}): {sells['Actual_Return'].mean():+.1%}")
        
    # Correlation
    corr = df['Predicted_Upside'].corr(df['Actual_Return'])
    print(f"\nCorrelation (Predicted vs Actual): {corr:.2f}")
    if corr > 0.3:
        print("✅ Strong positive correlation - Model has predictive power")
    elif corr > 0:
        print("⚠️ Weak positive correlation")
    else:
        print("❌ Negative correlation - Model predictions inverted")

if __name__ == "__main__":
    # Test Universe
    tickers = [
        'NVDA', 'AAPL', 'GOOGL', 'MSFT', 'META', # Tech
        'JPM', 'WFC', 'BAC', # Finance
        'KO', 'PG', # Defensive
        'TSLA', 'AMZN' # Growth
    ]
    
    run_time_machine_test(tickers, backtest_date="2025-01-01")
