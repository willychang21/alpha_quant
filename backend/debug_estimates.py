import yfinance as yf
import pandas as pd

def check_estimates(ticker):
    t = yf.Ticker(ticker)
    print(f"--- {ticker} Earnings Estimate ---")
    try:
        est = t.earnings_estimate
        print(est)
    except Exception as e:
        print(f"Error fetching earnings_estimate: {e}")

    print(f"\n--- {ticker} EPS Trend ---")
    try:
        trend = t.eps_trend
        print(trend)
    except Exception as e:
        print(f"Error fetching eps_trend: {e}")

if __name__ == "__main__":
    check_estimates("AAPL")
