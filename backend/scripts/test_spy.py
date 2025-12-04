import yfinance as yf
import pandas as pd

def test_spy():
    print("Testing yf.Ticker('SPY').history()...")
    try:
        spy = yf.Ticker("SPY")
        hist = spy.history(period="2y")
        print(f"Shape: {hist.shape}")
        print(hist.head())
    except Exception as e:
        print(f"Ticker Error: {e}")

    print("\nTesting yf.download('SPY')...")
    try:
        data = yf.download("SPY", period="2y", progress=False)
        print(f"Shape: {data.shape}")
        print(data.head())
    except Exception as e:
        print(f"Download Error: {e}")

if __name__ == "__main__":
    test_spy()
