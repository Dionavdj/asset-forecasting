"""
Simple test script to load data and plot prices.
"""
import os
from src.data_loader import fetch_yfinance
from src.eda import plot_price

TICKER = os.getenv("TICKER", "TSLA")

def main():
    print(f"Loading data for {TICKER}...")
    data = fetch_yfinance(TICKER, period="2y", cache_only=False)
    
    if data.empty:
        print("ERROR: No data loaded!")
        return
    
    print(f"Successfully loaded {len(data)} rows")
    print(f"Date range: {data.index[0]} to {data.index[-1]}")
    
    print("\nPlotting price...")
    plot_price(data, TICKER)

if __name__ == "__main__":
    main()