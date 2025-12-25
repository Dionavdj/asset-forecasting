"""
Simple test script to verify data loading works.
"""
from src.data_loader import fetch_yfinance

TICKER = "TSLA"

def main():
    print(f"Testing data loading for {TICKER}...")
    data = fetch_yfinance(TICKER, period="2y", cache_only=True)
    
    if data.empty:
        print("ERROR: No data loaded!")
        return
    
    print(f"Successfully loaded {len(data)} rows")
    print(f"Date range: {data.index[0]} to {data.index[-1]}")
    print(f"\nFirst few rows:")
    print(data.head())
    print(f"\nLast few rows:")
    print(data.tail())

if __name__ == "__main__":
    main()