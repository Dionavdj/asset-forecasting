"""
Test script to load data and run EDA analysis.
"""
import os
from src.data_loader import fetch_yfinance
from src.eda import (
    calculate_returns, 
    calculate_log_returns,
    calculate_volatility,
    plot_price,
    plot_returns_distribution,
    plot_volatility
)

TICKER = os.getenv("TICKER", "TSLA")

def main():
    print(f"Loading data for {TICKER}...")
    data = fetch_yfinance(TICKER, period="2y", cache_only=False)
    
    if data.empty:
        print("ERROR: No data loaded!")
        return
    
    print(f"Successfully loaded {len(data)} rows")
    print(f"Date range: {data.index[0]} to {data.index[-1]}")
    
    # Calculate metrics
    returns = calculate_returns(data)
    log_returns = calculate_log_returns(data)
    vol_21d = calculate_volatility(returns, window=21)
    
    print(f"\nReturns statistics:")
    print(f"  Mean: {returns.mean():.6f}")
    print(f"  Std: {returns.std():.6f}")
    print(f"  Min: {returns.min():.6f}")
    print(f"  Max: {returns.max():.6f}")
    
    print(f"\nGenerating plots...")
    plot_price(data, TICKER)
    plot_returns_distribution(returns, TICKER)
    plot_volatility(data['Close'], returns, TICKER)
    
    print("\nEDA complete!")

if __name__ == "__main__":
    main()