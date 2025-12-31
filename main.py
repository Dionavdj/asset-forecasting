"""
Test script to run EDA and test forecasting models.
"""
import os
from src.data_loader import fetch_yfinance
from src.eda import (
    calculate_returns,
    plot_price,
    plot_returns_distribution,
    plot_volatility
)
from src.models import train_ar1, forecast_ar1, random_walk_baseline
from src.evaluation import calculate_rmse, calculate_mae

TICKER = os.getenv("TICKER", "TSLA")

def main():
    print(f"Loading data for {TICKER}...")
    data = fetch_yfinance(TICKER, period="2y", cache_only=False)
    
    if data.empty:
        print("ERROR: No data loaded!")
        return
    
    print(f"Successfully loaded {len(data)} rows")
    print(f"Date range: {data.index[0]} to {data.index[-1]}")
    
    # EDA
    print("\n=== EDA ===")
    returns = calculate_returns(data)
    print(f"Returns statistics:")
    print(f"  Mean: {returns.mean():.6f}, Std: {returns.std():.6f}")
    
    print("\nGenerating EDA plots...")
    plot_price(data, TICKER)
    plot_returns_distribution(returns, TICKER)
    plot_volatility(data['Close'], returns, TICKER)
    
    # Model testing
    print("\n=== Model Testing ===")
    split_idx = int(len(returns) * 0.8)
    train_returns = returns.iloc[:split_idx].dropna()
    test_returns = returns.iloc[split_idx:].dropna()
    
    print(f"Train size: {len(train_returns)}, Test size: {len(test_returns)}")
    
    n_forecast = min(20, len(test_returns))
    actual = test_returns.values[:n_forecast]
    
    # Test AR(1)
    print("\nTraining AR(1) model...")
    ar1_model = train_ar1(train_returns)
    if ar1_model is not None:
        forecast = forecast_ar1(ar1_model, n_forecast)
        rmse = calculate_rmse(actual, forecast)
        mae = calculate_mae(actual, forecast)
        print(f"AR(1) - RMSE: {rmse:.6f}, MAE: {mae:.6f}")
    
    # Test baseline
    print("\nTesting baseline...")
    baseline_forecast = random_walk_baseline(train_returns, n_forecast)
    baseline_rmse = calculate_rmse(actual, baseline_forecast)
    baseline_mae = calculate_mae(actual, baseline_forecast)
    print(f"Baseline - RMSE: {baseline_rmse:.6f}, MAE: {baseline_mae:.6f}")

if __name__ == "__main__":
    main()