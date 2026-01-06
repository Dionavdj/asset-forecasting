"""
Test script to run EDA and evaluate all forecasting models.
"""
import os
from src.data_loader import fetch_yfinance
from src.eda import (
    calculate_returns,
    plot_price,
    plot_returns_distribution,
    plot_volatility,
    plot_volume
)
from src.models import (
    train_ar1, forecast_ar1,
    train_arima, forecast_arima,
    train_ridge, train_lasso, train_elasticnet,
    forecast_linear,
    train_xgb, forecast_xgb,
    random_walk_baseline
)
from src.evaluation import calculate_rmse, calculate_mae, calculate_mape, train_test_split

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
    plot_volume(data, TICKER)
    
    # Model testing
    print("\n=== Model Testing ===")
    train_returns, test_returns = train_test_split(returns.dropna(), test_size=0.2)
    
    print(f"Train size: {len(train_returns)}, Test size: {len(test_returns)}")
    
    n_forecast = min(20, len(test_returns))
    actual = test_returns.values[:n_forecast]
    
    results = []
    
    # Test AR(1)
    print("\nTraining AR(1)...")
    ar1_model = train_ar1(train_returns)
    if ar1_model is not None:
        forecast = forecast_ar1(ar1_model, n_forecast)
        rmse = calculate_rmse(actual, forecast)
        mae = calculate_mae(actual, forecast)
        mape = calculate_mape(actual, forecast)
        results.append(("AR(1)", rmse, mae, mape))
        print(f"AR(1) - RMSE: {rmse:.6f}, MAE: {mae:.6f}, MAPE: {mape:.4f}")
    
    # Test ARIMA
    print("\nTraining ARIMA...")
    arima_model = train_arima(train_returns)
    if arima_model is not None:
        forecast = forecast_arima(arima_model, n_forecast)
        rmse = calculate_rmse(actual, forecast)
        mae = calculate_mae(actual, forecast)
        mape = calculate_mape(actual, forecast)
        results.append(("ARIMA", rmse, mae, mape))
        print(f"ARIMA - RMSE: {rmse:.6f}, MAE: {mae:.6f}, MAPE: {mape:.4f}")
    
    # Test Ridge
    print("\nTraining Ridge...")
    ridge_model, ridge_scaler = train_ridge(train_returns)
    if ridge_model is not None:
        forecast = forecast_linear(ridge_model, ridge_scaler, train_returns, n_forecast)
        rmse = calculate_rmse(actual, forecast)
        mae = calculate_mae(actual, forecast)
        mape = calculate_mape(actual, forecast)
        results.append(("Ridge", rmse, mae, mape))
        print(f"Ridge - RMSE: {rmse:.6f}, MAE: {mae:.6f}, MAPE: {mape:.4f}")
    
    # Test XGBoost
    print("\nTraining XGBoost...")
    xgb_model = train_xgb(train_returns)
    if xgb_model is not None:
        forecast = forecast_xgb(xgb_model, train_returns, n_forecast)
        rmse = calculate_rmse(actual, forecast)
        mae = calculate_mae(actual, forecast)
        mape = calculate_mape(actual, forecast)
        results.append(("XGBoost", rmse, mae, mape))
        print(f"XGBoost - RMSE: {rmse:.6f}, MAE: {mae:.6f}, MAPE: {mape:.4f}")
    
    # Test baseline
    print("\nTesting baseline...")
    baseline_forecast = random_walk_baseline(train_returns, n_forecast)
    baseline_rmse = calculate_rmse(actual, baseline_forecast)
    baseline_mae = calculate_mae(actual, baseline_forecast)
    baseline_mape = calculate_mape(actual, baseline_forecast)
    results.append(("Baseline", baseline_rmse, baseline_mae, baseline_mape))
    print(f"Baseline - RMSE: {baseline_rmse:.6f}, MAE: {baseline_mae:.6f}, MAPE: {baseline_mape:.4f}")
    
    # Print summary
    print("\n=== Summary ===")
    print(f"{'Model':<12} {'RMSE':<12} {'MAE':<12} {'MAPE':<12}")
    print("-" * 50)
    for name, rmse, mae, mape in results:
        print(f"{name:<12} {rmse:<12.6f} {mae:<12.6f} {mape:<12.4f}")

if __name__ == "__main__":
    main()