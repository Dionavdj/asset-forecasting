"""
Forecasting models for stock returns.
"""
import numpy as np
import pandas as pd
from statsmodels.tsa.ar_model import AutoReg


def train_ar1(returns: pd.Series):
    """Train AR(1) model on returns."""
    # Remove NaN values
    clean_returns = returns.dropna()
    
    if len(clean_returns) < 10:
        return None
    
    try:
        model = AutoReg(clean_returns, lags=1)
        fitted = model.fit()
        return fitted
    except Exception as e:
        print(f"Error training AR(1): {e}")
        return None


def forecast_ar1(model, n_periods: int) -> np.ndarray:
    """Forecast using AR(1) model."""
    if model is None:
        return np.array([])
    
    try:
        forecast = model.forecast(steps=n_periods)
        return forecast
    except Exception as e:
        print(f"Error forecasting with AR(1): {e}")
        return np.array([])