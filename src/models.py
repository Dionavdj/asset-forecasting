"""
Forecasting models for stock returns.
"""
import numpy as np
import pandas as pd
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.arima.model import ARIMA
from pmdarima import auto_arima
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


def random_walk_baseline(returns: pd.Series, n_periods: int) -> np.ndarray:
    """Random walk with drift - simple baseline model."""
    clean_returns = returns.dropna()
    
    if len(clean_returns) == 0:
        return np.zeros(n_periods)
    
    # Mean return (drift)
    mean_return = clean_returns.mean()
    
    # Forecast: just use mean return for all periods
    forecast = np.full(n_periods, mean_return)
    return forecast


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


def train_arima(returns: pd.Series):
    """Train ARIMA model with auto-selected order."""
    clean_returns = returns.dropna()
    
    if len(clean_returns) < 20:
        return None
    
    try:
        # Use auto_arima to find best order
        auto_model = auto_arima(
            clean_returns,
            start_p=0, start_q=0,
            max_p=3, max_q=3,
            seasonal=False,
            stepwise=True,
            suppress_warnings=True,
            error_action='ignore',
            random_state=42,
            n_jobs=1
        )
        
        # Get the order and fit with statsmodels for consistency
        order = auto_model.order
        model = ARIMA(clean_returns, order=order)
        fitted = model.fit()
        return fitted
    except Exception as e:
        print(f"Error training ARIMA: {e}")
        # Fallback to simple AR(1)
        try:
            model = ARIMA(clean_returns, order=(1, 0, 0))
            fitted = model.fit()
            return fitted
        except:
            return None


def forecast_arima(model, n_periods: int) -> np.ndarray:
    """Forecast using ARIMA model."""
    if model is None:
        return np.array([])
    
    try:
        forecast = model.forecast(steps=n_periods)
        return forecast
    except Exception as e:
        print(f"Error forecasting with ARIMA: {e}")
        return np.array([])
        

def train_ridge(returns: pd.Series, lags: int = 5, alpha: float = 1.0):
    """Train Ridge regression model with lag features."""
    clean_returns = returns.dropna()
    
    if len(clean_returns) < lags + 10:
        return None, None
    
    # Create lag features
    X = []
    y = []
    
    for i in range(lags, len(clean_returns)):
        features = [clean_returns.iloc[i - j] for j in range(1, lags + 1)]
        X.append(features)
        y.append(clean_returns.iloc[i])
    
    X = np.array(X)
    y = np.array(y)
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    try:
        model = Ridge(alpha=alpha, random_state=42)
        model.fit(X_scaled, y)
        return model, scaler
    except Exception as e:
        print(f"Error training Ridge: {e}")
        return None, None


def forecast_ridge(model, scaler, returns: pd.Series, n_periods: int, lags: int = 5) -> np.ndarray:
    """Forecast using Ridge model."""
    if model is None or scaler is None:
        return np.array([])
    
    clean_returns = returns.dropna()
    
    if len(clean_returns) < lags:
        return np.array([])
    
    forecast = []
    current_series = clean_returns.copy()
    
    try:
        for _ in range(n_periods):
            # Get last lags values
            features = [current_series.iloc[-j] for j in range(1, lags + 1)]
            features = np.array(features).reshape(1, -1)
            features_scaled = scaler.transform(features)
            
            # Predict next value
            pred = model.predict(features_scaled)[0]
            forecast.append(pred)
            
            # Add to series for next prediction
            current_series = pd.concat([current_series, pd.Series([pred])])
        
        return np.array(forecast)
    except Exception as e:
        print(f"Error forecasting with Ridge: {e}")
        return np.array([])