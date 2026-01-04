"""
Evaluation metrics for forecasting models.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error


def calculate_rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Calculate Root Mean Squared Error."""
    if len(actual) != len(predicted):
        raise ValueError("Arrays must have same length")
    
    mse = mean_squared_error(actual, predicted)
    rmse = np.sqrt(mse)
    return rmse


def calculate_mae(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Calculate Mean Absolute Error."""
    if len(actual) != len(predicted):
        raise ValueError("Arrays must have same length")
    
    mae = mean_absolute_error(actual, predicted)
    return mae


def train_test_split(data: pd.Series, test_size: float = 0.2):
    """Split time series data into train and test sets."""
    if test_size <= 0 or test_size >= 1:
        raise ValueError("test_size must be between 0 and 1")
    
    n = len(data)
    split_idx = int(n * (1 - test_size))
    
    train = data.iloc[:split_idx]
    test = data.iloc[split_idx:]
    
    return train, test