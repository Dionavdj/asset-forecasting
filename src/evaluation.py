"""
Evaluation metrics for forecasting models.
"""
import numpy as np
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