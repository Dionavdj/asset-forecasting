"""
Basic exploratory data analysis for stock data.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from src.data_loader import fetch_yfinance


def calculate_returns(data: pd.DataFrame) -> pd.Series:
    """Calculate daily returns."""
    if 'Close' not in data.columns:
        raise ValueError("Data must have 'Close' column")
    returns = data['Close'].pct_change()
    return returns


def plot_price(data: pd.DataFrame, ticker: str, save_path: Optional[str] = None):
    """Plot stock price over time."""
    if data.empty:
        print("No data to plot")
        return
    
    plt.figure(figsize=(10, 6))
    plt.plot(data.index, data['Close'], label='Close Price', linewidth=1.5)
    plt.title(f'{ticker} Stock Price Over Time')
    plt.xlabel('Date')
    plt.ylabel('Price ($)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Saved plot to {save_path}")
    else:
        plt.show()
    plt.close()