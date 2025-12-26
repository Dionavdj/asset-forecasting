"""
Basic exploratory data analysis for stock data.
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional
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
    
    # Use non-interactive backend to avoid GUI issues
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    
    plt.figure(figsize=(10, 6))
    plt.plot(data.index, data['Close'], label='Close Price', linewidth=1.5)
    plt.title(f'{ticker} Stock Price Over Time')
    plt.xlabel('Date')
    plt.ylabel('Price ($)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Always save the plot (non-blocking)
    if save_path is None:
        # Default save path if not provided
        os.makedirs('results', exist_ok=True)
        save_path = f'results/{ticker}_price.png'
    
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved plot to {save_path}")
    plt.close()