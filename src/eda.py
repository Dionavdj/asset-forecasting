"""
Basic exploratory data analysis for stock data.
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Dict
from scipy import stats
from src.data_loader import fetch_yfinance


def calculate_returns(data: pd.DataFrame) -> pd.Series:
    """Calculate daily returns."""
    if 'Close' not in data.columns:
        raise ValueError("Data must have 'Close' column")
    returns = data['Close'].pct_change()
    return returns


def calculate_log_returns(data: pd.DataFrame) -> pd.Series:
    """Calculate log returns."""
    if 'Close' not in data.columns:
        raise ValueError("Data must have 'Close' column")
    log_returns = np.log(data['Close']).diff()
    return log_returns


def calculate_volatility(returns: pd.Series, window: int = 21) -> pd.Series:
    """Calculate rolling volatility (annualized)."""
    volatility = returns.rolling(window=window).std() * np.sqrt(252) * 100
    return volatility


def plot_price(data: pd.DataFrame, ticker: str, save_path: Optional[str] = None):
    """Plot stock price over time."""
    if data.empty:
        print("No data to plot")
        return
    
    import matplotlib
    matplotlib.use('Agg')
    
    plt.figure(figsize=(10, 6))
    plt.plot(data.index, data['Close'], label='Close Price', linewidth=1.5)
    plt.title(f'{ticker} Stock Price Over Time')
    plt.xlabel('Date')
    plt.ylabel('Price ($)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path is None:
        os.makedirs('results', exist_ok=True)
        save_path = f'results/{ticker}_price.png'
    
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved plot to {save_path}")
    plt.close()


def plot_returns_distribution(
    returns: pd.Series, 
    ticker: str, 
    save_path: Optional[str] = None
):
    """Plot returns distribution with histogram and Q-Q plot."""
    if returns.empty or returns.isna().all():
        print("No returns data to plot")
        return
    
    clean_returns = returns.dropna()
    
    import matplotlib
    matplotlib.use('Agg')
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Histogram
    ax1.hist(clean_returns, bins=50, density=True, alpha=0.7, edgecolor='black')
    ax1.set_title(f'{ticker} Returns Distribution')
    ax1.set_xlabel('Returns')
    ax1.set_ylabel('Density')
    ax1.grid(True, alpha=0.3)
    
    # Q-Q plot
    stats.probplot(clean_returns, dist="norm", plot=ax2)
    ax2.set_title(f'{ticker} Q-Q Plot (Normal)')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path is None:
        os.makedirs('results', exist_ok=True)
        save_path = f'results/{ticker}_returns_dist.png'
    
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved plot to {save_path}")
    plt.close()


def plot_volatility(
    prices: pd.Series,
    returns: pd.Series,
    ticker: str,
    save_path: Optional[str] = None
):
    """Plot rolling volatility over time."""
    if returns.empty:
        print("No returns data to plot")
        return
    
    vol_21d = calculate_volatility(returns, window=21)
    vol_252d = calculate_volatility(returns, window=252)
    
    import matplotlib
    matplotlib.use('Agg')
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Price and volatility
    ax1_twin = ax1.twinx()
    ax1.plot(prices.index, prices.values, label='Price', color='blue', linewidth=1)
    ax1_twin.plot(vol_21d.index, vol_21d.values, label='21-day Volatility', color='red', alpha=0.7)
    ax1.set_ylabel('Price ($)', color='blue')
    ax1_twin.set_ylabel('Volatility (%)', color='red')
    ax1.set_title(f'{ticker} Price and Volatility')
    ax1.legend(loc='upper left')
    ax1_twin.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    # Rolling volatility comparison
    ax2.plot(vol_21d.index, vol_21d.values, label='21-day', alpha=0.7)
    ax2.plot(vol_252d.index, vol_252d.values, label='252-day', alpha=0.7)
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Volatility (%)')
    ax2.set_title(f'{ticker} Rolling Volatility')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path is None:
        os.makedirs('results', exist_ok=True)
        save_path = f'results/{ticker}_volatility.png'
    
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved plot to {save_path}")
    plt.close()