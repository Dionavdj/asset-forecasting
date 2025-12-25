"""
Basic exploratory data analysis for stock data.
"""
import pandas as pd
import matplotlib.pyplot as plt
from src.data_loader import fetch_yfinance


def plot_price(data: pd.DataFrame, ticker: str):
    """Plot stock price over time."""
    if data.empty:
        print("No data to plot")
        return
    
    plt.figure(figsize=(10, 6))
    plt.plot(data.index, data['Close'], label='Close Price')
    plt.title(f'{ticker} Stock Price')
    plt.xlabel('Date')
    plt.ylabel('Price ($)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()