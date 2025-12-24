"""
Data loading helpers for fetching stock data.
"""
import os
import pandas as pd
import yfinance as yf
from typing import Optional
from datetime import datetime


# Maximum date for data cutoff
MAX_DATE = datetime(2025, 12, 12)


def _get_cache_filename(ticker: str) -> str:
    """Generate a cache filename based on ticker."""
    return f"yfinance_cache_{ticker}.csv"


def _filter_to_max_date(data: pd.DataFrame) -> pd.DataFrame:
    """Filter data to include only rows up to MAX_DATE (inclusive)."""
    if data.empty:
        return data
    
    # Ensure index is datetime
    if not isinstance(data.index, pd.DatetimeIndex):
        data.index = pd.to_datetime(data.index)
    
    # Filter to max date (inclusive)
    filtered = data[data.index <= MAX_DATE]
    return filtered


def _load_from_cache(ticker: str) -> Optional[pd.DataFrame]:
    """Load data from cache if it exists."""
    cache_dir = os.path.join(os.getcwd(), "data", "raw")
    cache_file = os.path.join(cache_dir, _get_cache_filename(ticker))
    
    if os.path.exists(cache_file):
        try:
            data = pd.read_csv(cache_file, index_col=0, parse_dates=True)
            # Convert index to datetime if it's not already
            if not isinstance(data.index, pd.DatetimeIndex):
                data.index = pd.to_datetime(data.index)
            # Convert to timezone-naive if needed
            if hasattr(data.index, 'tz') and data.index.tz is not None:
                data.index = data.index.tz_localize(None)
            
            # Filter to max date
            data = _filter_to_max_date(data)
            
            print(f"[Cache] Loaded {ticker} data from cache ({len(data)} rows, filtered to <= {MAX_DATE.date()})")
            return data
        except Exception as e:
            print(f"[Cache] Error loading cache: {e}")
            return None
    return None


def _save_to_cache(ticker: str, data: pd.DataFrame):
    """Save downloaded data to cache."""
    if data.empty:
        return
    
    # Filter to max date before saving
    data = _filter_to_max_date(data)
    
    cache_dir = os.path.join(os.getcwd(), "data", "raw")
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, _get_cache_filename(ticker))
    
    try:
        # Make a copy to avoid modifying original
        data_to_save = data.copy()
        # Convert timezone-aware index to naive for CSV storage
        if data_to_save.index.tz is not None:
            data_to_save.index = data_to_save.index.tz_localize(None)
        data_to_save.to_csv(cache_file)
        print(f"[Cache] Saved {ticker} data to cache ({len(data)} rows, filtered to <= {MAX_DATE.date()})")
    except Exception as e:
        print(f"[Cache] Error saving cache: {e}")


def fetch_yfinance(
    ticker: str,
    period: str = "2y",
    interval: str = "1d",
    use_cache: bool = True,
    cache_only: bool = False,
) -> pd.DataFrame:
    """
    Fetch stock data using yfinance library with caching support.
    Data is automatically filtered to stop at 2025-12-12 (inclusive).
    
    Args:
        ticker: Stock ticker symbol (e.g., 'TSLA')
        period: Download period (e.g., '1y', '2y', '3y')
        interval: Data interval (e.g., '1d', '1wk', '1mo')
        use_cache: If True, use cached data if available, and cache new downloads
        cache_only: If True, never download; return cached data or empty DataFrame.
        
    Returns:
        DataFrame with stock data (columns: Open, High, Low, Close, Volume, etc.)
        Filtered to dates <= 2025-12-12
    """
    # Try to load from cache first
    if use_cache or cache_only:
        cached_data = _load_from_cache(ticker)
        if cached_data is not None:
            return cached_data
    
    # In cache-only mode, never attempt a download
    if cache_only:
        print(f"[Cache] Cache-only mode enabled but no cache found for {ticker}.")
        return pd.DataFrame()
    
    # Download from Yahoo Finance
    try:
        print(f"[Download] Fetching {ticker} data from Yahoo Finance (period={period}, interval={interval})...")
        ticker_obj = yf.Ticker(ticker)
        data = ticker_obj.history(period=period, interval=interval)
        
        if data.empty:
            print(f"[Download] No data returned for {ticker}")
            return pd.DataFrame()
        
        # Ensure Adj Close column exists
        if "Adj Close" not in data.columns and "Close" in data.columns:
            data["Adj Close"] = data["Close"]
        
        # Filter to max date
        data = _filter_to_max_date(data)
        
        # Save to cache
        if use_cache:
            _save_to_cache(ticker, data)
        
        print(f"[Download] Successfully downloaded {len(data)} rows for {ticker} (filtered to <= {MAX_DATE.date()})")
        return data
        
    except Exception as e:
        print(f"[Download] Error fetching {ticker} data: {e}")
        return pd.DataFrame()