"""
Data loading helpers for fetching stock data.
"""
import os
import time
import random
import logging
import pandas as pd
import yfinance as yf
from typing import Optional
from datetime import datetime, timedelta

# Suppress yfinance warnings
logging.getLogger('yfinance').setLevel(logging.CRITICAL)

# Date range: 14.12.2020 to 12.12.2025 (inclusive)
START_DATE = datetime(2020, 12, 14)
END_DATE = datetime(2025, 12, 12)
MAX_DATE = END_DATE


def _normalize_index(data: pd.DataFrame) -> pd.DataFrame:
    """Convert index to timezone-naive datetime if needed."""
    if data.empty:
        return data
    if not isinstance(data.index, pd.DatetimeIndex):
        data.index = pd.to_datetime(data.index)
    if hasattr(data.index, 'tz') and data.index.tz is not None:
        data.index = data.index.tz_localize(None)
    return data


def _filter_to_date_range(data: pd.DataFrame) -> pd.DataFrame:
    """Filter data to START_DATE to END_DATE (inclusive)."""
    if data.empty:
        return data
    data = _normalize_index(data)
    mask = (data.index >= START_DATE) & (data.index <= END_DATE)
    return data[mask]


def _get_cache_path(ticker: str) -> str:
    """Get cache file path."""
    cache_dir = os.path.join(os.getcwd(), "data", "raw")
    os.makedirs(cache_dir, exist_ok=True)
    return os.path.join(cache_dir, f"yfinance_cache_{ticker}.csv")


def _load_from_cache(ticker: str) -> Optional[pd.DataFrame]:
    """Load data from cache if it exists."""
    cache_file = _get_cache_path(ticker)
    if not os.path.exists(cache_file):
        return None
    
    try:
        data = pd.read_csv(cache_file, index_col=0, parse_dates=True)
        data = _normalize_index(data)
        data = _filter_to_date_range(data)
        print(f"[Cache] Loaded {ticker} data from cache ({len(data)} rows)")
        return data
    except Exception as e:
        print(f"[Cache] Error loading cache: {e}")
        return None


def _save_to_cache(ticker: str, data: pd.DataFrame):
    """Save downloaded data to cache."""
    if data.empty:
        return
    try:
        data_to_save = _filter_to_date_range(data.copy())
        data_to_save = _normalize_index(data_to_save)
        data_to_save.to_csv(_get_cache_path(ticker))
        print(f"[Cache] Saved {ticker} data to cache ({len(data_to_save)} rows)")
    except Exception as e:
        print(f"[Cache] Error saving cache: {e}")


def _fetch_data(ticker: str, interval: str) -> pd.DataFrame:
    """Fetch data from Yahoo Finance using the most reliable method."""
    request_start = (START_DATE - timedelta(days=7)).strftime("%Y-%m-%d")
    request_end = (END_DATE + timedelta(days=1)).strftime("%Y-%m-%d")  # yfinance end is exclusive
    
    try:
        ticker_obj = yf.Ticker(ticker)
        data = ticker_obj.history(start=request_start, end=request_end, interval=interval)
        if not data.empty:
            data = _filter_to_date_range(data)
        return data
    except Exception:
        # Fallback: use period parameter
        try:
            ticker_obj = yf.Ticker(ticker)
            data = ticker_obj.history(period="5y", interval=interval)
            if not data.empty:
                data = _filter_to_date_range(data)
            return data
        except Exception:
            return pd.DataFrame()


def fetch_yfinance(
    ticker: str,
    period: str = "2y",
    interval: str = "1d",
    use_cache: bool = True,
    cache_only: bool = False,
    max_retries: int = 3,
) -> pd.DataFrame:
    """
    Fetch stock data using yfinance library with caching support.
    Data is automatically filtered to 2020-12-14 to 2025-12-12 (inclusive).
    
    Args:
        ticker: Stock ticker symbol (e.g., 'TSLA')
        period: Not used, kept for compatibility
        interval: Data interval (e.g., '1d', '1wk', '1mo')
        use_cache: If True, use cached data if available, and cache new downloads
        cache_only: If True, never download; return cached data or empty DataFrame
        max_retries: Maximum number of retry attempts
        
    Returns:
        DataFrame with stock data filtered to START_DATE to END_DATE
    """
    # Try cache first
    if use_cache or cache_only:
        cached_data = _load_from_cache(ticker)
        if cached_data is not None:
            return cached_data
    
    if cache_only:
        print(f"[Cache] Cache-only mode enabled but no cache found for {ticker}.")
        return pd.DataFrame()
    
    # Add initial delay to avoid rate limiting
    time.sleep(1 + random.uniform(0, 1))
    
    # Retry logic
    for attempt in range(max_retries):
        try:
            if attempt > 0:
                wait_time = (5 * (2 ** attempt)) + random.uniform(0, 2)
                print(f"[Download] Retry attempt {attempt + 1}/{max_retries} after {wait_time:.1f}s...")
                time.sleep(wait_time)
            
            print(f"[Download] Fetching {ticker} data from Yahoo Finance...")
            data = _fetch_data(ticker, interval)
            
            if data.empty:
                print(f"[Download] No data returned for {ticker}")
                if attempt < max_retries - 1:
                    continue
                print(f"[Download] Failed to fetch {ticker} after {max_retries} attempts")
                return pd.DataFrame()
            
            # Ensure Adj Close column exists
            if "Adj Close" not in data.columns and "Close" in data.columns:
                data["Adj Close"] = data["Close"]
            
            # Save to cache
            if use_cache:
                _save_to_cache(ticker, data)
            
            print(f"[Download] Successfully downloaded {len(data)} rows for {ticker}")
            return data
            
        except Exception as e:
            print(f"[Download] Error (attempt {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                continue
            return pd.DataFrame()
    
    return pd.DataFrame()
