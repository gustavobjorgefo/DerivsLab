# src/derivslab/utils/utils.py

from __future__ import annotations
import pickle
import yfinance
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Literal, Optional, Sequence


def load_pickle(path):
    """
    Load an object from a pickle file.

    Parameters
    ----------
    path : str
        Path to the pickle file.

    Returns
    -------
    object
        The Python object stored in the pickle file.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    pickle.UnpicklingError
        If the file is not a valid pickle.
    """
    try:
        with open(path, "rb") as fp:
            obj = pickle.load(fp)
        return obj
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Pickle file not found: {path}") from e
    except pickle.UnpicklingError as e:
        raise pickle.UnpicklingError(f"Error unpickling file: {path}") from e

def save_pickle(path, obj):
    """
    Save an object to a pickle file.

    Parameters
    ----------
    path : str
        Path to save the pickle file.
    obj : object
        The Python object to serialize.
    """
    with open(path, "wb") as fp:
        pickle.dump(obj, fp, protocol=pickle.HIGHEST_PROTOCOL)

def get_history(
    ticker: str,
    period_start: str | pd.Timestamp,
    period_end: str | pd.Timestamp,
    granularity: Literal['1m', '5m', '15m', '30m', '1h', '1d', '1wk', '1mo'] = '1d',
    tries: int = 0,
    max_tries: int = 5
) -> pd.DataFrame:
    
    """
    Downloads historical price data for a given ticker from Yahoo Finance.

    Args:
        ticker (str): The stock symbol.
        period_start (str | pd.Timestamp): Start date of the period.
        period_end (str | pd.Timestamp): End date of the period.
        granularity (str, optional): Data frequency. Defaults to '1d'.
        tries (int, optional): Current retry count (for recursion).
        max_tries (int, optional): Maximum retry attempts in case of failure.

    Returns:
        pd.DataFrame: DataFrame containing OHLCV data with datetime index.
                      Returns an empty DataFrame if download fails.
    """

    try:
        df = yfinance.download(
            tickers=ticker,
            start=period_start,
            end=period_end,
            interval=granularity,
            auto_adjust=True,
            progress=False
        )
    except Exception as err:
        # Retry recursively in case of transient network or API issues
        if tries < max_tries:
            return get_history(ticker, period_start, period_end, granularity, tries + 1, max_tries)
        print(f"[ERROR] Failed to download data for {ticker}: {err}")
        return pd.DataFrame()
    
    if df.empty:
        return pd.DataFrame()
    
    # Handle MultiIndex columns (some intervals in Yahoo Finance return MultiIndex)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # Clean up and standardize column names
    df = (
        df.reset_index()
          .rename(columns={
              'Date': 'datetime',
              'Open': 'open',
              'High': 'high',
              'Low': 'low',
              'Close': 'close',
              'Volume': 'volume'
          })
          .loc[:, ['datetime', 'open', 'high', 'low', 'close', 'volume']]
          .set_index('datetime')
    )
    return df

def get_histories(
    tickers: Sequence[str],
    period_start: str | pd.Timestamp,
    period_end: str | pd.Timestamp,
    granularity: Literal["1m", "5m", "15m", "30m", "1h", "1d", "1wk", "1mo"] = "1d"
) -> tuple[list[str], list[pd.DataFrame]]:

    """
    Retrieve historical OHLCV data for multiple tickers.

    Args:
        tickers (Sequence[str]): List or tuple of ticker symbols.
        period_start (str | pd.Timestamp): Start date of the historical period.
        period_end (str | pd.Timestamp): End date of the historical period.
        granularity (Literal): Data frequency (e.g., '1d', '1wk', '1mo'). Defaults to '1d'.

    Returns:
        tuple[list[str], list[pd.DataFrame]]:
            - valid_tickers: List of tickers that successfully returned non-empty data.
            - dfs: Corresponding list of DataFrames indexed by datetime.
    """

    # Collect data for each ticker
    results: list[tuple[str, pd.DataFrame]] = [
        (ticker, get_history(ticker, period_start, period_end, granularity))
        for ticker in tickers
    ]

    # Filter out tickers with no data
    valid_results = [(t, df) for t, df in results if not df.empty]

    if not valid_results:
        return [], []
    
    # Unpack valid results into separate lists
    valid_tickers, dfs = zip(*valid_results)

    return list(valid_tickers), list(dfs)

def get_ticker_dfs(
    tickers: Sequence[str],
    period_start: str | pd.Timestamp,
    period_end: str | pd.Timestamp,
    cache_path: str | Path = None # Path("utils/cache/dataset.obj") 
) -> tuple[list[str], dict[str, pd.DataFrame]]:
    
    """
    Retrieve historical DataFrames for multiple tickers, using local cache when available.

    The function first attempts to load cached data from a pickle file.
    If not found or corrupted, it downloads new data and saves it to the specified path.

    Args:
        tickers (Sequence[str]): List or tuple of ticker symbols.
        period_start (str | pd.Timestamp): Start date of the historical period.
        period_end (str | pd.Timestamp): End date of the historical period.
        cache_path (Optional[str | Path]): Path to the pickle cache file.
            Must be provided; if None, raises a ValueError.

    Returns:
        tuple[list[str], dict[str, pd.DataFrame]]:
            - tickers: List of tickers with valid historical data.
            - ticker_dfs: Dictionary mapping each ticker to its DataFrame.
    """

    if cache_path is None:
        raise ValueError("cache_path must be provided (e.g., 'data/cache/dataset.obj').")

    cache_path = Path(cache_path)

    try:
        # Attempt to load cached dataset
        tickers, ticker_dfs = load_pickle(cache_path)
    except (FileNotFoundError, EOFError, pickle.UnpicklingError, Exception):
        # If cache is missing or invalid, fetch fresh data
        tickers, dfs = get_histories(tickers, period_start, period_end)
        ticker_dfs = {t: df for t, df in zip(tickers, dfs)}
        # Save to cache for future reuse
        save_pickle(cache_path, (tickers, ticker_dfs))

    return list(tickers), ticker_dfs


if __name__ == '__main__':
    
    test_path = Path(__file__).parent / "cache" / "test_dataset.obj"
    tickers = ["AAPL", "MSFT", "SPY", "QQQ", "VALE3.SA", "PETR4.SA", "BOVA11.SA", "7203.T", "6758.T", "1321.T"]

    period_start = '2020-01-01'
    period_end = '2025-09-30'

    tickers, tickers_df = get_ticker_dfs(
        tickers=tickers,
        period_start=period_start,
        period_end=period_end,
        cache_path=test_path
    )

    for ticker in tickers:
        print(ticker)
        input(tickers_df[ticker])