"""
Utility functions for fetching and preparing cryptocurrency market data from
Binance.  These helpers rely on the public Binance REST API for kline (candlestick)
data, so no API key is required.  The returned data is in pandas DataFrame
format with properly typed columns.

If you run this code outside of our learning environment, please ensure that
your network configuration allows outbound HTTPS connections to api.binance.com.
"""

from __future__ import annotations

import pandas as pd
import requests
from typing import Tuple


def fetch_klines(
    symbol: str,
    interval: str = "1m",
    limit: int = 500,
    start_time: int | None = None,
    end_time: int | None = None,
) -> pd.DataFrame:
    """Fetch historical candlestick data from Binance.

    Parameters
    ----------
    symbol : str
        Trading symbol, e.g. "BTCUSDT" or "ETHUSDT".
    interval : str, optional
        Kline interval supported by Binance.  Common values include
        "1m", "3m", "5m", "15m", "30m", "1h", etc.  By default "1m".
    limit : int, optional
        Maximum number of klines to return.  Binance caps this at 1000.
        The default is 500.
    start_time : int, optional
        Epoch millisecond timestamp to start from.  If provided,
        Binance will return data beginning at this timestamp.
    end_time : int, optional
        Epoch millisecond timestamp to end at.  If provided, Binance
        will return data up to this timestamp.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing columns timestamp, open, high, low, close,
        and volume.  The timestamp column is converted to pandas
        datetime64[ns] dtype.

    Notes
    -----
    This function makes an unauthenticated GET request to the Binance
    `/api/v3/klines` endpoint.  If you experience connection issues,
    consider reducing the `limit` parameter or increasing the interval.
    """
    url = "https://api.binance.com/api/v3/klines"
    params = {
        "symbol": symbol.upper(),
        "interval": interval,
        "limit": limit,
    }
    if start_time is not None:
        params["startTime"] = int(start_time)
    if end_time is not None:
        params["endTime"] = int(end_time)
    response = requests.get(url, params=params, timeout=10)
    response.raise_for_status()
    data = response.json()
    # Binance returns a list of lists where each inner list has the form:
    # [open_time, open, high, low, close, volume, close_time, quote_asset_volume,
    #  number_of_trades, taker_buy_base, taker_buy_quote, ignore]
    columns = [
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "close_time",
        "quote_asset_volume",
        "number_of_trades",
        "taker_buy_base",
        "taker_buy_quote",
        "ignore",
    ]
    df = pd.DataFrame(data, columns=columns)
    # Convert numeric columns
    numeric_cols = ["open", "high", "low", "close", "volume"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    # Keep only essential columns
    df = df[["timestamp", "open", "high", "low", "close", "volume"]]
    return df


def prepare_target(df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Add a binary target label indicating whether price declines after a horizon.

    The target column is 1 if the close price after `horizon` periods is
    strictly less than the current close price (i.e. price has fallen),
    otherwise 0.  Rows near the end of the DataFrame where the future
    horizon extends beyond available data are dropped.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing at least a 'close' column.
    horizon : int
        Number of periods ahead to look for the target.  For example,
        if each row represents one minute, horizon=10 uses 10 minutes.

    Returns
    -------
    pd.DataFrame
        DataFrame with an additional 'target' column.
    """
    df = df.copy()
    df["future_close"] = df["close"].shift(-horizon)
    df["target"] = (df["future_close"] < df["close"]).astype(int)
    df.dropna(inplace=True)
    df.drop(columns=["future_close"], inplace=True)
    return df