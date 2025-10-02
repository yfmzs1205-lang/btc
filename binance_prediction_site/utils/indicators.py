"""
Technical indicator computation utilities.

These functions compute common indicators such as moving averages (SMA, EMA),
relative strength index (RSI) and moving average convergence divergence (MACD).
The indicators are added to an existing DataFrame.
"""
from __future__ import annotations

import pandas as pd
import numpy as np
from typing import List


def add_moving_averages(df: pd.DataFrame, windows: List[int]) -> pd.DataFrame:
    """Append simple moving averages for a list of window sizes.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing a 'close' column.
    windows : List[int]
        List of integer window sizes for computing the SMA.

    Returns
    -------
    pd.DataFrame
        DataFrame with additional SMA columns (e.g. ma5 for window=5).
    """
    df = df.copy()
    for w in windows:
        df[f"ma{w}"] = df["close"].rolling(window=w).mean()
    return df


def add_exponential_moving_averages(df: pd.DataFrame, spans: List[int]) -> pd.DataFrame:
    """Append exponential moving averages for a list of spans.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing a 'close' column.
    spans : List[int]
        List of spans for computing the EMA.

    Returns
    -------
    pd.DataFrame
        DataFrame with additional EMA columns (e.g. ema12 for span=12).
    """
    df = df.copy()
    for s in spans:
        df[f"ema{s}"] = df["close"].ewm(span=s, adjust=False).mean()
    return df


def add_rsi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """Append relative strength index (RSI) to the DataFrame.

    The RSI is computed using the standard Wilder's smoothing method.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing a 'close' column.
    period : int, optional
        Look-back period for RSI calculation.  The default is 14.

    Returns
    -------
    pd.DataFrame
        DataFrame with an additional 'rsi' column.
    """
    df = df.copy()
    delta = df["close"].diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    # First smoothed values
    gain_smooth = pd.Series(gain).rolling(window=period).mean()
    loss_smooth = pd.Series(loss).rolling(window=period).mean()
    # Avoid divide-by-zero warnings
    rs = gain_smooth / loss_smooth.replace(0, np.nan)
    rsi = 100 - 100 / (1 + rs)
    df["rsi"] = rsi
    return df


def add_macd(df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    """Append MACD and signal line to the DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with a 'close' column.
    fast : int, optional
        Fast EMA period. The default is 12.
    slow : int, optional
        Slow EMA period. The default is 26.
    signal : int, optional
        Signal line EMA period. The default is 9.

    Returns
    -------
    pd.DataFrame
        DataFrame with 'macd' and 'macd_signal' columns added.
    """
    df = df.copy()
    ema_fast = df["close"].ewm(span=fast, adjust=False).mean()
    ema_slow = df["close"].ewm(span=slow, adjust=False).mean()
    df["macd"] = ema_fast - ema_slow
    df["macd_signal"] = df["macd"].ewm(span=signal, adjust=False).mean()
    return df


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Compute and append a standard set of technical indicators.

    This function computes SMAs for 5, 10 and 20 periods, EMAs for 12 and 26,
    RSI for a 14-period look-back and MACD.  Additional indicators can
    easily be added here.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with at least a 'close' column.

    Returns
    -------
    pd.DataFrame
        DataFrame with new indicator columns.
    """
    df = add_moving_averages(df, [5, 10, 20])
    df = add_exponential_moving_averages(df, [12, 26])
    df = add_rsi(df, period=14)
    df = add_macd(df)
    # Drop rows with NaN due to indicator warm-up
    df.dropna(inplace=True)
    return df