"""
fetch_data.py
-------------
Downloads 1 year of daily OHLCV data for AAPL via yfinance,
computes technical features and the next-day log-return target,
and saves the result to data/processed/prices.csv.

Features produced:
  - open, high, low, close, volume  (raw OHLCV)
  - daily_return   : log(close_t / close_t-1)
  - ma5            : 5-day simple moving average of close
  - ma20           : 20-day simple moving average of close
  - volatility20   : 20-day rolling std of daily_return
  - target         : next-day log return = log(close_t+1 / close_t)

Rows with NaN (first 20 days due to rolling windows, last row
because target requires t+1) are dropped before saving.
"""

import numpy as np
import pandas as pd
import yfinance as yf
from pathlib import Path

TICKER = "AAPL"
PERIOD = "1y"          # roughly 252 trading days
OUT_PATH = Path("data/processed/prices.csv")


def fetch_ohlcv(ticker: str, period: str) -> pd.DataFrame:
    df = yf.download(ticker, period=period, auto_adjust=True, progress=False)
    # yfinance returns a MultiIndex when multiple tickers are used; flatten it
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns = [c.lower() for c in df.columns]
    df.index.name = "date"
    return df[["open", "high", "low", "close", "volume"]]


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["daily_return"] = np.log(df["close"] / df["close"].shift(1))
    df["ma5"]          = df["close"].rolling(5).mean()
    df["ma20"]         = df["close"].rolling(20).mean()
    df["volatility20"] = df["daily_return"].rolling(20).std()
    # Target: the return that will be realised at tomorrow's close
    df["target"]       = df["daily_return"].shift(-1)
    return df


def main():
    print(f"Downloading {PERIOD} of {TICKER} OHLCV data …")
    df = fetch_ohlcv(TICKER, PERIOD)
    df = add_features(df)

    rows_before = len(df)
    df.dropna(inplace=True)
    print(f"Rows after dropping NaN: {len(df)} (dropped {rows_before - len(df)})")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH)
    print(f"Saved → {OUT_PATH}")
    print(df.tail(3).to_string())


if __name__ == "__main__":
    main()
