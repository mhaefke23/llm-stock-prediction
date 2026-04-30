"""
fetch_data.py — Download AAPL OHLCV data from yfinance and compute features.

Outputs: data/processed/prices.csv
"""

import os
import sys
import yfinance as yf
import pandas as pd
import numpy as np

# ── Configuration ────────────────────────────────────────────────────────────
TICKER      = "AAPL"
START_DATE  = "2024-07-01"
END_DATE    = "2026-04-30"
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "prices.csv")


def fetch_and_engineer() -> pd.DataFrame:
    """Download OHLCV data and compute all derived features."""
    print(f"Downloading {TICKER} from {START_DATE} to {END_DATE} ...")
    raw = yf.download(TICKER, start=START_DATE, end=END_DATE, auto_adjust=True, progress=False)

    if raw.empty:
        print("ERROR: yfinance returned no data. Check ticker and date range.")
        sys.exit(1)

    # Flatten multi-level columns that yfinance sometimes produces
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = [col[0] for col in raw.columns]

    df = pd.DataFrame(index=raw.index)
    df.index.name = "date"

    # Core OHLCV
    df["open"]   = raw["Open"]
    df["high"]   = raw["High"]
    df["low"]    = raw["Low"]
    df["close"]  = raw["Close"]
    df["volume"] = raw["Volume"]

    # Log return for day t: log(close_t / close_{t-1})
    df["daily_log_return"] = np.log(df["close"] / df["close"].shift(1))

    # Prediction target: next-day log return  ← no leakage (shift(-1) looks forward)
    df["target"] = df["daily_log_return"].shift(-1)

    # Rolling technical indicators (computed on close and log return)
    df["ma5"]          = df["close"].rolling(5).mean()
    df["ma20"]         = df["close"].rolling(20).mean()
    df["volatility20"] = df["daily_log_return"].rolling(20).std()

    # Drop the first row (NaN daily_log_return) and the last row (NaN target)
    df = df.dropna(subset=["daily_log_return", "target"])

    return df


def main():
    df = fetch_and_engineer()

    # ── Save ────────────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    df.to_csv(OUTPUT_PATH)
    print(f"\nSaved {len(df)} rows → {OUTPUT_PATH}")

    # ── Summary ─────────────────────────────────────────────────────────────
    print(f"\nDate range : {df.index[0].date()} → {df.index[-1].date()}")
    print(f"Row count  : {len(df)}")
    print(f"\nFirst 3 rows:\n{df.head(3).to_string()}")
    print(f"\nLast 3 rows:\n{df.tail(3).to_string()}")
    print(f"\nMissing values:\n{df.isnull().sum().to_string()}")


if __name__ == "__main__":
    main()
