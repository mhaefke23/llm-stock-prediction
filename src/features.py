"""
features.py
-----------
Builds the three feature sets used by the LSTM experiments:

  C1 — OHLCV + technical features only (no text)
  C2 — C1 features + PCA-reduced sentence embeddings of LLM summaries
  C3 — C2 features + LLM direction (binary) + LLM confidence score

Inputs:
  data/processed/prices.csv        (from fetch_data.py)
  data/processed/llm_features.csv  (from llm_features.py)

Outputs:
  data/processed/features_c1.csv
  data/processed/features_c2.csv
  data/processed/features_c3.csv

All scalers and the PCA are fitted on the training split only (first 70%
of rows) and then applied to val and test — no data leakage.

A leakage assertion is run before saving each file.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# ── Config ────────────────────────────────────────────────────────────────────
PRICES_CSV   = Path("data/processed/prices.csv")
LLM_CSV      = Path("data/processed/llm_features.csv")
OUT_DIR      = Path("data/processed")

EMBED_MODEL  = "all-MiniLM-L6-v2"   # 384-dim sentence embeddings
N_PCA_DIMS   = 8                     # reduce embeddings to 8 dims via PCA
TRAIN_FRAC   = 0.70
VAL_FRAC     = 0.15
# test fraction = 1 - TRAIN_FRAC - VAL_FRAC = 0.15

# Columns from prices.csv to use as features (target excluded)
PRICE_FEATURE_COLS = [
    "open", "high", "low", "close", "volume",
    "daily_return", "ma5", "ma20", "volatility20",
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def split_indices(n: int):
    """Return (train_end, val_end) row indices for a strict time-based split."""
    train_end = int(n * TRAIN_FRAC)
    val_end   = train_end + int(n * VAL_FRAC)
    return train_end, val_end


def assert_no_leakage(df: pd.DataFrame, train_end: int, scaler: StandardScaler):
    """
    Verify that the scaler's mean matches the training-set mean for all
    feature columns. Raises AssertionError if test-set statistics leaked in.
    """
    feature_cols = df.columns.drop("target")
    train_mean = df[feature_cols].iloc[:train_end].mean().values
    np.testing.assert_allclose(
        scaler.mean_,
        train_mean,
        rtol=1e-4,
        atol=1e-6,  # tolerate floating-point differences for near-zero PCA components
        err_msg="Data leakage detected: scaler was not fitted on training set only.",
    )


def scale_features(df: pd.DataFrame, train_end: int):
    """
    Fit StandardScaler on training rows, transform all rows.
    Returns (scaled_df, fitted_scaler).
    """
    feature_cols = [c for c in df.columns if c != "target"]
    scaler = StandardScaler()
    scaler.fit(df[feature_cols].iloc[:train_end])

    scaled = df.copy()
    scaled[feature_cols] = scaler.transform(df[feature_cols])
    return scaled, scaler


# ── C1: price + technical features ───────────────────────────────────────────

def build_c1(prices: pd.DataFrame) -> pd.DataFrame:
    """Select OHLCV + technical columns; keep target."""
    cols = PRICE_FEATURE_COLS + ["target"]
    return prices[cols].copy()


# ── C2: C1 + sentence embeddings (PCA-reduced) ───────────────────────────────

def build_c2(prices: pd.DataFrame, llm: pd.DataFrame, train_end: int) -> pd.DataFrame:
    """
    Embed LLM summaries with all-MiniLM-L6-v2, reduce to N_PCA_DIMS via PCA
    fitted on training data only, then concatenate with C1 features.
    """
    print(f"  Loading sentence-transformer ({EMBED_MODEL}) …")
    embedder = SentenceTransformer(EMBED_MODEL)

    summaries = llm["summary"].tolist()
    print(f"  Embedding {len(summaries)} summaries …")
    embeddings = embedder.encode(summaries, show_progress_bar=True, batch_size=32)
    # embeddings shape: (n_days, 384)

    # Fit PCA on training rows only
    pca = PCA(n_components=N_PCA_DIMS, random_state=42)
    pca.fit(embeddings[:train_end])
    reduced = pca.transform(embeddings)  # shape: (n_days, N_PCA_DIMS)

    embed_cols = [f"emb_{i}" for i in range(N_PCA_DIMS)]
    embed_df   = pd.DataFrame(reduced, index=llm.index, columns=embed_cols)

    c1 = build_c1(prices)
    merged = c1.join(embed_df, how="inner")
    return merged


# ── C3: C2 + LLM direction + LLM confidence ──────────────────────────────────

def build_c3(c2: pd.DataFrame, llm: pd.DataFrame) -> pd.DataFrame:
    """
    Add two extra features to C2:
      llm_direction  : 1 if LLM predicted "up", 0 if "down"
      llm_confidence : float 0.0–1.0
    """
    extra = llm[["direction", "confidence"]].copy()
    extra["llm_direction"]  = (extra["direction"] == "up").astype(float)
    extra["llm_confidence"] = extra["confidence"].astype(float)
    extra = extra[["llm_direction", "llm_confidence"]]

    return c2.join(extra, how="inner")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    # Load inputs
    prices = pd.read_csv(PRICES_CSV, index_col="date", parse_dates=True)
    llm    = pd.read_csv(LLM_CSV,    index_col="date", parse_dates=True)

    # Inner-join on date so only days with both price data and LLM features
    # are kept. This automatically drops the first 5 rows (skipped in llm_features.py).
    llm    = llm.loc[llm.index.isin(prices.index)]
    prices = prices.loc[prices.index.isin(llm.index)]

    n = len(prices)
    train_end, val_end = split_indices(n)
    print(f"Dataset: {n} days  |  train={train_end}  val={val_end - train_end}  test={n - val_end}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── C1 ──────────────────────────────────────────────────────────────────
    print("\nBuilding C1 (price + technical) …")
    c1_raw = build_c1(prices)
    c1, scaler_c1 = scale_features(c1_raw, train_end)
    assert_no_leakage(c1_raw, train_end, scaler_c1)
    c1.to_csv(OUT_DIR / "features_c1.csv")
    print(f"  Saved features_c1.csv  shape={c1.shape}")

    # ── C2 ──────────────────────────────────────────────────────────────────
    print("\nBuilding C2 (C1 + sentence embeddings via PCA) …")
    c2_raw = build_c2(prices, llm, train_end)
    c2, scaler_c2 = scale_features(c2_raw, train_end)
    assert_no_leakage(c2_raw, train_end, scaler_c2)
    c2.to_csv(OUT_DIR / "features_c2.csv")
    print(f"  Saved features_c2.csv  shape={c2.shape}")

    # ── C3 ──────────────────────────────────────────────────────────────────
    print("\nBuilding C3 (C2 + LLM direction + confidence) …")
    c3_raw = build_c3(c2_raw, llm)
    c3, scaler_c3 = scale_features(c3_raw, train_end)
    assert_no_leakage(c3_raw, train_end, scaler_c3)
    c3.to_csv(OUT_DIR / "features_c3.csv")
    print(f"  Saved features_c3.csv  shape={c3.shape}")

    # ── Summary ─────────────────────────────────────────────────────────────
    print("\nAll feature files written successfully.")
    print(f"  C1 input dims : {c1.shape[1] - 1}")   # exclude target
    print(f"  C2 input dims : {c2.shape[1] - 1}")
    print(f"  C3 input dims : {c3.shape[1] - 1}")
    print(f"\nLeakage assertion passed for all three conditions.")


if __name__ == "__main__":
    main()
