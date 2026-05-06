"""
train_eval_xgb_news.py — XGB+N: XGBoost classifier with OHLCV + PCA news embeddings.

Extends condition X3 (XGBoost classifier, single-day features) by appending
PCA-reduced news report embeddings to the feature vector.

At each retraining step:
  1. Scale price features (StandardScaler fitted on training window only).
  2. Fit PCA on training-window embeddings (no leakage into eval period).
  3. Transform both training and inference embeddings.
  4. Concatenate scaled price features + PCA components → train XGBClassifier.

Days with missing embeddings fall back to a zero vector.

Output: results/predictions_XN.csv
Columns: date, actual_return, actual_direction, predicted_direction

Usage:
    python src/train_eval_xgb_news.py
    python src/train_eval_xgb_news.py --eval-days 20     # quick test
    python src/train_eval_xgb_news.py --pca-components 8 # default
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from xgboost import XGBClassifier

PRICES_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "prices.csv")
EMBED_PATH  = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "report_embeddings.parquet")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
OUT_PATH    = os.path.join(RESULTS_DIR, "predictions_XN.csv")

FEATURE_COLS  = ["open", "high", "low", "close", "volume",
                 "daily_log_return", "ma5", "ma20", "volatility20"]
TARGET_COL    = "target"

TRAIN_WINDOW  = 250
RETRAIN_EVERY = 20
VAL_FRAC      = 0.2
DEFAULT_PCA_K = 8   # all-MiniLM-L6-v2 produces 384-dim vectors; PCA reduces to k

XGB_CLF_PARAMS = dict(
    n_estimators=500,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    early_stopping_rounds=20,
    eval_metric="logloss",
    random_state=42,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-days",      type=int, default=None)
    parser.add_argument("--pca-components", type=int, default=DEFAULT_PCA_K,
                        help=f"PCA components for embeddings (default {DEFAULT_PCA_K})")
    args = parser.parse_args()

    if not os.path.exists(EMBED_PATH):
        print(f"ERROR: {EMBED_PATH} not found. Run embed_reports.py first.")
        sys.exit(1)

    # ── Load data ─────────────────────────────────────────────────────────────
    prices_df = pd.read_csv(PRICES_PATH, parse_dates=["date"], index_col="date")
    prices_df = prices_df.dropna(subset=FEATURE_COLS + [TARGET_COL])

    emb_df = pd.read_parquet(EMBED_PATH)
    emb_df.index = pd.to_datetime(emb_df.index).normalize()
    EMBED_DIM = emb_df.shape[1]

    features   = prices_df[FEATURE_COLS].values.astype(np.float32)
    targets    = prices_df[TARGET_COL].values.astype(np.float32)
    directions = (targets > 0).astype(np.int32)
    dates      = prices_df.index.normalize()
    N          = len(prices_df)

    # Build embedding matrix aligned to prices (zero-fill missing dates)
    embeddings = np.zeros((N, EMBED_DIM), dtype=np.float32)
    for i, date in enumerate(dates):
        if date in emb_df.index:
            embeddings[i] = emb_df.loc[date].values

    n_missing = np.sum(embeddings.sum(axis=1) == 0)
    n_eval    = N - TRAIN_WINDOW

    print(f"Condition       : XN  (XGBoost classifier + PCA news, k={args.pca_components})")
    print(f"Total rows      : {N}  ({n_missing} with missing embeddings → zero-filled)")
    print(f"Warm-up window  : {TRAIN_WINDOW} days")
    print(f"Eval days       : {n_eval}")

    if N < TRAIN_WINDOW + 1:
        print("ERROR: not enough data.")
        sys.exit(1)

    eval_end = TRAIN_WINDOW + (args.eval_days if args.eval_days else n_eval)
    eval_end = min(eval_end, N)

    model  = None
    scaler = None
    pca    = None
    rows   = []

    for step, eval_idx in enumerate(range(TRAIN_WINDOW, eval_end)):
        if step % RETRAIN_EVERY == 0:
            train_start = eval_idx - TRAIN_WINDOW
            X_raw  = features[train_start:eval_idx]
            d_raw  = directions[train_start:eval_idx]
            E_raw  = embeddings[train_start:eval_idx]   # (250, EMBED_DIM)

            # Scale price features
            scaler   = StandardScaler()
            X_scaled = scaler.fit_transform(X_raw).astype(np.float32)

            # Fit PCA on training-window embeddings (leakage-free)
            k   = min(args.pca_components, E_raw.shape[0] - 1)
            pca = PCA(n_components=k, random_state=42)
            E_pca_train = pca.fit_transform(E_raw).astype(np.float32)

            # Drop last row to avoid leaking the label we're about to predict
            X_win = np.hstack([X_scaled[:-1], E_pca_train[:-1]])
            y_win = d_raw[:-1]

            split = int(len(X_win) * (1 - VAL_FRAC))
            X_tr, X_vl = X_win[:split],  X_win[split:]
            y_tr, y_vl = y_win[:split],  y_win[split:]

            print(f"[step {step:4d} | day {dates[eval_idx].date()}] "
                  f"Retraining on rows {train_start}–{eval_idx-1}  "
                  f"({len(X_tr)} train / {len(X_vl)} val samples, "
                  f"features={X_tr.shape[1]})", flush=True)

            model = XGBClassifier(**XGB_CLF_PARAMS)
            model.fit(X_tr, y_tr, eval_set=[(X_vl, y_vl)], verbose=False)

        # ── Predict ───────────────────────────────────────────────────────────
        x_price = scaler.transform(features[eval_idx : eval_idx + 1])
        e_pca   = pca.transform(embeddings[eval_idx : eval_idx + 1]).astype(np.float32)
        x_input = np.hstack([x_price, e_pca])

        pred_dir      = int(model.predict(x_input)[0])
        actual_return = float(targets[eval_idx])
        actual_dir    = int(directions[eval_idx])

        rows.append({
            "date":                dates[eval_idx].date(),
            "actual_return":       actual_return,
            "actual_direction":    actual_dir,
            "predicted_direction": pred_dir,
        })

    results_df = pd.DataFrame(rows)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    results_df.to_csv(OUT_PATH, index=False)
    print(f"\nSaved {len(results_df)} predictions → {OUT_PATH}")

    act_dir = results_df["actual_direction"].values
    prd_dir = results_df["predicted_direction"].values
    dir_acc = np.mean(act_dir == prd_dir)

    print(f"\n── Condition XN — XGBoost + News (PCA k={args.pca_components}) ──")
    print(f"Evaluation days : {len(results_df)}")
    print(f"Dir. Accuracy   : {dir_acc:.1%}  (random baseline: 50.0%)")


if __name__ == "__main__":
    main()
