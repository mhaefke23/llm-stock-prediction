"""
train_eval_xgb.py — Walk-forward evaluation for XGBoost conditions.

X1: Regressor, single-day features (9 features). Direction derived from sign of predicted return.
X2: Regressor, flattened 20-day lags (180 features). Direction derived from sign.
X3: Classifier, single-day features. Predicts direction (0/1) directly.
X4: Classifier, flattened 20-day lags. Predicts direction (0/1) directly.

All conditions retrain every 20 evaluation days on a rolling 250-day window.

Outputs: results/predictions_{X1,X2,X3,X4}.csv
Columns: date, actual_return, predicted_return*, actual_direction, predicted_direction
  * predicted_return is present for X1/X2 (log return), absent for X3/X4 (classifiers).

Usage:
    python src/train_eval_xgb.py --condition X1
    python src/train_eval_xgb.py --condition X3 --eval-days 20  # quick test
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor, XGBClassifier

PRICES_PATH  = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "prices.csv")
RESULTS_DIR  = os.path.join(os.path.dirname(__file__), "..", "results")

FEATURE_COLS  = ["open", "high", "low", "close", "volume",
                 "daily_log_return", "ma5", "ma20", "volatility20"]
TARGET_COL    = "target"

TRAIN_WINDOW  = 250
RETRAIN_EVERY = 20
SEQ_LEN       = 20
VAL_FRAC      = 0.2

XGB_REG_PARAMS = dict(
    n_estimators=500,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    early_stopping_rounds=20,
    eval_metric="rmse",
    random_state=42,
)

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

CONDITION_META = {
    "X1": {"model": "regressor", "features": "single-day"},
    "X2": {"model": "regressor", "features": "20-day lags"},
    "X3": {"model": "classifier", "features": "single-day"},
    "X4": {"model": "classifier", "features": "20-day lags"},
}


def make_flat_windows(features: np.ndarray, targets: np.ndarray, seq_len: int):
    """
    Build (X, y) pairs where each X row is seq_len days of features flattened.
    Label y[i] is the target for day i+seq_len-1 (same alignment as LSTM).
    """
    X, y = [], []
    for i in range(len(features) - seq_len + 1):
        X.append(features[i : i + seq_len].flatten())
        y.append(targets[i + seq_len - 1])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", choices=list(CONDITION_META), required=True)
    parser.add_argument("--eval-days", type=int, default=None)
    args = parser.parse_args()

    meta         = CONDITION_META[args.condition]
    is_clf       = meta["model"] == "classifier"
    use_lags     = meta["features"] == "20-day lags"
    results_path = os.path.join(RESULTS_DIR, f"predictions_{args.condition}.csv")

    df = pd.read_csv(PRICES_PATH, parse_dates=["date"], index_col="date")
    df = df.dropna(subset=FEATURE_COLS + [TARGET_COL])

    features  = df[FEATURE_COLS].values.astype(np.float32)
    targets   = df[TARGET_COL].values.astype(np.float32)
    directions = (targets > 0).astype(np.int32)
    dates     = df.index
    N         = len(df)

    n_eval = N - TRAIN_WINDOW
    print(f"Condition       : {args.condition}  ({meta['model']}, {meta['features']})")
    print(f"Total rows      : {N}")
    print(f"Warm-up window  : {TRAIN_WINDOW} days")
    print(f"Eval days       : {n_eval}")

    if N < TRAIN_WINDOW + 1:
        print("ERROR: not enough data for walk-forward evaluation.")
        sys.exit(1)

    eval_end = TRAIN_WINDOW + (args.eval_days if args.eval_days else n_eval)
    eval_end = min(eval_end, N)

    model  = None
    scaler = None
    rows   = []

    for step, eval_idx in enumerate(range(TRAIN_WINDOW, eval_end)):
        if step % RETRAIN_EVERY == 0:
            train_start = eval_idx - TRAIN_WINDOW
            X_raw = features[train_start:eval_idx]
            y_raw = targets[train_start:eval_idx]
            d_raw = directions[train_start:eval_idx]

            scaler   = StandardScaler()
            X_scaled = scaler.fit_transform(X_raw).astype(np.float32)

            if use_lags:
                X_win, y_win = make_flat_windows(X_scaled, y_raw, SEQ_LEN)
                _, d_win     = make_flat_windows(X_scaled, d_raw.astype(np.float32), SEQ_LEN)
                d_win        = d_win.astype(np.int32)
            else:
                # Drop the last row to avoid leaking the label we're about to predict.
                X_win = X_scaled[:-1]
                y_win = y_raw[:-1]
                d_win = d_raw[:-1]

            split = int(len(X_win) * (1 - VAL_FRAC))
            X_tr, X_vl = X_win[:split], X_win[split:]

            print(f"[step {step:4d} | day {dates[eval_idx].date()}] "
                  f"Retraining on rows {train_start}–{eval_idx-1}  "
                  f"({len(X_tr)} train / {len(X_vl)} val samples)", flush=True)

            if is_clf:
                d_tr, d_vl = d_win[:split], d_win[split:]
                model = XGBClassifier(**XGB_CLF_PARAMS)
                model.fit(X_tr, d_tr, eval_set=[(X_vl, d_vl)], verbose=False)
            else:
                y_tr, y_vl = y_win[:split], y_win[split:]
                model = XGBRegressor(**XGB_REG_PARAMS)
                model.fit(X_tr, y_tr, eval_set=[(X_vl, y_vl)], verbose=False)

        # ── Predict ───────────────────────────────────────────────────────────
        if use_lags:
            raw_window = features[eval_idx - SEQ_LEN + 1 : eval_idx + 1]
            x_input    = scaler.transform(raw_window).flatten().reshape(1, -1)
        else:
            x_input = scaler.transform(features[eval_idx : eval_idx + 1])

        actual_return = float(targets[eval_idx])
        actual_dir    = int(directions[eval_idx])

        if is_clf:
            pred_dir = int(model.predict(x_input)[0])
            rows.append({
                "date":                dates[eval_idx].date(),
                "actual_return":       actual_return,
                "actual_direction":    actual_dir,
                "predicted_direction": pred_dir,
            })
        else:
            pred_return = float(model.predict(x_input)[0])
            rows.append({
                "date":                dates[eval_idx].date(),
                "actual_return":       actual_return,
                "predicted_return":    pred_return,
                "actual_direction":    actual_dir,
                "predicted_direction": 1 if pred_return > 0 else 0,
            })

    results_df = pd.DataFrame(rows)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    results_df.to_csv(results_path, index=False)
    print(f"\nSaved {len(results_df)} predictions → {results_path}")

    act_dir = results_df["actual_direction"].values
    prd_dir = results_df["predicted_direction"].values
    dir_acc = np.mean(act_dir == prd_dir)

    label = f"{meta['model'].capitalize()}, {meta['features']}"
    print(f"\n── Condition {args.condition} — XGBoost ({label}) ──")
    print(f"Evaluation days : {len(results_df)}")

    if not is_clf:
        actual = results_df["actual_return"].values
        pred   = results_df["predicted_return"].values
        print(f"MAE             : {np.mean(np.abs(actual - pred)):.6f}")
        print(f"RMSE            : {np.sqrt(np.mean((actual - pred) ** 2)):.6f}")

    print(f"Dir. Accuracy   : {dir_acc:.1%}  (random baseline: 50.0%)")


if __name__ == "__main__":
    main()
