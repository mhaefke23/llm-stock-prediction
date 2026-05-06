"""
train_eval_lstm_clf.py — Walk-forward LSTM classifier conditions.

B1: LSTM classifier, seq_len=1 (single timestep — analogous to XGBoost X3).
B2: LSTM classifier, seq_len=20 (full sequence — analogous to XGBoost X4, same
    input window as condition B but trained with BCEWithLogitsLoss on direction).

Architecture is identical to condition B (LSTMForecaster scalar output used as
a logit). Direction is predicted directly; no regression step.

Outputs: results/predictions_{B1,B2}.csv
Columns: date, actual_return, actual_direction, predicted_direction

Usage:
    python src/train_eval_lstm_clf.py --condition B1
    python src/train_eval_lstm_clf.py --condition B2
    python src/train_eval_lstm_clf.py --condition B1 --eval-days 20  # quick test
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(__file__))
from model import LSTMForecaster

PRICES_PATH   = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "prices.csv")
RESULTS_DIR   = os.path.join(os.path.dirname(__file__), "..", "results")

FEATURE_COLS  = ["open", "high", "low", "close", "volume",
                 "daily_log_return", "ma5", "ma20", "volatility20"]
TARGET_COL    = "target"

TRAIN_WINDOW  = 250
RETRAIN_EVERY = 20
VAL_FRAC      = 0.2
HIDDEN_SIZE   = 64
NUM_LAYERS    = 2
DROPOUT       = 0.2
LR            = 1e-3
MAX_NORM      = 1.0
PATIENCE      = 10
MAX_EPOCHS    = 200
BATCH_SIZE    = 32

SEQ_LENS = {"B1": 1, "B2": 20}


def make_windows(features: np.ndarray, directions: np.ndarray, seq_len: int):
    """Sliding-window (X, y) pairs; y is binary direction of the last day in the window."""
    X, y = [], []
    for i in range(len(features) - seq_len + 1):
        X.append(features[i : i + seq_len])
        y.append(directions[i + seq_len - 1])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def train_one_window(X_train, y_train, X_val, y_val, device) -> LSTMForecaster:
    """Train LSTM classifier with BCEWithLogitsLoss and early stopping."""
    n_features = X_train.shape[2]
    model = LSTMForecaster(
        input_size=n_features,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.BCEWithLogitsLoss()

    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

    val_x = torch.from_numpy(X_val).to(device)
    val_y = torch.from_numpy(y_val).to(device)

    best_val_loss = float("inf")
    best_state    = None
    patience_cnt  = 0

    for epoch in range(MAX_EPOCHS):
        model.train()
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), MAX_NORM)
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(val_x), val_y).item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state    = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_cnt  = 0
        else:
            patience_cnt += 1
            if patience_cnt >= PATIENCE:
                break

    model.load_state_dict(best_state)
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", choices=["B1", "B2"], required=True,
                        help="B1=seq_len 1 (single-day), B2=seq_len 20 (full sequence)")
    parser.add_argument("--eval-days", type=int, default=None)
    args = parser.parse_args()

    seq_len      = SEQ_LENS[args.condition]
    results_path = os.path.join(RESULTS_DIR, f"predictions_{args.condition}.csv")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Condition       : {args.condition}  (LSTM classifier, seq_len={seq_len})")
    print(f"Device          : {device}")

    df = pd.read_csv(PRICES_PATH, parse_dates=["date"], index_col="date")
    df = df.dropna(subset=FEATURE_COLS + [TARGET_COL])

    features   = df[FEATURE_COLS].values.astype(np.float32)
    targets    = df[TARGET_COL].values.astype(np.float32)
    directions = (targets > 0).astype(np.float32)
    dates      = df.index
    N          = len(df)

    n_eval = N - TRAIN_WINDOW
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
            d_raw = directions[train_start:eval_idx]

            scaler   = StandardScaler()
            X_scaled = scaler.fit_transform(X_raw).astype(np.float32)

            X_win, y_win = make_windows(X_scaled, d_raw, seq_len)

            split = int(len(X_win) * (1 - VAL_FRAC))
            X_tr, X_vl = X_win[:split], X_win[split:]
            y_tr, y_vl = y_win[:split], y_win[split:]

            print(f"[step {step:4d} | day {dates[eval_idx].date()}] "
                  f"Retraining on rows {train_start}–{eval_idx-1}  "
                  f"({len(X_tr)} train / {len(X_vl)} val windows)", flush=True)

            model = train_one_window(X_tr, y_tr, X_vl, y_vl, device)
            model.eval()

        # Predict: feed seq_len rows ending at eval_idx (inclusive)
        input_raw    = features[eval_idx - seq_len + 1 : eval_idx + 1]
        input_scaled = scaler.transform(input_raw).astype(np.float32)
        x_t          = torch.from_numpy(input_scaled).unsqueeze(0).to(device)

        with torch.no_grad():
            logit    = model(x_t).item()
            pred_dir = 1 if logit > 0 else 0   # sigmoid > 0.5 ↔ logit > 0

        actual_return = float(targets[eval_idx])
        actual_dir    = 1 if actual_return > 0 else 0

        rows.append({
            "date":                dates[eval_idx].date(),
            "actual_return":       actual_return,
            "actual_direction":    actual_dir,
            "predicted_direction": pred_dir,
        })

    results_df = pd.DataFrame(rows)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    results_df.to_csv(results_path, index=False)
    print(f"\nSaved {len(results_df)} predictions → {results_path}")

    act_dir = results_df["actual_direction"].values
    prd_dir = results_df["predicted_direction"].values
    dir_acc = np.mean(act_dir == prd_dir)

    seq_label = "single-day" if seq_len == 1 else f"{seq_len}-day sequence"
    print(f"\n── Condition {args.condition} — LSTM Classifier ({seq_label}) ──")
    print(f"Evaluation days : {len(results_df)}")
    print(f"Dir. Accuracy   : {dir_acc:.1%}  (random baseline: 50.0%)")


if __name__ == "__main__":
    main()
