"""
train_eval.py — Walk-forward rolling window evaluation for condition B (LSTM baseline).

For each evaluation day the model is trained on the most recent 250 trading days,
retrained every 20 days. Predicts next-day log return; direction is derived from sign.

Outputs: results/predictions_B.csv
Columns: date, actual_return, predicted_return, actual_direction, predicted_direction

Usage:
    python src/train_eval.py                 # full evaluation (~200+ days)
    python src/train_eval.py --eval-days 20  # first 20 days only (for testing)
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

# ── Config ────────────────────────────────────────────────────────────────────
PRICES_PATH   = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "prices.csv")
RESULTS_DIR   = os.path.join(os.path.dirname(__file__), "..", "results")
RESULTS_PATH  = os.path.join(RESULTS_DIR, "predictions_B.csv")

FEATURE_COLS  = ["open", "high", "low", "close", "volume",
                 "daily_log_return", "ma5", "ma20", "volatility20"]
TARGET_COL    = "target"

SEQ_LEN       = 20    # days of history fed into the LSTM per prediction
TRAIN_WINDOW  = 250   # rolling training window size
RETRAIN_EVERY = 20    # retrain every N evaluation days
HIDDEN_SIZE   = 64
NUM_LAYERS    = 2
DROPOUT       = 0.2
LR            = 1e-3
MAX_NORM      = 1.0   # gradient clipping
PATIENCE      = 10    # early stopping patience (validation loss)
MAX_EPOCHS    = 200
BATCH_SIZE    = 32
VAL_FRAC      = 0.2   # last 20% of training windows used for validation


def make_windows(features: np.ndarray, targets: np.ndarray, seq_len: int):
    """
    Build sliding-window (X, y) pairs from a scaled feature array.

    Window i uses features[i : i+seq_len] as input and targets[i+seq_len-1]
    as the label — the next-day return for the last day in the window.
    This matches the inference setup where the input ends at the current day
    and the model predicts that day's next-day return.
    """
    X, y = [], []
    for i in range(len(features) - seq_len + 1):
        X.append(features[i : i + seq_len])
        y.append(targets[i + seq_len - 1])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def train_one_window(X_train, y_train, X_val, y_val, device) -> LSTMForecaster:
    """Train an LSTMForecaster on one rolling window with early stopping."""
    n_features = X_train.shape[2]
    model = LSTMForecaster(
        input_size=n_features,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()

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
    parser.add_argument("--eval-days", type=int, default=None,
                        help="Evaluate only the first N evaluation days (for testing)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Load and clean data ───────────────────────────────────────────────────
    df = pd.read_csv(PRICES_PATH, parse_dates=["date"], index_col="date")
    df = df.dropna(subset=FEATURE_COLS + [TARGET_COL])

    features = df[FEATURE_COLS].values.astype(np.float32)  # (N, 9)
    targets  = df[TARGET_COL].values.astype(np.float32)    # (N,)
    dates    = df.index
    N        = len(df)

    n_eval = N - TRAIN_WINDOW
    print(f"Total rows (after dropna) : {N}")
    print(f"Warm-up window            : {TRAIN_WINDOW} days")
    print(f"Evaluation days available : {n_eval}")

    if N < TRAIN_WINDOW + 1:
        print("ERROR: not enough data for walk-forward evaluation.")
        sys.exit(1)

    # ── Walk-forward loop ─────────────────────────────────────────────────────
    eval_end = TRAIN_WINDOW + (args.eval_days if args.eval_days else n_eval)
    eval_end = min(eval_end, N)

    model  = None
    scaler = None
    rows   = []

    for step, eval_idx in enumerate(range(TRAIN_WINDOW, eval_end)):
        # Retrain on the rolling 250-day window every RETRAIN_EVERY steps
        if step % RETRAIN_EVERY == 0:
            train_start = eval_idx - TRAIN_WINDOW
            X_raw = features[train_start:eval_idx]   # (250, 9)
            y_raw = targets[train_start:eval_idx]    # (250,)

            # Scaler fitted on training window only — never touches future data
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_raw).astype(np.float32)

            X_win, y_win = make_windows(X_scaled, y_raw, SEQ_LEN)

            # Chronological 80/20 split (no shuffling for the split itself)
            split   = int(len(X_win) * (1 - VAL_FRAC))
            X_tr, X_vl = X_win[:split], X_win[split:]
            y_tr, y_vl = y_win[:split], y_win[split:]

            print(f"[step {step:4d} | day {dates[eval_idx].date()}] "
                  f"Retraining on rows {train_start}–{eval_idx-1}  "
                  f"({len(X_tr)} train / {len(X_vl)} val windows)", flush=True)

            model = train_one_window(X_tr, y_tr, X_vl, y_vl, device)
            model.eval()

        # Predict: feed the last SEQ_LEN rows ending at eval_idx (inclusive)
        input_raw    = features[eval_idx - SEQ_LEN + 1 : eval_idx + 1]  # (20, 9)
        input_scaled = scaler.transform(input_raw).astype(np.float32)
        x_t = torch.from_numpy(input_scaled).unsqueeze(0).to(device)    # (1, 20, 9)

        with torch.no_grad():
            pred_return = model(x_t).item()

        actual_return = float(targets[eval_idx])
        rows.append({
            "date":               dates[eval_idx].date(),
            "actual_return":      actual_return,
            "predicted_return":   pred_return,
            "actual_direction":   1 if actual_return > 0 else 0,
            "predicted_direction": 1 if pred_return  > 0 else 0,
        })

    # ── Save ──────────────────────────────────────────────────────────────────
    results_df = pd.DataFrame(rows)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    results_df.to_csv(RESULTS_PATH, index=False)
    print(f"\nSaved {len(results_df)} predictions → {RESULTS_PATH}")

    # ── Metrics ───────────────────────────────────────────────────────────────
    actual  = results_df["actual_return"].values
    pred    = results_df["predicted_return"].values
    act_dir = results_df["actual_direction"].values
    prd_dir = results_df["predicted_direction"].values

    mae     = np.mean(np.abs(actual - pred))
    rmse    = np.sqrt(np.mean((actual - pred) ** 2))
    dir_acc = np.mean(act_dir == prd_dir)

    print(f"\n── Condition B — LSTM Baseline ───────────────")
    print(f"Evaluation days : {len(results_df)}")
    print(f"MAE             : {mae:.6f}")
    print(f"RMSE            : {rmse:.6f}")
    print(f"Dir. Accuracy   : {dir_acc:.1%}  (random baseline: 50.0%)")


if __name__ == "__main__":
    main()
