"""
train.py
--------
Trains one LSTMForecaster per condition (C1, C2, C3) and evaluates each
on the held-out test set. Saves predictions and metrics to results/.

Adapted from github.com/AmirhosseinHonardoust/Stock-LSTM-Forecasting.
Key changes vs source:
  - Multivariate input (all feature columns, not just close price)
  - Target is next-day log return (already in features_c*.csv as "target")
  - Features are pre-scaled by features.py — no re-scaling here
  - 70 / 15 / 15 time-based splits (source used 80/20, no test set)
  - SEQ_LEN=60 lookback (source default)
  - shuffle=True for train DataLoader (valid for independent windows)
  - Early stopping patience=10
  - Additional metrics: R², directional accuracy

Split (strictly by time, no shuffling of rows):
  Train : first 70%
  Val   : next  15%  (early stopping)
  Test  : last  15%

Outputs (written to results/):
  predictions_C1.csv, predictions_C2.csv, predictions_C3.csv
  loss_C1.csv, loss_C2.csv, loss_C3.csv
  metrics.csv
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from model import LSTMForecaster

# ── Config ────────────────────────────────────────────────────────────────────
FEATURE_FILES = {
    "C1": Path("data/processed/features_c1.csv"),
    "C2": Path("data/processed/features_c2.csv"),
    "C3": Path("data/processed/features_c3.csv"),
}
RESULTS_DIR = Path("results")
SEQ_LEN     = 20    # lookback window — source default is 60, but our dataset
                    # is ~224 rows; 60 leaves too few val/test windows
BATCH_SIZE  = 64
MAX_EPOCHS  = 100
LR          = 1e-3
PATIENCE    = 10
HIDDEN_SIZE = 64
NUM_LAYERS  = 2
DROPOUT     = 0.2
TRAIN_FRAC  = 0.70
VAL_FRAC    = 0.15
SEED        = 42

torch.manual_seed(SEED)
np.random.seed(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Dataset helpers ───────────────────────────────────────────────────────────

def make_windows(features: np.ndarray, targets: np.ndarray, seq_len: int):
    """
    Build sliding-window (X, y) arrays.
    X[i] = features[i : i+seq_len]   shape (seq_len, n_features)
    y[i] = targets[i + seq_len]       scalar
    """
    X, y = [], []
    for i in range(len(features) - seq_len):
        X.append(features[i : i + seq_len])
        y.append(targets[i + seq_len])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


# ── Data loading ──────────────────────────────────────────────────────────────

def load_splits(csv_path: Path):
    """
    Load a pre-scaled feature CSV and return train/val/test tensors
    plus the test-set dates for saving predictions.

    Windows are built over the FULL series, then split by target date.
    Window i has target at index (i + SEQ_LEN), so:
      train : target index < train_end
      val   : train_end <= target index < val_end
      test  : target index >= val_end
    This ensures val/test have SEQ_LEN-sized lookback even when the
    individual splits are shorter than SEQ_LEN rows (which happens with
    our ~224-row dataset and larger lookback values).
    """
    df = pd.read_csv(csv_path, index_col="date", parse_dates=True)
    n  = len(df)

    train_end = int(n * TRAIN_FRAC)
    val_end   = train_end + int(n * VAL_FRAC)

    feature_cols = [c for c in df.columns if c != "target"]
    X = df[feature_cols].values.astype(np.float32)
    y = df["target"].values.astype(np.float32)
    dates = df.index

    # Build all windows once over the full series
    X_all, y_all = make_windows(X, y, SEQ_LEN)
    # Window i → target row index (i + SEQ_LEN) in the original df

    train_cut = train_end - SEQ_LEN   # last train window index (exclusive)
    val_cut   = val_end   - SEQ_LEN   # last val window index   (exclusive)

    X_train, y_train = X_all[:train_cut],          y_all[:train_cut]
    X_val,   y_val   = X_all[train_cut:val_cut],   y_all[train_cut:val_cut]
    X_test,  y_test  = X_all[val_cut:],            y_all[val_cut:]

    # Target dates for test windows: rows val_end .. n-1
    test_dates = dates[val_end:]

    print(f"  Windows — train: {len(X_train)}  val: {len(X_val)}  test: {len(X_test)}")
    return X_train, y_train, X_val, y_val, X_test, y_test, test_dates, X.shape[1]


# ── Training ──────────────────────────────────────────────────────────────────

def train_model(condition: str, csv_path: Path) -> tuple:
    """
    Train a LSTMForecaster for one condition.
    Returns (metrics_dict, predictions_df, loss_history_df).
    """
    print(f"\n{'='*60}")
    print(f"  Training {condition}  ({csv_path.name})")
    print(f"{'='*60}")

    X_train, y_train, X_val, y_val, X_test, y_test, test_dates, n_features = load_splits(csv_path)
    print(f"  Features: {n_features}")

    def make_loader(X, y, shuffle):
        ds = TensorDataset(torch.tensor(X), torch.tensor(y))
        return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle)

    train_dl = make_loader(X_train, y_train, shuffle=True)   # shuffle independent windows
    val_dl   = make_loader(X_val,   y_val,   shuffle=False)
    test_dl  = make_loader(X_test,  y_test,  shuffle=False)

    model     = LSTMForecaster(n_features, HIDDEN_SIZE, NUM_LAYERS, DROPOUT).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()

    best_val_loss = float("inf")
    best_state    = None
    patience_cnt  = 0
    loss_history  = []

    for epoch in range(1, MAX_EPOCHS + 1):
        # ── train ──
        model.train()
        train_loss = 0.0
        for xb, yb in train_dl:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item() * xb.size(0)
        train_loss /= len(X_train)

        # ── validate ──
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_dl:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                val_loss += criterion(model(xb), yb).item() * xb.size(0)
        val_loss /= max(len(X_val), 1)

        loss_history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})

        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:>3}  train={train_loss:.6f}  val={val_loss:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state    = {k: v.clone() for k, v in model.state_dict().items()}
            patience_cnt  = 0
        else:
            patience_cnt += 1
            if patience_cnt >= PATIENCE:
                print(f"  Early stopping at epoch {epoch}  (best val={best_val_loss:.6f})")
                break

    model.load_state_dict(best_state)
    metrics, pred_df = evaluate(model, test_dl, test_dates, condition)

    loss_df = pd.DataFrame(loss_history)
    return metrics, pred_df, loss_df


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate(model, test_dl, test_dates, condition: str) -> tuple:
    """
    Compute MAE, RMSE, MAPE, R², and directional accuracy on the test set.
    """
    model.eval()
    preds, actuals = [], []
    with torch.no_grad():
        for xb, yb in test_dl:
            preds.append(model(xb.to(DEVICE)).cpu().numpy())
            actuals.append(yb.numpy())

    preds   = np.concatenate(preds)
    actuals = np.concatenate(actuals)

    mae  = float(np.mean(np.abs(preds - actuals)))
    rmse = float(np.sqrt(np.mean((preds - actuals) ** 2)))

    nonzero = actuals != 0
    mape = float(np.mean(np.abs((preds[nonzero] - actuals[nonzero])
                                 / actuals[nonzero])) * 100) if nonzero.any() else float("nan")

    ss_res = np.sum((actuals - preds) ** 2)
    ss_tot = np.sum((actuals - actuals.mean()) ** 2)
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")

    dir_acc = float(np.mean(np.sign(preds) == np.sign(actuals)) * 100)

    metrics = {
        "MAE":                 mae,
        "RMSE":                rmse,
        "MAPE (%)":            mape,
        "R²":                  r2,
        "Directional Acc (%)": dir_acc,
    }

    print(f"\n  Test results for {condition}:")
    for k, v in metrics.items():
        print(f"    {k:<22}: {v:.4f}")

    pred_df = pd.DataFrame({
        "date":      test_dates,
        "actual":    actuals,
        "predicted": preds,
    }).set_index("date")

    return metrics, pred_df


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print(f"Device: {DEVICE}")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    all_metrics = {}
    for condition, path in FEATURE_FILES.items():
        if not path.exists():
            print(f"Skipping {condition} — {path} not found.")
            continue
        metrics, pred_df, loss_df = train_model(condition, path)
        all_metrics[condition] = metrics
        pred_df.to_csv(RESULTS_DIR / f"predictions_{condition}.csv")
        loss_df.to_csv(RESULTS_DIR / f"loss_{condition}.csv", index=False)

    metrics_df = pd.DataFrame(all_metrics).T
    metrics_df.to_csv(RESULTS_DIR / "metrics.csv")

    print(f"\n{'='*60}")
    print("  RESULTS COMPARISON")
    print(f"{'='*60}")
    print(metrics_df.round(4).to_string())
    print(f"\nSaved predictions and metrics → {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
