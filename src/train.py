"""
train.py
--------
Trains one StockLSTM per condition (C1, C2, C3) and evaluates each on
the held-out test set. Saves predictions and metrics to results/.

Split (strictly by time, no shuffling):
  Train : first 70%
  Val   : next  15%  (used for early stopping)
  Test  : last  15%

Sequence length: 20 trading days (sliding window)
Loss: MSE  |  Optimizer: Adam, lr=1e-3
Early stopping: patience=10 epochs on val loss

Outputs (written to results/):
  predictions_C1.csv, predictions_C2.csv, predictions_C3.csv
  metrics.csv
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from model import StockLSTM

# ── Config ────────────────────────────────────────────────────────────────────
FEATURE_FILES = {
    "C1": Path("data/processed/features_c1.csv"),
    "C2": Path("data/processed/features_c2.csv"),
    "C3": Path("data/processed/features_c3.csv"),
}
RESULTS_DIR = Path("results")
SEQ_LEN     = 20
BATCH_SIZE  = 16
MAX_EPOCHS  = 100
LR          = 1e-3
PATIENCE    = 10
HIDDEN_DIM  = 64
NUM_LAYERS  = 2
DROPOUT     = 0.2
TRAIN_FRAC  = 0.70
VAL_FRAC    = 0.15
SEED        = 42

torch.manual_seed(SEED)
np.random.seed(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Dataset ───────────────────────────────────────────────────────────────────

class SequenceDataset(Dataset):
    """Sliding-window dataset. Each sample is SEQ_LEN consecutive rows."""

    def __init__(self, features: np.ndarray, targets: np.ndarray, seq_len: int):
        self.X = torch.tensor(features, dtype=torch.float32)
        self.y = torch.tensor(targets,  dtype=torch.float32)
        self.seq_len = seq_len

    def __len__(self):
        return len(self.X) - self.seq_len

    def __getitem__(self, idx):
        x = self.X[idx : idx + self.seq_len]
        y = self.y[idx + self.seq_len]
        return x, y


# ── Data loading ──────────────────────────────────────────────────────────────

def load_splits(csv_path: Path):
    """Load a feature CSV and return train/val/test feature and target arrays."""
    df = pd.read_csv(csv_path, index_col="date", parse_dates=True)
    n  = len(df)

    train_end = int(n * TRAIN_FRAC)
    val_end   = train_end + int(n * VAL_FRAC)

    feature_cols = [c for c in df.columns if c != "target"]
    X = df[feature_cols].values.astype(np.float32)
    y = df["target"].values.astype(np.float32)
    dates = df.index

    return (
        X[:train_end],        y[:train_end],
        X[train_end:val_end], y[train_end:val_end],
        X[val_end:],          y[val_end:],
        dates[val_end:],      # test dates (for saving predictions)
    )


# ── Training ──────────────────────────────────────────────────────────────────

def train_model(condition: str, csv_path: Path) -> tuple:
    """
    Train a StockLSTM for one condition.
    Returns (metrics_dict, predictions_df, loss_history_df).
    """
    print(f"\n{'='*60}")
    print(f"  Training {condition}  ({csv_path.name})")
    print(f"{'='*60}")

    X_train, y_train, X_val, y_val, X_test, y_test, test_dates, = load_splits(csv_path)
    input_dim = X_train.shape[1]
    print(f"  Input dim: {input_dim}  |  Train: {len(X_train)}  Val: {len(X_val)}  Test: {len(X_test)}")

    train_ds = SequenceDataset(X_train, y_train, SEQ_LEN)
    val_ds   = SequenceDataset(X_val,   y_val,   SEQ_LEN)
    test_ds  = SequenceDataset(X_test,  y_test,  SEQ_LEN)

    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=False)
    val_dl   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False)
    test_dl  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False)

    model     = StockLSTM(input_dim, HIDDEN_DIM, NUM_LAYERS, DROPOUT).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()

    best_val_loss = float("inf")
    best_state    = None
    patience_cnt  = 0
    loss_history  = []

    for epoch in range(1, MAX_EPOCHS + 1):
        # Train
        model.train()
        train_loss = 0.0
        for xb, yb in train_dl:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item() * len(xb)
        train_loss /= len(train_ds)

        # Validate
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_dl:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                val_loss += criterion(model(xb), yb).item() * len(xb)
        val_loss /= max(len(val_ds), 1)

        loss_history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})

        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:>3}  train_loss={train_loss:.6f}  val_loss={val_loss:.6f}")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state    = {k: v.clone() for k, v in model.state_dict().items()}
            patience_cnt  = 0
        else:
            patience_cnt += 1
            if patience_cnt >= PATIENCE:
                print(f"  Early stopping at epoch {epoch} (best val_loss={best_val_loss:.6f})")
                break

    model.load_state_dict(best_state)

    # Evaluate
    metrics, pred_df = evaluate(model, test_dl, test_dates)
    print(f"\n  Test results for {condition}:")
    for k, v in metrics.items():
        print(f"    {k:<22}: {v:.4f}")

    loss_df = pd.DataFrame(loss_history)
    return metrics, pred_df, loss_df


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate(model: StockLSTM, test_dl: DataLoader, test_dates) -> tuple:
    """
    Compute MAE, RMSE, MAPE, R², and directional accuracy on the test set.
    Returns (metrics_dict, predictions_df with columns [date, actual, predicted]).
    """
    model.eval()
    preds, actuals = [], []

    with torch.no_grad():
        for xb, yb in test_dl:
            xb = xb.to(DEVICE)
            preds.append(model(xb).cpu().numpy())
            actuals.append(yb.numpy())

    preds   = np.concatenate(preds)
    actuals = np.concatenate(actuals)

    mae  = np.mean(np.abs(preds - actuals))
    rmse = np.sqrt(np.mean((preds - actuals) ** 2))

    nonzero = actuals != 0
    mape = np.mean(np.abs((preds[nonzero] - actuals[nonzero]) / actuals[nonzero])) * 100

    ss_res = np.sum((actuals - preds) ** 2)
    ss_tot = np.sum((actuals - actuals.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    dir_acc = np.mean(np.sign(preds) == np.sign(actuals)) * 100

    metrics = {
        "MAE":                 mae,
        "RMSE":                rmse,
        "MAPE (%)":            mape,
        "R²":                  r2,
        "Directional Acc (%)": dir_acc,
    }

    # Align dates: the dataset drops the first SEQ_LEN rows (no history yet)
    pred_dates = test_dates[SEQ_LEN:]
    pred_df = pd.DataFrame({
        "date":      pred_dates,
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
