"""
model.py — LSTM model definition for next-day log-return prediction (condition B).

Architecture: 2-layer LSTM → linear head → scalar output
Training logic lives in train_eval.py.
"""

import torch
import torch.nn as nn


class LSTMForecaster(nn.Module):
    """
    2-layer LSTM that maps a sequence of daily feature vectors to a
    scalar prediction of the next-day log return.

    Args:
        input_size:  number of features per time step
        hidden_size: LSTM hidden units (default 64)
        num_layers:  stacked LSTM layers (default 2)
        dropout:     dropout between LSTM layers (default 0.2)
    """

    def __init__(self, input_size: int, hidden_size: int = 64,
                 num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,       # applied between layers (no effect when num_layers=1)
            batch_first=True,      # input shape: (batch, seq_len, input_size)
        )
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, input_size)
        Returns:
            out: (batch,) — predicted log return for each sequence
        """
        lstm_out, _ = self.lstm(x)          # (batch, seq_len, hidden_size)
        last_hidden = lstm_out[:, -1, :]    # take the final time step
        out = self.head(last_hidden)        # (batch, 1)
        return out.squeeze(1)               # (batch,)
