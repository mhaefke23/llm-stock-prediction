"""
model.py
--------
Defines the LSTM model used for all three conditions (C1, C2, C3).
The architecture is identical across conditions — only input_dim changes.

Architecture:
  - 2-layer LSTM (hidden_size=64, dropout between layers)
  - Fully connected output layer → single scalar (next-day log return)
"""

import torch
import torch.nn as nn


class StockLSTM(nn.Module):
    """
    2-layer LSTM for next-day log return prediction.

    Args:
        input_dim  : number of input features (varies by condition)
        hidden_dim : number of LSTM hidden units (default 64)
        num_layers : number of stacked LSTM layers (default 2)
        dropout    : dropout probability between LSTM layers (default 0.2)
    """

    def __init__(self, input_dim: int, hidden_dim: int = 64, num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,           # input shape: (batch, seq_len, input_dim)
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : (batch_size, seq_len, input_dim)
        Returns:
            out : (batch_size,) — predicted next-day log return
        """
        lstm_out, _ = self.lstm(x)
        # Use only the last time step's output for prediction
        last = lstm_out[:, -1, :]       # (batch_size, hidden_dim)
        out  = self.fc(last).squeeze(1) # (batch_size,)
        return out
