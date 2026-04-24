"""
model.py
--------
LSTMForecaster: adapted from
  github.com/AmirhosseinHonardoust/Stock-LSTM-Forecasting

Changes from source:
  - input_size is a required argument (multivariate features, not
    univariate close price)
  - Output is a single scalar — next-day log return (horizon fixed at 1)
  - squeeze(1) on output to match (batch,) shape expected by train.py
"""

import torch
import torch.nn as nn


class LSTMForecaster(nn.Module):
    """
    2-layer LSTM for next-day log return prediction.

    Args:
        input_size  : number of input features (varies by condition)
        hidden_size : LSTM hidden units          (default 64)
        num_layers  : stacked LSTM layers        (default 2)
        dropout     : dropout between LSTM layers (default 0.2)
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x   : (batch_size, seq_len, input_size)
        Returns:
            out : (batch_size,) — predicted next-day log return
        """
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :]).squeeze(1)
