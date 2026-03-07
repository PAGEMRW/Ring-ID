import torch
import torch.nn as nn

class TinyLSTM(nn.Module):
    def __init__(self, input_size=6, hidden_size=64):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)  # (B, n, L) -> (B, L, n)
        _, (h, _) = self.lstm(x)
        return h[-1]  # (B, hidden_size)