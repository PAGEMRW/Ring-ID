import torch
import torch.nn as nn

class TinyGRU(nn.Module):
    def __init__(self,input_size=6,hidden_size=64):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True
        )

    def forward(self, x):
        # (B, 6, 1250) -> (B, 1250, 6)
        x = x.permute(0, 2, 1)
        _, h = self.gru(x)
        return h[-1]
