import torch
import torch.nn as nn

class TinyConv1D(nn.Module):
    def __init__(self, input_size=6, output_dim=64):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(input_size, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(32, output_dim, kernel_size=5, padding=2),
            nn.ReLU()
        )
        self.global_pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        # (B, n, L) -> (B, output_dim)
        x = self.conv(x)           # (B, output_dim, L)
        x = self.global_pool(x)    # (B, output_dim, 1)
        return x.squeeze(-1)       # (B, output_dim)