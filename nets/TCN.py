import torch
import torch.nn as nn
from torch.nn.utils import weight_norm

class TCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1):
        super().__init__()
        self.conv = weight_norm(nn.Conv1d(in_channels, out_channels, kernel_size,
                                         padding=(kernel_size-1)//2 * dilation,
                                         dilation=dilation))
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.conv(x))

class TinyTCN(nn.Module):
    def __init__(self, input_size=6, output_dim=64, num_layers=2):
        super().__init__()
        layers = []
        channels = input_size
        for i in range(num_layers):
            layers.append(TCNBlock(channels, output_dim, kernel_size=3, dilation=2**i))
            channels = output_dim
        self.tcn = nn.Sequential(*layers)
        self.global_pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        x = self.tcn(x)          # (B, output_dim, L)
        x = self.global_pool(x)  # (B, output_dim, 1)
        return x.squeeze(-1)     # (B, output_dim)