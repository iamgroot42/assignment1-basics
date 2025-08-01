import torch
import torch.nn as nn

from cs336_basics.linear import Linear


class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        # have a 2d pre-computed buffer of sin and cos values
        # created during init with self.register_buffer(persistent=False
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        
        # Create all R^i_k matrices for i uptil max_seq_len and k up to d_k
        rows = torch.arange(max_seq_len, device=device).unsqueeze(1)
        columns = torch.pow(self.theta, (2 * torch.arange(1, d_k // 2 + 1, device=device)) // d_k)
        angles = rows * columns
        sin_values = torch.sin(angles)
        cos_values = torch.cos(angles)
        self.register_buffer('sin_values', sin_values, persistent=True)
        self.register_buffer('cos_values', cos_values, persistent=True)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        # Get position part according to [[cos(theta), -sin(theta)], [sin(theta), cos(theta)]]
        # Rotate each [2k-1, 2k] pair in input x according to the position
        # x is of shape [..., seq_len, d_k]
        sin_part = self.sin_values[token_positions]
        cos_part = self.cos_values[token_positions]
        # TODO