import torch
import torch.nn as nn


class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        # have a 2d pre-computed buffer of sin and cos values
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        
        # Create all R^i_k matrices for i uptil max_seq_len, k up to d_k
        rows = torch.arange(max_seq_len, device=device).unsqueeze(1)
        column_exponents = (2 * torch.arange(d_k // 2, device=device)) / d_k
        columns = torch.pow(self.theta, -column_exponents)
        angles = rows * columns # Gives us i / theta^(2k/d)
        sin_values = torch.sin(angles)
        cos_values = torch.cos(angles)
        self.register_buffer('sin_values', sin_values, persistent=False)
        self.register_buffer('cos_values', cos_values, persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        # x shape: (..., seq_len, d_k)
        # token_positions shape: (..., seq_len)
        sin_values = self.sin_values[token_positions.contiguous().view(-1)]#.view(*token_positions.shape, -1)
        cos_values = self.cos_values[token_positions.contiguous().view(-1)]#.view(*token_positions.shape, -1)

        sin_values = sin_values.view(*token_positions.shape, -1)
        cos_values = cos_values.view(*token_positions.shape, -1)
        
        x_even = x[..., ::2]
        x_odd  = x[..., 1::2]
        
        # Apply rotation
        x_rot = torch.empty_like(x)
        x_rot[..., ::2] = x_even * cos_values - x_odd * sin_values  # (..., seq_len, d_k//2)
        x_rot[..., 1::2] = x_even * sin_values + x_odd * cos_values  # (..., seq_len, d_k//2)
        
        # Return the rotated tensor
        return x_rot