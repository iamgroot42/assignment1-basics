import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.gain = torch.ones(d_model, device=device, dtype=dtype)
        self.gain = nn.Parameter(self.gain)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Take note of x dtype
        x_dtype = x.dtype
        # Upcase to float32
        x = x.to(torch.float32)

        # Mean with eps
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        scaled_in_features = x / rms
        # Scale with weights
        out_features = scaled_in_features * self.gain

        # Scale back to original dtype
        out_features = out_features.to(x_dtype)
        return out_features
