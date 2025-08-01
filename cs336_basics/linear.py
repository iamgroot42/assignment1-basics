import torch
import math
from einops import einsum
import torch.nn as nn


class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, device=None, dtype=None):
        super().__init__()
        sigma = math.sqrt(2 / (in_features + out_features))
        self.W = nn.init.trunc_normal_(
            torch.empty(out_features, in_features, device=device, dtype=dtype),
            mean=0,
            std=sigma
        )
        # Truncate weights at [-3sigma, 3sigma]
        self.W.clamp_(-3 * sigma, 3 * sigma)
        self.W = nn.Parameter(self.W)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out_features = einsum(self.W, x, "out_features in_features, ... in_features -> ... out_features")
        return out_features
