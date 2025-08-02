import torch
import torch.nn as nn

from cs336_basics.linear import Linear
from cs336_basics.utils import silu


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int, device=None, dtype=None):
        super().__init__()
        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)
        self.w3 = Linear(d_model, d_ff, device=device, dtype=dtype)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w1x = silu(self.w1(x))
        w3x = self.w3(x)
        element_wise = w1x * w3x
        output = self.w2(element_wise)
        return output
