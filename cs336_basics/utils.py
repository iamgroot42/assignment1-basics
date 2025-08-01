

import torch
import torch.nn as nn


def silu(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)

def softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    x_ = x - x.max(dim=dim, keepdim=True).values
    exp_x = torch.exp(x_)
    return exp_x / exp_x.sum(dim=dim, keepdim=True)


def attention(Q: torch.Tensor, K: torch.Tensor,
              V: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    d_k = Q.shape[-1]
    # We need to perform softmax(Q @ K / sqrt(d)) @ V
    QK = Q.matmul(K.transpose(-2, -1))
    # Scale outputs
    QK = QK / (d_k ** 0.5)
    # Apply mask (fill out everything not in mask with inf)
    QK.masked_fill_(~mask, float('-inf'))
    # Apply softmax
    QK = softmax(QK, dim=-1)
    # Multiply by V
    out = QK.matmul(V)
    return out
