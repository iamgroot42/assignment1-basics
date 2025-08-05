

import torch
import torch.nn as nn
from einops import einsum

from jaxtyping import Float, Int


def silu(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)

def softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    x_ = x - x.max(dim=dim, keepdim=True).values
    exp_x = torch.exp(x_)
    return exp_x / exp_x.sum(dim=dim, keepdim=True)


def self_attention(Q: torch.Tensor,
                   K: torch.Tensor,
                   V: torch.Tensor,
                   mask: torch.Tensor | None = None) -> torch.Tensor:
    d_k = Q.shape[-1]
    # We need to perform softmax(Q @ K / sqrt(d)) @ V
    QK = einsum(Q, K, "batch_size ... n d_k, batch_size ... m d_k -> batch_size ... n m")
    # Scale outputs
    QK = QK / (d_k ** 0.5)
    # Apply mask (fill out everything not in mask with inf)
    QK.masked_fill_(~mask, float('-inf'))
    # Apply softmax
    QK = softmax(QK, dim=-1)
    # Multiply by V
    out = einsum(QK, V, "batch_size ... n m, batch_size ... m d_v -> batch_size ... n d_v")
    return out


def cross_entropy(inputs: Float[torch.Tensor, " batch_size vocab_size"],
                  targets: Int[torch.Tensor, " batch_size"]) -> Float[torch.Tensor, ""]:
    # Use the same numerical stability trick as your softmax function
    inputs_max = inputs.max(dim=-1, keepdim=True).values
    inputs_stable = inputs - inputs_max
    
    # Compute log-softmax directly: log(exp(x_i) / sum(exp(x_j))) = x_i - log(sum(exp(x_j)))
    log_sum_exp = torch.log(torch.exp(inputs_stable).sum(dim=-1, keepdim=True))
    log_probs = inputs_stable - log_sum_exp
    
    # Gather the log probabilities for the target classes
    target_log_probs = log_probs.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
    
    # Compute negative log likelihood
    nll = -target_log_probs
    
    # Return the mean of the negative log likelihood
    return nll.mean()
