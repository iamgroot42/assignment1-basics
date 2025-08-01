import torch
from einops import einsum
import torch.nn as nn


class Embedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, device=None, dtype=None):
        super().__init__()
        self.W = nn.init.trunc_normal_(
            torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype),
            mean=0,
            std=1
        )
        # Truncate weights at [-3sigma, 3sigma]
        self.W.clamp_(-3, 3)
        self.W = nn.Parameter(self.W)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        token_ids_shape = token_ids.shape
        token_ids = token_ids.view(-1, 1)
        embeddings = self.W[token_ids].view(*token_ids_shape, -1)
        return embeddings
