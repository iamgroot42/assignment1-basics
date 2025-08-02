import torch
import torch.nn as nn

from cs336_basics.linear import Linear
from cs336_basics.utils import self_attention


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, device=None, dtype=None):
        super().__init__()
        self.num_heads =num_heads
        self.d_model = d_model
        self.device = device
        self.q_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.k_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.v_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.o_proj = Linear(d_model, d_model, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Create view that splits heads
        q = q.view(*x.shape[:-1], self.num_heads, -1)
        k = k.view(*x.shape[:-1], self.num_heads, -1)
        v = v.view(*x.shape[:-1], self.num_heads, -1)

        # q,k,v are shape (batch_size, seq_len, num_heads, d_k)
        # Transpose to bring num_heads after batch_size
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        seq_len = x.shape[-2]
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=self.device), diagonal=1)
        causal_mask = ~causal_mask.bool()  # Mask set to True for positions that should be masked

        # We want to compute attention scores independently for each head
        # Can make it so that n_heads is treated as a batch dimension
        mha = self_attention(q, k, v, mask=causal_mask)
    
        # Reshape things back so that we may concat the attentions before using projection
        mha = mha.transpose(1, 2)
        # Squeeze last 2 dimensions into one
        mha = mha.flatten(start_dim=-2)

        output = self.o_proj(mha)
        return output
