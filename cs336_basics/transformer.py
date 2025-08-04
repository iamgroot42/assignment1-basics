import torch
import torch.nn as nn
from jaxtyping import Float

from cs336_basics.swiglu import SwiGLU
from cs336_basics.mha import MultiHeadSelfAttention
from cs336_basics.rmsnorm import RMSNorm
from cs336_basics.rope import RotaryPositionalEmbedding


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int,
                 d_ff: int, max_seq_len: int = None, theta: int = None,
                 device=None, dtype=None):
        super().__init__()
        self.num_heads =num_heads
        self.d_model = d_model
        self.d_ff = d_ff
        self.max_seq_len = max_seq_len
        self.device = device

        rope_layer = None
        if max_seq_len is not None:
            rope_layer = RotaryPositionalEmbedding(theta, d_model // num_heads, max_seq_len, device=device)

        self.attn = MultiHeadSelfAttention(d_model, num_heads, rope_layer=rope_layer, device=device, dtype=dtype)
        self.ln1 = RMSNorm(d_model, device=device, dtype=dtype)
        self.ln2 = RMSNorm(d_model, device=device, dtype=dtype)
        self.ffn = SwiGLU(d_model, d_ff, device=device, dtype=dtype)

    def forward(self, x: Float[torch.Tensor, " batch sequence_length d_model"]) -> Float[torch.Tensor, " batch sequence_length d_model"]:
        mha_branch = self.ln1(x)
        mha_branch = self.attn(mha_branch)
        x = x + mha_branch

        ffn_branch = self.ln2(x)
        ffn_branch = self.ffn(ffn_branch)
        out = x + ffn_branch
        return out
