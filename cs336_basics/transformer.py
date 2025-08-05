import torch
import torch.nn as nn
from jaxtyping import Float, Int

from cs336_basics.swiglu import SwiGLU
from cs336_basics.mha import MultiHeadSelfAttention
from cs336_basics.rmsnorm import RMSNorm
from cs336_basics.rope import RotaryPositionalEmbedding
from cs336_basics.linear import Linear

from cs336_basics.embedding import Embedding


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int,
                 d_ff: int, max_seq_len: int = None,
                 theta: int = None,
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


class Transformer(nn.Module):
    def __init__(self, vocab_size: int, context_length: int,
                 num_layers: int, num_heads: int,
                 d_model: int, d_ff: int, rope_theta: float,
                 device = None, dtype = None):
        super().__init__()
        # Create embedding layer
        self.embedding = Embedding(vocab_size, d_model, device=device, dtype=dtype)
        # Create transformer blocks
        layers = []
        for _ in range(num_layers):
            layer = TransformerBlock(d_model, num_heads, d_ff, context_length, rope_theta, device=device, dtype=dtype)
            layers.append(layer)
        self.layers = nn.ModuleList(layers)
        self.ln_final = RMSNorm(d_model, device=device, dtype=dtype)
        self.lm_head = Linear(d_model, vocab_size)

    
    def forward(self, x: Int[torch.Tensor, " batch_size sequence_length"]) -> Float[torch.Tensor, "batch_size sequence_length vocab_size"]:
        # First, get embeddings
        x_embed = self.embedding(x)
        # Pass through each layer
        for layer in self.layers:
            x_embed = layer(x_embed)
        
        # Apply norm
        out = self.ln_final(x_embed)
        # Classification head
        logits = self.lm_head(out)
        # Return logits
        return logits
