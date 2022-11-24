# %%

import os
# This makes a certain kind of error message more legible
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

import torch as t
from torch import nn
import torch.nn.functional as F
from dataclasses import dataclass
from einops import rearrange, reduce, repeat
from fancy_einsum import einsum
from typing import Optional

MAIN = __name__ == "__main__"

device = t.device("cuda" if t.cuda.is_available() else "cpu")

# %%

@dataclass(frozen=True)
class TransformerConfig:
    '''Constants used throughout your decoder-only transformer model.'''

    num_layers: int
    num_heads: int
    vocab_size: int
    hidden_size: int
    max_seq_len: int
    dropout: float = 0.1
    layer_norm_epsilon: float = 1e-05
    print_param_count: bool = True

# %%

class MultiheadAttention(nn.Module):
    W_QKV: nn.Linear
    W_O: nn.Linear

    def __init__(self, hidden_size: int, num_heads: int, head_size: Optional[int] = None):
        """
        Adding option to override head_size (defaults to hidden_size / num_heads otherwise)
        """
        super().__init__()
        self.hidden_size = hidden_size
        assert hidden_size % num_heads == 0
        self.num_heads = num_heads
        self.head_size = hidden_size // num_heads if head_size is None else head_size
        
        # Note that these weight matrices are usually called projections and defined as linear layers without bias, but they are 
        # still implemented with bias in some papers.
        self.W_QKV = nn.Linear(hidden_size, 3*num_heads*self.head_size)
        self.W_O = nn.Linear(num_heads*self.head_size, hidden_size)

    def forward(self, x: t.Tensor, additive_attention_mask: Optional[t.Tensor]) -> t.Tensor:
        """
        x: shape (batch, seq, hidden_size)

        Return: shape (batch, seq, hidden_size)
        """
        QKV = self.W_QKV(x)
        Q, K, V = t.split(QKV, self.num_heads*self.head_size, dim=-1)

        attention_values = self.multihead_attention(Q, K, V, additive_attention_mask, num_heads=self.num_heads)

        output = self.W_O(attention_values)

        return output

    # Now moving this function into a class method, so it can refer to the dropout layers
    def multihead_attention(self, Q: t.Tensor, K: t.Tensor, V: t.Tensor, additive_attention_mask: Optional[t.Tensor], num_heads: int) -> t.Tensor:

        q = rearrange(Q, "batch seq (nheads headsize) -> batch seq nheads headsize", nheads=num_heads)
        k = rearrange(K, "batch seq (nheads headsize) -> batch seq nheads headsize", nheads=num_heads)
        v = rearrange(V, "batch seq (nheads headsize) -> batch seq nheads headsize", nheads=num_heads)

        batch, seq_len, nheads, headsize = q.shape
        attention_scores = einsum("batch seqQ nheads headsize, batch seqK nheads headsize -> batch nheads seqQ seqK", q, k) / (headsize ** 0.5)
        if additive_attention_mask is not None:
            attention_scores = attention_scores + additive_attention_mask

        attention_probabilities = attention_scores.softmax(dim=-1)

        attention_values = einsum("batch nheads seqQ seqK, batch seqK nheads headsize -> batch seqQ nheads headsize", attention_probabilities, v)

        out = rearrange(attention_values, "batch seqQ nheads headsize -> batch seqQ (nheads headsize)")

        return out


# %%

class MLP(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        self.fc1 = nn.Linear(config.hidden_size, 4 * config.hidden_size)
        self.fc2 = nn.Linear(4 * config.hidden_size, config.hidden_size)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(config.dropout)
        
    def forward(self, x: t.Tensor) -> t.Tensor:
        return self.dropout(self.fc2(self.gelu(self.fc1(x))))

# %%

class TransformerBlock(nn.Module):
    
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        self.ln2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        self.attn = MultiheadAttention(config.hidden_size, config.num_heads)
        self.mlp = MLP(config)
        
    def forward(self, x: t.Tensor, additive_attention_mask: Optional[t.Tensor]) -> t.Tensor:
        x = x + self.ln1(self.attn(x, additive_attention_mask))
        x = x + self.ln2(self.mlp(x))
        return x

# %%
