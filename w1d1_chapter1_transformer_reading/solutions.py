import torch as t
from torch import nn
from typing import Union, List


# ============================= Positional encoding =============================

class PositionalEncoding(nn.Module):

    def __init__(self, max_seq_len: int, embedding_dim: int):
        super().__init__()
        # Defining our positional encoding array, with `max_seq_len` rows
        # This is an advantage of using sinusoidal encoding: we can easily expand to sequences of greater length without adding more learned params
        angles = t.outer(t.arange(max_seq_len), 1 / 10000 ** (2 * t.arange(embedding_dim//2) / embedding_dim))
        pe = t.zeros((max_seq_len, embedding_dim))
        pe[:, ::2] = t.sin(angles)
        pe[:, 1::2] = t.cos(angles)
        # Register array as a buffer, rather than parameter (we don't want it to be updated by gradient descent)
        self.register_buffer('pe', pe)

    def forward(self, x: t.Tensor) -> t.Tensor:
        """
        x: shape (batch, seq_len, embedding_dim)
        """
        batch, seq_len, embedding_dim = x.shape
        # We slice the positional encoding, so it's the same shape as x
        # This is equivalent to just using an nn.Embedding, but having the input be t.arange(seq_len)
        return x + self.pe[:seq_len, :] # type: ignore



# ============================= Embedding =============================

class Embedding(nn.Module):
    num_embeddings: int
    embedding_dim: int
    weight: nn.Parameter

    def __init__(self, num_embeddings: int, embedding_dim: int):

        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = nn.Parameter(t.randn(num_embeddings, embedding_dim) * 0.02)

    def forward(self, x: t.LongTensor) -> t.Tensor:
        return self.weight[x]

    def extra_repr(self) -> str:
        return f"num_embeddings={self.num_embeddings}, embedding_dim={self.embedding_dim}"

def test_embedding(Embedding):
    """Indexing into the embedding should fetch the corresponding rows of the embedding."""
    emb = Embedding(6, 100)
    out = emb(t.tensor([1, 3, 5], dtype=t.int64))
    t.testing.assert_close(out[0], emb.weight[1])
    t.testing.assert_close(out[1], emb.weight[3])
    t.testing.assert_close(out[2], emb.weight[5])



# ============================= LayerNorm =============================

class LayerNorm(nn.Module):

    def __init__(self, normalized_shape: Union[int, List[int]], eps: float = 1e-05, elementwise_affine: bool = True):
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = normalized_shape
        self.eps = eps
        
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = nn.Parameter(t.ones(normalized_shape))
            self.bias = nn.Parameter(t.zeros(normalized_shape))

    def forward(self, x: t.Tensor) -> t.Tensor:
        assert len(self.normalized_shape) <= len(x.shape)

        dims = tuple(range(len(x.shape)-len(self.normalized_shape), len(x.shape)))

        mean = x.mean(dim=dims, keepdims=True)
        var = x.var(dim=dims, unbiased=False, keepdims=True)

        x = (x - mean) / ((var + self.eps) ** 0.5)
        if self.elementwise_affine:
            x = x * self.weight + self.bias
        return x

    def extra_repr(self) -> str:
        return f"normalized_shape={self.normalized_shape}, eps{self.eps}, elementwise_affine={self.elementwise_affine}"



# ============================= GELU =============================

class GELU(nn.Module):

    def forward(self, x: t.Tensor) -> t.Tensor:
        return 0.5 * x * (1 + t.tanh((2 / t.pi) * (x + 0.044715 * x**3)))



# ============================= Dropout =============================

class Dropout(nn.Module):

    def __init__(self, p: float):
        super().__init__()
        assert 0 <= p < 1
        self.p = p

    def forward(self, x: t.Tensor) -> t.Tensor:
        if self.training:
            # Apply dropout: mask all but p of the activations, and scale the remainder
            mask = (t.rand(size=x.shape) < self.p).to(x.device)
            return t.where(mask, 0.0, x / (1 - self.p))
        else:
            # If not in training mode, dropout doesn't do anything
            return x

    def extra_repr(self) -> str:
        return f"p={self.p}"