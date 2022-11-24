# %%
import os
# This makes a certain kind of error message more legible
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

import plotly.express as px
import torch as t
import numpy as np
from torch import nn, optim
import pandas as pd
from dataclasses import dataclass
from einops import rearrange, repeat
from fancy_einsum import einsum
from typing import Optional
from tqdm.notebook import tqdm_notebook
from IPython.display import display
from torch.utils.data import Dataset, DataLoader

device = t.device("cuda" if t.cuda.is_available() else "cpu")
assert str(device) == "cuda"

from w1d1_chapter1_transformer_reading.solutions import PositionalEncoding

MAIN = __name__ == "__main__"

# ============================= TRANSFORMER ARCHITECTURE =============================

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

def multihead_masked_attention(Q, K, V, num_heads):

    # Rearrange Q, K and V to separate the `headsize` dimension (because this is the one we take the inner product over)
    q = rearrange(Q, "batch seq (nheads headsize) -> batch seq nheads headsize", nheads=num_heads)
    k = rearrange(K, "batch seq (nheads headsize) -> batch seq nheads headsize", nheads=num_heads)
    v = rearrange(V, "batch seq (nheads headsize) -> batch seq nheads headsize", nheads=num_heads)

    # Calculate attention scores as inner product of q and k, and divide by sqrt(headsize)
    batch, seq_len, nheads, headsize = q.shape
    attention_scores = einsum("batch seqQ nheads headsize, batch seqK nheads headsize -> batch nheads seqQ seqK", q, k) / (headsize ** 0.5)

    # Create the attention mask
    # Note we don't need to add batch and nheads, for broadcasting reasons
    # Also note you could do this with much less code using e.g. t.triu(t.ones(...)), but this way is more explicit
    q_idx = repeat(t.arange(seq_len), "seqQ -> seqQ seqK", seqK=seq_len)
    k_idx = repeat(t.arange(seq_len), "seqK -> seqQ seqK", seqQ=seq_len)
    # Any index positions with q<k should be masked (this prevents tokens "reading info from the future")
    mask = (q_idx >= k_idx).to(device)
    neg_inf = t.tensor(-1e6, dtype=attention_scores.dtype, device=device)
    attention_scores = t.where(mask, attention_scores, neg_inf)

    # Take softmax over the key dimension (i.e. each query index has a corresponding probability distribution over tokens in the sequence)
    attention_probabilities = attention_scores.softmax(dim=-1)

    # Get attention values by taking convex combination of value vectors according to the attention probabilities
    attention_values = einsum("batch nheads seqQ seqK, batch seqK nheads headsize -> batch seqQ nheads headsize", attention_probabilities, v)

    # Rearrange to combine the nheads and headsize dimensions
    return rearrange(attention_values, "batch seqQ nheads headsize -> batch seqQ (nheads headsize)")


# %%

class MultiheadMaskedAttention(nn.Module):
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

    def forward(self, x: t.Tensor) -> t.Tensor:
        """
        x: shape (batch, seq, hidden_size)

        Return: shape (batch, seq, hidden_size)
        """
        # Computationally faster to apply W_QKV on x before splitting
        QKV = self.W_QKV(x)
        Q, K, V = t.split(QKV, self.num_heads*self.head_size, dim=-1)

        masked_attention_values = multihead_masked_attention(Q, K, V, num_heads=self.num_heads)

        output = self.W_O(masked_attention_values)

        return output


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

class DecoderBlock(nn.Module):
    
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        self.ln2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        self.attn = MultiheadMaskedAttention(config.hidden_size, config.num_heads)
        self.mlp = MLP(config)
        
    def forward(self, x: t.Tensor) -> t.Tensor:
        x = x + self.ln1(self.attn(x))
        x = x + self.ln2(self.mlp(x))
        return x

        
class DecoderOnlyTransformer(nn.Module):
    
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.positional_encoding = PositionalEncoding(config.max_seq_len, config.hidden_size)
        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)
        self.layers = nn.Sequential(*[DecoderBlock(config) for _ in range(config.num_layers)])
        self.ln = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)

        # Function to print a dataframe visualising parameter count (this can be omitted, but it's pretty useful!)
        if config.print_param_count:
            print(f"Total params = {sum([param.numel() for param in self.parameters()])}")
            with pd.option_context("display.max_rows", 1000):
                df = pd.DataFrame([
                    {"name": name, "shape": tuple(param.shape), "num params": param.numel()}
                    for name, param in self.named_parameters()
                ])
                display(df.style.background_gradient(cmap="viridis", subset=["num params"], gmap=np.log(df["num params"])))

        
    def forward(self, x: t.Tensor) -> t.Tensor:
        # If x has no batch dimension, give it one (this means the transformer can also be run on 1D inputs with no batch dimension)
        if len(x.shape) == 1:
            x = rearrange(x, "seq -> 1 seq")
        # Apply token embedding before positional encoding (this is easier, because then PE can just be added to x)
        x = self.token_embedding(x)
        x = self.positional_encoding(x)
        x = self.dropout(x)
        x = self.layers(x)
        x = self.ln(x)
        x = einsum("batch seq hidden, vocab hidden -> batch seq vocab", x, self.token_embedding.weight)
        return x
# %%



# ============================= REVERSED SEQUENCES =============================

class ReverseDataset(Dataset):
    def __init__(self, seq_len):
        self.seq_len = seq_len
        self.vocab_size = 10 # digits from 0 to 9 inclusive
        self.size = 10 ** seq_len # so that each seq appears once in the dataset (in expectation)

    def __len__(self):
        # This is what is returned when you call len(dataset)
        # And it's what PyTorch uses to construct the dataset when initialised
        return self.size

    def __getitem__(self, idx):
        # Rather than randomising, could also generate every single sequence
        seq = t.randint(self.vocab_size, size=(self.seq_len,), dtype=t.long)
        seq_reversed = seq.flip(-1)
        return seq, seq_reversed

if MAIN:
    # Create dataset for training
    seq_len = 6
    trainset = ReverseDataset(seq_len=seq_len)

    batch_size = 1024
    trainloader = DataLoader(trainset, shuffle=True, batch_size=batch_size)

# %%
if MAIN:
    config = TransformerConfig(
        num_layers = 2,
        num_heads = 6,
        vocab_size = trainset.vocab_size,
        hidden_size = 96,
        max_seq_len = trainset.seq_len,
    )

    model = DecoderOnlyTransformer(config).to(device).train()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    epochs = 2

# %%

def train(model, optimizer, loss_fn, trainloader, epochs, dataset_name=None, plot_loss=True):

    loss_list = []

    for epoch in range(epochs):
        
        progress_bar = tqdm_notebook(trainloader)
        
        for (x, y) in progress_bar:
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            
            logits = model(x)
            # logits dimensions are (batch, seq, digits), but we care about probabilities for each digit
            # so we need to reshape into (batch * seq, digits)
            loss = loss_fn(rearrange(logits, "b s d -> (b s) d"), y.flatten())
            loss.backward()

            optimizer.step()
            
            progress_bar.set_description(f"epoch = {epoch+1}, loss = {loss.item():.4f}")

            loss_list.append(loss.item())

    # Function to plot the loss over epochs
    if plot_loss:
        fig = px.line(
            y=loss_list, 
            template="simple_white", 
            labels={
                "x": "No. batches seen", 
                "y": str(loss_fn).replace("()", "") # This gets a name like "CrossEntropyLoss" from the loss function
            }, 
            title=f"Training loss on {dataset_name} dataset" if dataset_name is not None else "Training loss"
        )
        # This next bit of code plots vertical lines corresponding to the epochs
        if epochs > 1:
            for idx, epoch_start in enumerate(np.linspace(0, len(loss_list), epochs, endpoint=False)):
                fig.add_vline(x=epoch_start, line_width=3, line_dash="dash", annotation_text=f"Epoch {idx}", annotation_position="top right")
        fig.show()
    
    return model
# %%

if MAIN:
    model = train(model, optimizer, loss_fn, trainloader, epochs, "ReversedDigits")
# With this model and parameters, I found loss dropping to about 1.17 after second epoch

# %%

if MAIN:
    model.eval()
    seq = t.randint(10, size=(6,), dtype=t.long, device=device)
    seq_reversed = seq.flip(-1)
    logits = model(seq)
    prediction = logits.argmax(dim=-1).squeeze()
    print("prediction:", prediction)
    print("answer:", seq_reversed)
    t.testing.assert_close(seq_reversed[-3:], prediction[-3:])
    # As expected, model is getting the first three digits wrong, but the last three incorrect (so attention masking is working)