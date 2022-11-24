# %%
# ============================= IMPORTS =============================

import os
# This makes a certain kind of error message more legible
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

import plotly.express as px
import torch as t
import re
import numpy as np
from torch import nn, optim
import pandas as pd
from dataclasses import dataclass
from einops import rearrange, repeat
from fancy_einsum import einsum
from typing import Optional, Union
from tqdm.notebook import tqdm_notebook
from IPython.display import display
from torch.utils.data import Dataset, DataLoader

MAIN = __name__ == "__main__"

device = t.device("cuda" if t.cuda.is_available() else "cpu")
assert str(device) == "cuda"

from w1d2_chapter1_transformer_building.solutions import TransformerConfig, DecoderOnlyTransformer


# %%
# ============================= TRAINING LOOP =============================

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



# ============================= SHAKESPEARE =============================

# Load the text data
if MAIN:
    with open("100-0.txt", encoding="utf-8") as file:
        text = file.read()
        words = re.split(r"\b", text)

# %%

class WordsDataset(Dataset):
    def __init__(self, words, seq_len, fraction):
        """
        `fraction` is so we can scale down the amount of training that we do (otherwise it's a big dataset!). 
        
        This parameter will change the total length, and hence changes epoch duration (from hours to minutes).
        """
        self.fraction = fraction
        self.seq_len = seq_len
        self.words = words
        # Max len is less than # words, because we need to take a slice of tokens for getitem
        self.max_len = len(self.words) - (self.seq_len + 1)
        self.vocab_size = len(set(words))
        self.words_to_token_idx = {word: idx for (idx, word) in enumerate(sorted(set(words)))}
        self.token_idx_to_words = {idx: word for (word, idx) in self.words_to_token_idx.items()}
        self.tokens = t.tensor([self.words_to_token_idx[word] for word in words]).to(dtype=t.long)

    def __len__(self):
        return int(self.max_len * self.fraction)

    def __getitem__(self, idx):
        # Given tokens (t_1, ..., t_n), we want to predict (t_2, ..., t_n+1)
        # This is actually n separate instances of task "predict j+1th token from first j tokens", for 1<=j<=n
        x_and_y = self.tokens[idx: idx + self.seq_len + 1]
        x = x_and_y[:-1]
        y = x_and_y[1:]
        return x, y

if MAIN:
    max_seq_len = 48
    trainset = WordsDataset(words=words, seq_len=max_seq_len, fraction=0.02)

    batch_size = 32
    trainloader = DataLoader(trainset, shuffle=True, pin_memory=True, batch_size=batch_size)

# Create a tokenizer, so I can do things like tokenizer.encode(initial_text) and tokenizer.decode(list_of_ids)
# Also using the `return_tensors` argument of the encode method, just like gpt's tokenizer does
# This object is optional though, you could just use `self.words_to_token_idx` directly from dataset
class WordsTokenizer():
    def __init__(self, wordsdataset: WordsDataset):
        self.words_to_token_idx = wordsdataset.words_to_token_idx
        self.token_idx_to_words = wordsdataset.token_idx_to_words

    def encode(self, initial_text: str, return_tensors: Optional[str] = None) -> Union[list, np.ndarray, t.Tensor]:
        list_of_strings = [s for s in re.split(r"\b", initial_text) if len(s) > 0]
        tensors_list = [self.words_to_token_idx[s] for s in list_of_strings]
        if return_tensors is None:
            return tensors_list
        elif return_tensors == "pt":
            return t.tensor(tensors_list)
        elif return_tensors == "np":
            return np.array(tensors_list)
        else:
            raise Exception("Unexpected value for `return_tensors`.")

    def decode(self, list_of_ids: Union[t.Tensor, list]) -> str:
        return ''.join([self.token_idx_to_words[int(token)] for token in list_of_ids])

if MAIN:
    tokenizer = WordsTokenizer(trainset)

    config = TransformerConfig(
        num_layers = 8,
        num_heads = 8,
        vocab_size = trainset.vocab_size,
        hidden_size = 512,
        max_seq_len = trainset.seq_len,
        dropout = 0.1,
        layer_norm_epsilon = 1e-05
    )

    model = DecoderOnlyTransformer(config).to(device).train()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    epochs = 1

    model = train(model, optimizer, loss_fn, trainloader, epochs, "WordsDataset")
    # With this model and parameters, I had loss down to about 1.7 by the end of one epoch

# %%

# import the sampling methods
from w1d3_chapter1_transformer_building.solutions_sampling import *

def apply_sampling_methods(
    input_ids: t.Tensor, logits: t.Tensor, temperature=1.0, freq_penalty=0.0, top_k=0, top_p=0.0
) -> int:
    '''
    Return the next token, sampled from the model's probability distribution with modifiers.
x
    input_ids: shape (seq,)
    '''
    assert input_ids.ndim == 1, "input_ids should be a 1D sequence of token ids"
    assert temperature >= 0, "Temperature should be non-negative"
    assert 0 <= top_p <= 1.0, "Top-p must be a probability"
    assert 0 <= top_k, "Top-k must be non-negative"
    assert not (top_p != 0 and top_k != 0), "At most one of top-p and top-k supported"

    if temperature == 0:
        return greedy_search(logits)
    if temperature != 1.0:
        logits = apply_temperature(logits, temperature)
    if freq_penalty != 0.0:
        logits = apply_freq_penalty(input_ids, logits, freq_penalty)
    if top_k > 0:
        return sample_top_k(logits, top_k)
    if top_p > 0:
        return sample_top_p(logits, top_p)
    return sample_basic(logits)


def sample_tokens(
    model: DecoderOnlyTransformer,
    tokenizer: WordsTokenizer,
    initial_text: str,
    max_tokens_generated=30,
    **kwargs # kwargs are for params like temperature, top_k, etc
) -> str:
    '''
    Sample tokens until the model outputs `tokenizer.eos_token_id` or the specified token limit is reached.

    Return: the prompt and continuation concatenated
    '''
    # Note - an alternative to model.eval() is to use the @t.inference_mode() decorator for this whole function.
    model.eval()
    input_ids: list = tokenizer.encode(initial_text) # type: ignore
    generated = []
    for _ in range(max_tokens_generated):
        new_input_ids = t.tensor(input_ids + generated, dtype=t.long, device=device)
        new_input_ids_window = new_input_ids[-min(max_seq_len, new_input_ids.shape[0]):]
        logits = model(new_input_ids_window)[0, -1]
        new_token = apply_sampling_methods(new_input_ids, logits, **kwargs)
        generated.append(new_token)
        if new_token == getattr(tokenizer, "eos_token_id", None):
            break
    return tokenizer.decode(input_ids + generated)

if MAIN:
    # Note, some initial text strings might not work because they weren't present in the text you used for training
    initial_text = "turn down for what"

    text_output = sample_tokens(model, tokenizer, initial_text, max_tokens_generated=100, temperature=1.0, top_k=10)

    print(text_output)

# Result:

# turn down for what you do you think,
# That take the last, of many, which is so much I
# As this blows along than my life thou say’st, which makes thy hand,
# Thou wilt be given, or more
# Entitled in thy great world’s fresh blood will,
# To answer th’ alluring countenance, beauty 

# %%
