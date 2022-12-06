# %%

import pandas as pd
from IPython.display import display
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch as t
from torch import nn, optim
import torch.nn.functional as F
import plotly.express as px
import numpy as np
from einops import repeat, rearrange
from fancy_einsum import einsum
from dataclasses import dataclass
from typing import Optional, Callable, Tuple

from my_transformer import BalancedBracketsTransformer, TransformerConfig
from balanced_brackets import SimpleTokenizer, BracketsDataset

MAIN = __name__ == "__main__"

device = t.device("cuda" if t.cuda.is_available() else "cpu")

if MAIN:
    max_len = 18 # this is max length of brackets, so we actually pad to seq_len=20
    seq_len = max_len + 2

    config = TransformerConfig(
        num_layers = 3,
        num_heads = 2,
        hidden_size = 56,
        vocab_size = 5, # five types of tokens: "(", ")", [start], [pad] and [end]
        max_seq_len = seq_len,
    )

    tokenizer = SimpleTokenizer("()")
    trainset = BracketsDataset(tokenizer, size=10_000, max_length=6, fraction_balanced=0.5)

# %%

# ==== Check output looks reasonable ====
if MAIN:
    bracket_string = "()()()"
    bracket_array_padded, one_zero_attention_mask = tokenizer.tokenize(bracket_string, max_len=max_len)
    model = BalancedBracketsTransformer(config).to(device)
    output = model(bracket_array_padded.to(device), one_zero_attention_mask.to(device))
    print(output)

# ================================== Building and testing my transformer ==================================

# %%

@dataclass
class BracketTransformerArgs():
    num_layers: int = 3
    num_heads: int = 2
    hidden_size: int = 56
    vocab_size: int = 5
    max_len: int = 18
    trainset_size: int = 100_000
    testset_size: int = 1_000
    epochs: int = 5
    batch_size: int = 512
    fraction_balanced: float = 0.5
    print_output_every: Optional[int] = 10
    plot_loss: bool = True
    loss_fn: Callable = nn.CrossEntropyLoss()
    lr: float = 1e-3
    betas: Tuple[float, float] = (0.9, 0.999)

    def __init__(self):
        self.max_seq_len = self.max_len + 2


# %%

def train(args: BracketTransformerArgs):

    tokenizer = SimpleTokenizer("()")
    trainset = BracketsDataset(tokenizer, size=args.trainset_size, max_length=args.max_len, fraction_balanced=args.fraction_balanced)
    testset = BracketsDataset(tokenizer, size=args.testset_size, max_length=args.max_len, fraction_balanced=args.fraction_balanced)
    trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True)
    testloader = DataLoader(testset, batch_size=args.batch_size, shuffle=True)

    config = TransformerConfig(
        num_layers = args.num_layers,
        num_heads = args.num_heads,
        hidden_size = args.hidden_size,
        vocab_size = args.vocab_size,
        max_seq_len = args.max_seq_len,
    )
    model = BalancedBracketsTransformer(config).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=args.betas)


    loss_list = []
    accuracy_list = []

    model = model.train().to(device)

    for epoch in range(args.epochs):

        # TRAINING LOOP

        model.train()
        progress_bar = tqdm(trainloader)
        for _, (brackets, one_zero_attention_mask, bracket_str, label) in enumerate(progress_bar):

            optimizer.zero_grad()
            
            logits = model(brackets.to(device), one_zero_attention_mask.to(device))
            loss = args.loss_fn(logits, label.to(device, dtype=t.long))
            loss.backward()

            optimizer.step()
            
            progress_bar.set_description(f"epoch = {epoch+1}, loss = {loss.item():.4f}")

            loss_list.append(loss.item())

        # TESTING LOOP

        model.eval()
        with t.inference_mode():
            accuracy = 0
            for _, (brackets, one_zero_attention_mask, bracket_str, label) in enumerate(testloader):
                
                logits = model(brackets.to(device), one_zero_attention_mask.to(device))
                predictions = logits.argmax(-1).cpu()

                accuracy += (predictions == label).sum().item()
            
            # Add accuracy to list, and print it out
            accuracy_list.append(accuracy)
            print(f"epoch = {epoch+1}, accuracy = {accuracy}/{len(testset)}")
            # Also print out some sample outputs
            for _ in range(5):
                brackets, one_zero_attention_mask, bracket_str, label = testset.random_sample()
                output = model(brackets.to(device), one_zero_attention_mask.to(device))
                output_probabilities = F.softmax(output, dim=-1)
                output_probabilities = output_probabilities.cpu().numpy().squeeze().round(6).tolist()
                print(f"{repr(bracket_str):{max_len+3}}, {label}, output = {output_probabilities}")


    # Function to plot the loss over epochs
    if args.plot_loss:
        fig = px.line(
            y=loss_list, 
            template="simple_white", 
            labels={
                "x": "No. batches seen", 
                "y": str(args.loss_fn).replace("()", "") # This gets a name like "CrossEntropyLoss" from the loss function
            }, 
            title="Loss on ReversedDigits dataset"
        )
        # This next bit of code plots vertical lines corresponding to the epochs
        if args.epochs > 1:
            for idx, epoch_start in enumerate(np.linspace(0, len(loss_list), args.epochs, endpoint=False)):
                fig.add_vline(x=epoch_start, line_width=3, line_dash="dash", annotation_text=f"Epoch {idx}", annotation_position="top right")
        fig.update_layout(yaxis_range=[0, max(loss_list) * 1.1])
        fig.show()
    
    return model

# %%

if MAIN:
    args = BracketTransformerArgs()
    args.num_layers = 2
    args.num_heads = 2
    args.hidden_size = 32
    model = train(args)

# %%