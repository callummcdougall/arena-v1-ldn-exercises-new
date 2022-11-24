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

from my_transformer import TransformerConfig, TransformerBlock
from balanced_brackets import BalancedBracketsDataset, bracket_arr_to_string, bracket_string_to_arr, pad_bracket_array

MAIN = __name__ == "__main__"

device = t.device("cuda" if t.cuda.is_available() else "cpu")

if MAIN:
    max_len = 20 # this is max length of brackets, so the max seq len will be 13 (this is what we pad to)

    config = TransformerConfig(
        num_layers = 2,
        num_heads = 8,
        hidden_size = 32,
        vocab_size = 5, # five types of tokens: "(", ")", [START/CLS], [END] and [PAD]
        max_seq_len = max_len + 1,
        print_param_count = False
    )


# %%

class BalancedBracketsTransformer(nn.Module):
    
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.positional_encoding = nn.Embedding(config.max_seq_len, config.hidden_size)
        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)
        self.blocks = nn.Sequential(*[TransformerBlock(config) for _ in range(config.num_layers)])
        self.ln = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        self.final_linear = nn.Linear(config.vocab_size, 2)

        # Function to print a dataframe visualising parameter count (this can be omitted, but it's pretty useful!)
        if config.print_param_count:
            self.print_param_count()

    def print_param_count(self):
        print(f"Total params = {sum([param.numel() for param in self.parameters()])}")
        with pd.option_context("display.max_rows", 1000):
            df = pd.DataFrame([
                {"name": name, "shape": tuple(param.shape), "num params": param.numel()}
                for name, param in self.named_parameters()
            ])
            display(df.style.background_gradient(cmap="viridis", subset=["num params"], gmap=np.log(df["num params"])))

        
    def forward(self, x: t.Tensor, one_zero_attention_mask: t.Tensor) -> t.Tensor:

        assert x.shape == one_zero_attention_mask.shape

        # If x has no batch dimension, give it one (this means the transformer can also be run on 1D inputs with no batch dimension)
        # Also, get the additive attention mask
        if len(x.shape) == 1:
            x = rearrange(x, "seq -> 1 seq")
            additive_attention_mask = -1e5 * repeat(1 - one_zero_attention_mask, "seqK -> 1 1 1 seqK")
        else:
            additive_attention_mask = -1e5 * repeat(1 - one_zero_attention_mask, "batch seqK -> batch 1 1 seqK")

        # Apply main transformer blocks
        seq_len = x.size(-1)
        x = self.token_embedding(x) + self.positional_encoding(t.arange(seq_len).to(device))
        x = self.dropout(x)
        for block in self.blocks:
            x = block(x, additive_attention_mask)
        x = self.ln(x)
        x = einsum("batch seq hidden, vocab hidden -> batch seq vocab", x, self.token_embedding.weight)

        # Take the output corresponding to [CLS], and interpret those as logits
        x_cls = self.final_linear(x[:, 0, :])
        return x_cls

# %%

# ==== Check output looks reasonable ====
if MAIN:
    bracket_string = "()()()"
    bracket_array = bracket_string_to_arr(bracket_string)
    bracket_array_padded, one_zero_attention_mask = pad_bracket_array(bracket_array, final_length=max_len+1)
    model = BalancedBracketsTransformer(config)
    output = model(bracket_array_padded.to(device), one_zero_attention_mask.to(device))
    print(output)

# ================================== Building and testing my transformer ==================================

# %%

def train(model, optimizer, loss_fn, trainset, testset, batch_size, epochs, plot_loss=True):

    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=True)

    loss_list = []
    accuracy_list = []

    model = model.train().to(device)

    for epoch in range(epochs):

        # TRAINING LOOP

        model.train()
        progress_bar = tqdm(trainloader)
        for it, (padded_bracket_array, one_zero_attention_mask, balanced_label) in enumerate(progress_bar):

            padded_bracket_array = padded_bracket_array.to(device)
            one_zero_attention_mask = one_zero_attention_mask.to(device)
            balanced_label = balanced_label.to(device, dtype=t.long)

            optimizer.zero_grad()
            
            logits = model(padded_bracket_array, one_zero_attention_mask)
            # logits dimensions are (batch, seq, digits), but we care about probabilities for each digit
            # so we need to reshape into (batch * seq, digits)
            loss = loss_fn(logits, balanced_label)
            loss.backward()

            optimizer.step()
            
            progress_bar.set_description(f"epoch = {epoch+1}, loss = {loss.item():.4f}")

            loss_list.append(loss.item())

        # TESTING LOOP

        model.eval()
        with t.inference_mode():
            accuracy = 0
            for it, (padded_bracket_array, one_zero_attention_mask, balanced_label) in enumerate(testloader):
                
                padded_bracket_array = padded_bracket_array.to(device)
                one_zero_attention_mask = one_zero_attention_mask.to(device)
                balanced_label = balanced_label.to(device, dtype=t.long)

                logits = model(padded_bracket_array, one_zero_attention_mask)
                # logits dimensions are (batch, seq, digits), but we care about probabilities for each digit
                # so we need to reshape into (batch * seq, digits)
                predictions = logits.argmax(-1)

                accuracy += (predictions == balanced_label).sum().item()
            
            # Add accuracy to list, and print it out
            accuracy_list.append(accuracy)
            print(f"epoch = {epoch+1}, accuracy = {accuracy}/{len(testset)}")
            # Also print out some sample outputs
            for _ in range(5):
                padded_bracket_array, one_zero_attention_mask, balanced_label = testset[t.randint(high=len(testset), size=(1,)).item()]
                bracket_string = bracket_arr_to_string(padded_bracket_array)
                output = model(padded_bracket_array.to(device), one_zero_attention_mask.to(device))
                output_probabilities = F.softmax(output, dim=-1)
                output_probabilities = output_probabilities.cpu().numpy().squeeze().round(6).tolist()
                print(f"{repr(bracket_string):{max_len+3}}, {balanced_label}, output = {output_probabilities}")


    # Function to plot the loss over epochs
    if plot_loss:
        fig = px.line(
            y=loss_list, 
            template="simple_white", 
            labels={
                "x": "No. batches seen", 
                "y": str(loss_fn).replace("()", "") # This gets a name like "CrossEntropyLoss" from the loss function
            }, 
            title="Loss on ReversedDigits dataset"
        )
        # This next bit of code plots vertical lines corresponding to the epochs
        if epochs > 1:
            for idx, epoch_start in enumerate(np.linspace(0, len(loss_list), epochs, endpoint=False)):
                fig.add_vline(x=epoch_start, line_width=3, line_dash="dash", annotation_text=f"Epoch {idx}", annotation_position="top right")
        fig.show()
    
    return model

# %%

if MAIN:
    trainset = BalancedBracketsDataset(
        size = 50_000,
        max_length = max_len,
        length_distribution_function = lambda x: (x-1) ** 2,
        p_balanced = 0.5,
    )
    testset = BalancedBracketsDataset(
        size = 1_000,
        max_length = max_len,
        length_distribution_function = lambda x: (x-1) ** 2,
        p_balanced = 0.5,
    )

    model = BalancedBracketsTransformer(config).to(device)
    optimizer = optim.Adam(model.parameters())
    loss_fn = nn.CrossEntropyLoss()
    epochs = 10
    batch_size = 512

# %%

if MAIN:
    model = train(model, optimizer, loss_fn, trainset, testset, batch_size, epochs)

# %%