# %%
import hashlib
import os
import zipfile
import torch as t
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.data.dataset import TensorDataset
import transformers
from einops import rearrange
import plotly.express as px
import wandb
import torch.nn.functional as F
from tqdm import tqdm
import requests
import utils
from dataclasses import dataclass

from w1d4_chapter1_further_investigations.solutions_build_bert import TransformerConfig, BertLanguageModel, predict

# %%

MAIN = __name__ == "__main__"
DATA_FOLDER = "./data"
DATASET = "2"
BASE_URL = "https://s3.amazonaws.com/research.metamind.io/wikitext/"
DATASETS = {"103": "wikitext-103-raw-v1.zip", "2": "wikitext-2-raw-v1.zip"}
TOKENS_FILENAME = os.path.join(DATA_FOLDER, f"wikitext_tokens_{DATASET}.pt")

if not os.path.exists(DATA_FOLDER):
    os.mkdir(DATA_FOLDER)

# %%

def maybe_download(url: str, path: str) -> None:
    """Download the file from url and save it to path. If path already exists, do nothing."""
    if not os.path.exists(path):
        with open(path, "wb") as file:
            data = requests.get(url).content
            file.write(data)

# %%

if MAIN:
    path = os.path.join(DATA_FOLDER, DATASETS[DATASET])

    maybe_download(BASE_URL + DATASETS[DATASET], path)
    expected_hexdigest = {"103": "0ca3512bd7a238be4a63ce7b434f8935", "2": "f407a2d53283fc4a49bcff21bc5f3770"}
    with open(path, "rb") as f:
        actual_hexdigest = hashlib.md5(f.read()).hexdigest()
        assert actual_hexdigest == expected_hexdigest[DATASET]

    print(f"Using dataset WikiText-{DATASET} - options are 2 and 103")
    tokenizer = transformers.AutoTokenizer.from_pretrained("bert-base-cased")

    z = zipfile.ZipFile(path)

    def decompress(*splits: str) -> str:
        return [
            z.read(f"wikitext-{DATASET}-raw/wiki.{split}.raw").decode("utf-8").splitlines()
            for split in splits
        ]

    train_text, val_text, test_text = decompress("train", "valid", "test")
# %%


import functools
def concat_lists(list_of_lists):
    return functools.reduce(lambda x, y: x+y, list_of_lists)

def tokenize_1d(tokenizer, lines: list, max_seq: int) -> t.Tensor:
    """Tokenize text and rearrange into chunks of the maximum length.

    Return (batch, seq) and an integer dtype.
    """

    lines_tokenized = tokenizer(
        lines, 
        truncation=False, 
        add_special_tokens=False, 
        padding=False,
        return_token_type_ids=False,
        return_attention_mask=False,
    )
    input_ids = lines_tokenized["input_ids"]
    input_ids = concat_lists(input_ids)

    n_to_truncate = len(input_ids) % max_seq
    input_ids = t.tensor(input_ids[:-n_to_truncate]).to(t.int)

    input_ids = rearrange(input_ids, "(b s) -> b s", s=max_seq)

    return input_ids


def tokenize_1d_with_bar(tokenizer, lines: "list[str]", max_seq: int, n_intervals: int) -> t.Tensor:
    input_ids = []
    interval_len = len(lines) // (n_intervals - 1)
    slices = [slice(i*interval_len, (i+1)*interval_len) for i in range(n_intervals)]
    progress_bar = tqdm(slices)
    for slice_ in progress_bar:
        lines_tokenized = tokenizer(
            lines[slice_], 
            truncation=False, 
            add_special_tokens=False, 
            padding=False,
            return_token_type_ids=False,
            return_attention_mask=False,
        )
        input_ids.append(concat_lists(lines_tokenized["input_ids"]))

    input_ids = concat_lists(input_ids)
    n_to_truncate = len(input_ids) % max_seq
    input_ids = t.tensor(input_ids[:-n_to_truncate]).to(t.int)

    input_ids = rearrange(input_ids, "(b s) -> b s", s=max_seq)

    return input_ids

if MAIN:
    max_seq = 128
    print("Tokenizing training text...")
    train_data = tokenize_1d_with_bar(tokenizer, train_text, max_seq, 100)
    print("Training data shape is: ", train_data.shape)
    print("Tokenizing validation text...")
    val_data = tokenize_1d(tokenizer, val_text, max_seq)
    print("Tokenizing test text...")
    test_data = tokenize_1d(tokenizer, test_text, max_seq)
    print("Saving tokens to: ", TOKENS_FILENAME)
    t.save((train_data, val_data, test_data), TOKENS_FILENAME)

# %%
import importlib
importlib.reload(utils)

def random_mask(
    input_ids: t.Tensor, mask_token_id: int, vocab_size: int, select_frac=0.15, mask_frac=0.8, random_frac=0.1
) -> tuple:
    '''Given a batch of tokens, return a copy with tokens replaced according to Section 3.1 of the paper.

    input_ids: (batch, seq)

    Return: (model_input, was_selected) where:

    model_input: (batch, seq) - a new Tensor with the replacements made, suitable for passing to the BertLanguageModel. Don't modify the original tensor!

    was_selected: (batch, seq) - 1 if the token at this index will contribute to the MLM loss, 0 otherwise
    '''

    input_ids_modified = input_ids.clone()

    # Create masks
    mask_seed = t.randperm(input_ids.numel()).reshape(input_ids.shape).to(input_ids.device)

    threshold_probabilities = t.tensor([
        0,
        select_frac * mask_frac,
        select_frac * (mask_frac + random_frac),
        select_frac
    ])
    threshold_values = input_ids.numel() * threshold_probabilities

    fill_values = [mask_token_id, input_ids.clone().random_(vocab_size)]
    for threshold_lower, threshold_higher, fill_value in zip(threshold_values[0:2], threshold_values[1:3], fill_values):
        input_ids_modified = t.where(
            (threshold_lower <= mask_seed) & (mask_seed < threshold_higher),
            fill_value,
            input_ids_modified
        )

    return input_ids_modified, mask_seed < threshold_values[-1]

if MAIN:
    utils.test_random_mask(random_mask, input_size=10000, max_seq=max_seq)

# %%

if MAIN:
    # Find the word frequencies
    word_frequencies = t.bincount(train_data.flatten())
    # Drop the words with occurrence zero (because these contribute zero to cross entropy)
    word_frequencies = word_frequencies[word_frequencies > 0]
    # Get probabilities
    word_probabilities = word_frequencies / word_frequencies.sum()
    # Calculate the cross entropy
    cross_entropy = (- word_probabilities * word_probabilities.log()).sum()
    print(cross_entropy) # ==> 7.3446

# %%


# %%

def flat(x: t.Tensor) -> t.Tensor:
    """Combines batch and sequence dimensions."""
    return rearrange(x, "b s ... -> (b s) ...")

def cross_entropy_selected(pred: t.Tensor, target: t.Tensor, was_selected: t.Tensor) -> t.Tensor:
    """
    pred: (batch, seq, vocab_size) - predictions from the model
    target: (batch, seq, ) - the original (not masked) input ids
    was_selected: (batch, seq) - 1 if the token at this index will contribute to the MLM loss, 0 otherwise

    Out: the mean loss per predicted token
    """
    target = t.where(was_selected.to(t.bool), target, -100).long()
    entropy = F.cross_entropy(flat(pred), flat(target))
    return entropy


if MAIN:
    utils.test_cross_entropy_selected(cross_entropy_selected)

    batch_size = 8
    seq_length = 512
    batch = t.randint(0, tokenizer.vocab_size, (batch_size, seq_length))
    pred = t.rand((batch_size, seq_length, tokenizer.vocab_size))
    (masked, was_selected) = random_mask(batch, tokenizer.mask_token_id, tokenizer.vocab_size)
    loss = cross_entropy_selected(pred, batch, was_selected).item()
    print(f"Random MLM loss on random tokens - does this make sense? {loss:.2f}")
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

hidden_size = 512
bert_config_tiny = TransformerConfig(
    num_layers = 8,
    num_heads = hidden_size // 64,
    vocab_size = 28996,
    hidden_size = hidden_size,
    max_seq_len = 128,
    dropout = 0.1,
    layer_norm_epsilon = 1e-12
)

config_dict = dict(
    lr=0.0002,
    epochs=40,
    batch_size=128,
    weight_decay=0.01,
    mask_token_id=tokenizer.mask_token_id,
    warmup_step_frac=0.01,
    eps=1e-06,
    max_grad_norm=None,
)

(train_data, val_data, test_data) = t.load("./data/wikitext_tokens_2.pt")
print("Training data size: ", train_data.shape)

# %%

train_loader = DataLoader(
    TensorDataset(train_data), shuffle=True, batch_size=config_dict["batch_size"], drop_last=True
)

# %%

def lr_for_step(step: int, max_step: int, max_lr: float, warmup_step_frac: float):
    '''Return the learning rate for use at this step of training.'''

    step_frac = step / max_step

    if step_frac < warmup_step_frac:
        return max_lr * (0.1 + 0.9 * step_frac/warmup_step_frac)
    else:
        return max_lr * (0.1 + 0.9 * (1-step_frac)/(1-warmup_step_frac))
    
if MAIN:
    max_step = int(len(train_loader) * config_dict["epochs"])
    lrs = [
        lr_for_step(step, max_step, max_lr=config_dict["lr"], warmup_step_frac=config_dict["warmup_step_frac"])
        for step in range(max_step)
    ]
    # TODO: YOUR CODE HERE, PLOT `lrs` AND CHECK IT RESEMBLES THE GRAPH ABOVE
    px.line(
        lrs, 
        template="simple_white",
        title="Learning rate",
        labels=dict(value="", index="step")
    ).update_layout(
        showlegend=False, 
        xaxis_range=[-100, 6000],
        yaxis_tickformat=".1e"
    ).show()

# %%


# ================================================================
# Note - this soln will depend on the details of your architecture
# you can also iterate through modules, 
# ================================================================

def make_optimizer(model: BertLanguageModel, config_dict: dict) -> t.optim.AdamW:
    '''
    Loop over model parameters and form two parameter groups:

    - The first group includes the weights of each Linear layer and uses the weight decay in config_dict
    - The second has all other parameters and uses weight decay of 0
    '''

    param_groups = [
        dict(params=[], weight_decay=config_dict["weight_decay"]),
        dict(params=[], weight_decay=0)
    ]

    for name, param in model.named_parameters():
        if "weight" in name and any([
            "attn" in name, "mlp" in name, "fc" in name
        ]):
            param_groups[0]["params"].append(param)
        else:
            param_groups[1]["params"].append(param)

    return t.optim.AdamW(
        params=param_groups,
        lr=config_dict["lr"],
        eps=config_dict["eps"]
    )


if MAIN:
    test_config = TransformerConfig(
        num_layers = 3,
        num_heads = 1,
        vocab_size = 28996,
        hidden_size = 1,
        max_seq_len = 4,
        dropout = 0.1,
        layer_norm_epsilon = 1e-12,
    )

    optimizer_test_model = BertLanguageModel(test_config)
    opt = make_optimizer(
        optimizer_test_model, 
        dict(weight_decay=0.1, lr=0.0001, eps=1e-06)
    )
    expected_num_with_weight_decay = test_config.num_layers * 6 + 1
    wd_group = opt.param_groups[0]
    actual = len(wd_group["params"])
    assert (
        actual == expected_num_with_weight_decay
    ), f"Expected 6 linear weights per layer (4 attn, 2 MLP) plus the final lm_linear weight to have weight decay, got {actual}"
    all_params = set()
    for group in opt.param_groups:
        all_params.update(group["params"])
    assert all_params == set(optimizer_test_model.parameters()), "Not all parameters were passed to optimizer!"

# %%

device = t.device("cuda:0" if t.cuda.is_available() else "cpu")
assert str(device) == "cuda:0"

def bert_mlm_pretrain(model: BertLanguageModel, config_dict: dict, train_loader: DataLoader) -> None:
    '''Train using masked language modelling.'''

    n_examples_seen = 0
    n_steps = 0

    model.to(device).train()
    
    optimizer = make_optimizer(model, config_dict)
    # Example of how you might use lr_scheduler (but it's easier just to change LRs directly!)
    # lr_scheduler = t.optim.lr_scheduler.LambdaLR(
    #     optimizer,
    #     lr_lambda = lambda step: lr_for_step(
    #         step = step,
    #         max_step = max_step, # global variable
    #         max_lr = config_dict["lr"], 
    #         warmup_step_frac = config_dict["warmup_step_frac"]
    #     ) / config_dict["lr"] # because lr_lambda applies a factor
    # )

    wandb.init()
    wandb.watch(model, log="all", log_freq=15)

    for epoch in range(config_dict["epochs"]):

        progress_bar = tqdm(train_loader)

        for (batch,) in progress_bar:

            n_examples_seen += batch.size(0)
            n_steps += 1

            batch = batch.to(device)
            model_input, was_selected = random_mask(
                input_ids=batch, 
                mask_token_id=tokenizer.mask_token_id, 
                vocab_size=tokenizer.vocab_size,
                select_frac=0.15,
                mask_frac=0.8, 
                random_frac=0.1
            )
            pred = model(model_input)

            loss = cross_entropy_selected(pred, batch, was_selected)
            loss.backward()

            if config_dict["max_grad_norm"] is not None:
                clipped_grads = nn.utils.clip_grad_norm_(model.parameters(), config_dict["max_grad_norm"])
            
            optimizer.step()
            optimizer.zero_grad()
            
            lr = lr_for_step(n_steps, max_step, config_dict["lr"], config_dict["warmup_step_frac"])
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

            progress_bar.set_description(f"Epoch {epoch+1}/{config_dict['epochs']}, Loss = {loss.item():>10.3f}, LRs = {lr:.3e}")
            wandb.log({"loss": loss.item(), "lr": lr}, step=n_examples_seen)

    wandb.run.save()
    wandb.finish()

    return model

# %%

if MAIN:
    model = BertLanguageModel(bert_config_tiny)
    num_params = sum((p.nelement() for p in model.parameters()))
    print("Number of model parameters: ", num_params)
    model = bert_mlm_pretrain(model, config_dict, train_loader)

    text = "Former President of the United States of America, George[MASK][MASK]"
    predictions = predict(model, tokenizer, text, k=15)