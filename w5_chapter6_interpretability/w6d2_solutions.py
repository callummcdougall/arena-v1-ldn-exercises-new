# %%
import plotly.io as pio
pio.renderers.default = "notebook_connected" # or use "browser" if you want plots to open with browser

import torch as t
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from einops import repeat, rearrange, reduce
from fancy_einsum import einsum
import tqdm.auto as tqdm
import random
from pathlib import Path
import plotly.express as px
from torch.utils.data import DataLoader

from torchtyping import TensorType as TT
from torchtyping import patch_typeguard
from typeguard import typechecked
from typing import List, Union, Optional, Tuple
from functools import partial
import copy

import itertools
from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer
import dataclasses
import datasets
from IPython.display import HTML, display

import circuitsvis as cv
import transformer_lens
import transformer_lens.utils as utils
from transformer_lens.hook_points import HookedRootModule, HookPoint  # Hooking utilities
from transformer_lens import HookedTransformer, HookedTransformerConfig, FactoredMatrix, ActivationCache

# Saves computation time, since we don't need it for the contents of this notebook
t.set_grad_enabled(False)

def imshow(tensor, renderer=None, xaxis="", yaxis="", caxis="", **kwargs):
    px.imshow(utils.to_numpy(tensor), color_continuous_midpoint=0.0, color_continuous_scale="RdBu", labels={"x":xaxis, "y":yaxis, "color":caxis}, **kwargs).show(renderer)

def line(tensor, renderer=None, xaxis="", yaxis="", **kwargs):
    px.line(utils.to_numpy(tensor), labels={"x":xaxis, "y":yaxis}, **kwargs).show(renderer)

def scatter(x, y, xaxis="", yaxis="", caxis="", renderer=None, **kwargs):
    x = utils.to_numpy(x)
    y = utils.to_numpy(y)
    px.scatter(y=y, x=x, labels={"x":xaxis, "y":yaxis, "color":caxis}, **kwargs).show(renderer)

# %%

MAIN = __name__ == "__main__"

device = t.device("cuda" if t.cuda.is_available() else "cpu")

cfg = HookedTransformerConfig(
    d_model=768,
    d_head=64,
    n_heads=12,
    n_layers=2,
    n_ctx=2048,
    d_vocab=50278,
    attention_dir="causal", # defaults to "bidirectional"
    attn_only=True, # defaults to False

    tokenizer_name="EleutherAI/gpt-neox-20b", 
    # if setting from config, set tokenizer this way rather than passing it in explicitly
    # model initialises via AutoTokenizer.from_pretrained(tokenizer_name)

    seed=398,
    use_attn_result=True,
    normalization_type=None, # defaults to "LN", i.e. use layernorm with weights and biases

    positional_embedding_type="shortformer" # this makes it so positional embeddings are used differently (makes induction heads cleaner to study)
)

WEIGHT_PATH = "./data/attn_only_2L_half.pth"

if MAIN:
    model = HookedTransformer(cfg)
    raw_weights = model.state_dict()
    pretrained_weights = t.load(WEIGHT_PATH, map_location=device)
    model.load_state_dict(pretrained_weights)


# %%  

# ====================================
# BUILDING INTERPRETABILITY TOOLS
# ====================================

if MAIN:
    text = "We think that powerful, significantly superhuman machine intelligence is more likely than not to be created this century. If current machine learning techniques were scaled up to this level, we think they would by default produce systems that are deceptive or manipulative, and that no solid plans are known for how to avoid this."
    str_tokens = model.to_str_tokens(text)
    tokens = model.to_tokens(text)
    tokens = tokens.to(device)
    logits, cache = model.run_with_cache(tokens, remove_batch_dim=True)
    model.reset_hooks()
# %%

def to_numpy(tensor):
    """Helper function to convert things to numpy before plotting with Plotly."""
    return tensor.detach().cpu().numpy()


def convert_tokens_to_string(tokens, batch_index=0):
    if len(tokens.shape) == 2:
        tokens = tokens[batch_index]
    return [f"|{model.tokenizer.decode(tok)}|_{c}" for (c, tok) in enumerate(tokens)]


def plot_logit_attribution(logit_attr: TT["seq", "path"], tokens: TT["seq"]):
    tokens = tokens.squeeze()
    y_labels = convert_tokens_to_string(tokens[:-1])
    x_labels = ["Direct"] + [f"L{l}H{h}" for l in range(model.cfg.n_layers) for h in range(model.cfg.n_heads)]
    imshow(to_numpy(logit_attr), x=x_labels, y=y_labels, xaxis="Term", yaxis="Position", caxis="logit", height=25*len(tokens))

# %%

n_components = model.cfg.n_layers * model.cfg.n_heads + 1
seq_len = tokens.shape[-1]

patch_typeguard()  # must call this before @typechecked

@typechecked
def logit_attribution(
    embed: TT["seq_len", "d_model"],
    l1_results: TT["seq_len", "n_heads", "d_model"],
    l2_results: TT["seq_len", "n_heads", "d_model"],
    W_U: TT["d_model", "d_vocab"],
    tokens: TT["seq_len"],
) -> TT[seq_len-1, "n_components": n_components]:
    """
    We have provided 'W_U_to_logits' which is a (d_model, seq_next) tensor where each row is the unembed for the correct NEXT token at the current position.
    Inputs:
        embed: the embeddings of the tokens (i.e. token + position embeddings)
        l1_results: the outputs of the attention heads at layer 1 (with head as one of the dimensions)
        l2_results: the outputs of the attention heads at layer 2 (with head as one of the dimensions)
        W_U: the unembedding matrix
    Returns:
        Tensor representing the concatenation (along dim=-1) of logit attributions from:
            the direct path (position-1,1)
            layer 0 logits (position-1, n_heads)
            and layer 1 logits (position-1, n_heads)
    """
    W_U_to_logits = W_U[:, tokens[1:]]

    direct_path_logits = einsum("emb seq_next, seq_next emb -> seq_next", W_U_to_logits, embed[:-1]).unsqueeze(1)
    l1_logits = einsum("emb seq_next, seq_next n_heads emb -> seq_next n_heads", W_U_to_logits, l1_results[:-1])
    l2_logits = einsum("emb seq_next, seq_next n_heads emb -> seq_next n_heads", W_U_to_logits, l2_results[:-1])
    logit_attribution = t.concat([direct_path_logits, l1_logits, l2_logits], dim=-1)
    return logit_attribution
    
if MAIN:
    with t.inference_mode():
        batch_index = 0
        embed = cache["hook_embed"]
        l1_results = cache["result", 0] # same as cache["blocks.0.attn.hook_result"]
        l2_results = cache["result", 1]
        logit_attr = logit_attribution(embed, l1_results, l2_results, model.unembed.W_U, tokens[0])
        # Uses fancy indexing to get a len(tokens[0])-1 length tensor, where the kth entry is the predicted logit for the correct k+1th token
        correct_token_logits = logits[batch_index, t.arange(len(tokens[0]) - 1), tokens[batch_index, 1:]]
        t.testing.assert_close(logit_attr.sum(1), correct_token_logits, atol=1e-2, rtol=0)

if MAIN:
    embed = cache["hook_embed"]
    l1_results = cache["blocks.0.attn.hook_result"]
    l2_results = cache["blocks.1.attn.hook_result"]
    logit_attr = logit_attribution(embed, l1_results, l2_results, model.unembed.W_U, tokens[0])
    plot_logit_attribution(logit_attr, tokens)

# %%

# =============================
# VISUALISING ATTENTION PATTERNS
# =============================

if MAIN:
    for layer in range(model.cfg.n_layers):
        attention_pattern = cache["pattern", layer]
        cv.attention.attention_heads(tokens=str_tokens, attention=attention_pattern)
        # html = cv.attention.attention_heads(tokens=str_tokens, attention=attention_pattern)
        # with open(f"layer_{layer}_attention.html", "w") as f:
        #     f.write(str(html))

# =============================
# SUMMARIZING ATTENTION PATTERNS
# =============================

seq_len = len(tokens[0])

def current_attn_detector(cache: ActivationCache) -> List[str]:
    """
    Returns a list e.g. ["0.2", "1.4", "1.9"] of "layer.head" which you judge to be current-token heads
    """
    current_attn_heads = []
    for layer in range(model.cfg.n_layers):
        for head in range(model.cfg.n_heads):
            attention_pattern = cache["pattern", layer][head]
            # take avg of diagonal elements
            current_attn_score = attention_pattern[t.arange(seq_len), t.arange(seq_len)].mean()
            if current_attn_score > 0.4:
                current_attn_heads.append(f"{layer}.{head}")
    return current_attn_heads

def prev_attn_detector(cache: ActivationCache):
    """
    Returns a list e.g. ["0.2", "1.4", "1.9"] of "layer.head" which you judge to be prev-token heads
    """
    prev_attn_heads = []
    for layer in range(model.cfg.n_layers):
        for head in range(model.cfg.n_heads):
            attention_pattern = cache["pattern", layer][head]
            # take avg of subdiagonal elements
            prev_attn_score = attention_pattern[t.arange(seq_len-1)+1, t.arange(seq_len-1)].mean()
            if prev_attn_score > 0.4:
                prev_attn_heads.append(f"{layer}.{head}")
    return prev_attn_heads

def first_attn_detector(cache: ActivationCache):
    """
    Returns a list e.g. ["0.2", "1.4", "1.9"] of "layer.head" which you judge to be first-token heads
    """
    first_attn_heads = []
    for layer in range(model.cfg.n_layers):
        for head in range(model.cfg.n_heads):
            attention_pattern = cache["pattern", layer][head]
            # take avg of first column
            first_attn_score = attention_pattern[:, 0].mean()
            if first_attn_score > 0.4:
                first_attn_heads.append(f"{layer}.{head}")
    return first_attn_heads

if MAIN:
    # Compare this printout with your attention pattern visualisations. Do they make sense?
    print("Heads attending to current token  = ", ", ".join(current_attn_detector(cache)))
    print("Heads attending to previous token = ", ", ".join(prev_attn_detector(cache)))
    print("Heads attending to first token    = ", ", ".join(first_attn_detector(cache)))


# %%

# =============================
# ABLATIONS
# =============================

def cross_entropy_loss(logits, tokens):
    log_probs = F.log_softmax(logits, dim=-1)
    pred_log_probs = t.gather(log_probs[:, :-1], -1, tokens[:, 1:, None])[..., 0]
    return -pred_log_probs.mean()

@typechecked
def head_ablation(
    attn_result: TT["batch", "seq", "n_heads", "d_model"],
    hook: HookPoint,
    head_no: int
) -> TT["batch", "seq", "n_heads", "d_model"]:
    attn_result[:, :, head_no, :] = 0.0
    return attn_result

def get_ablation_scores(model: HookedTransformer, tokens: t.Tensor):

    ablation_scores = t.zeros((model.cfg.n_layers, model.cfg.n_heads), device=model.cfg.device)

    model.reset_hooks()
    logits, cache = model.run_with_cache(tokens, remove_batch_dim=True)

    loss_no_ablation = cross_entropy_loss(logits, tokens)

    for layer in tqdm(range(model.cfg.n_layers)):
        for head_no in range(model.cfg.n_heads):
            # Use functools.partial to create a temporary hook function with the head number fixed
            temp_hook_fn = partial(head_ablation, head_no=head_no)
            # Run the model with the patching hook
            patched_logits = model.run_with_hooks(tokens, fwd_hooks=[
                (utils.get_act_name("result", layer, "attn"), temp_hook_fn)
            ])
            # Calculate the logit difference
            loss = cross_entropy_loss(patched_logits, tokens)
            # Store the result, subtracting the clean loss so that a value of zero means no change in loss
            ablation_scores[layer, head_no] = loss - loss_no_ablation

    return ablation_scores

if MAIN:
    ablation_scores = get_ablation_scores(model, tokens)
    imshow(ablation_scores, xaxis="Head", yaxis="Layer", caxis="logit diff", title="Logit Difference After Ablating Heads", text_auto=".2f")

# Note - remember to run `model.reset_hooks()` !

# %%

def run_and_cache_model_repeated_tokens(
    model: HookedTransformer, 
    seq_len: int, 
    batch=1
) -> Tuple[t.Tensor, t.Tensor, ActivationCache]:
    """
    Generates a sequence of repeated random tokens, and runs the model on it, returning logits, tokens and cache

    Add a prefix token, since the model was always trained to have one.

    Outputs are:
    rep_logits: [batch, 1+2*seq_len, d_vocab]
    rep_tokens: [batch, 1+2*seq_len]
    rep_cache: {} The cache of the model run on rep_tokens
    """
    prefix = t.ones((batch, 1), dtype=t.int64, device=model.cfg.device) * model.tokenizer.bos_token_id
    rand_tokens = t.randint(0, model.tokenizer.vocab_size, (batch, seq_len)).to(model.cfg.device)
    rep_tokens = t.cat([prefix, rand_tokens, rand_tokens], dim=1)

    rep_logits, rep_cache = model.run_with_cache(rep_tokens, remove_batch_dim=False)

    return rep_logits, rep_tokens, rep_cache

def per_token_losses(logits, tokens):
    log_probs = F.log_softmax(logits, dim=-1)
    pred_log_probs = t.gather(log_probs[:, :-1], -1, tokens[:, 1:, None])[..., 0]
    return -pred_log_probs[0]

if MAIN:
    "\n    These are small numbers, since the results are very obvious and this makes it easier to visualise - in practice we'd obviously use larger ones on more subtle tasks. But it's often easiest to iterate and debug on small tasks.\n"
    seq_len = 50
    batch = 1
    (rep_logits, rep_tokens, rep_cache) = run_and_cache_model_repeated_tokens(model, seq_len, batch)
    rep_str = model.to_str_tokens(rep_tokens)
    model.reset_hooks()
    ptl = per_token_losses(rep_logits, rep_tokens)
    print(f"Performance on the first half: {ptl[:seq_len].mean():.3f}")
    print(f"Performance on the second half: {ptl[seq_len:].mean():.3f}")
    fig = px.line(
        to_numpy(ptl),
        hover_name=rep_str[1:],
        title=f"Per token loss on sequence of length {seq_len} repeated twice",
        labels={"index": "Sequence position", "value": "Loss"},
    ).update_layout(showlegend=False, hovermode="x unified")
    fig.add_vrect(x0=0, x1=49.5, fillcolor="red", opacity=0.2, line_width=0)
    fig.add_vrect(x0=49.5, x1=99, fillcolor="green", opacity=0.2, line_width=0)
    fig.show()

# %%

# Note - remember that this cache has a batch dim; you should remove this before running cv.attention.attention_heads

for layer in range(model.cfg.n_layers):
    attention_pattern = rep_cache["pattern", layer][0]
    html = cv.attention.attention_heads(tokens=rep_str, attention=attention_pattern)
    with open(f"layer_{layer}_attention.html", "w") as f:
        f.write(str(html))

# %%

def induction_attn_detector(cache: ActivationCache) -> List[str]:
    """
    Returns a list e.g. ["0.2", "1.4", "1.9"] of "layer.head" which you judge to be induction heads

    Remember:
        The tokens used to generate rep_cache are (bos_token, *rand_tokens, *rand_tokens)
    """
    induction_attn_heads = []
    for layer in range(model.cfg.n_layers):
        for head in range(model.cfg.n_heads):
            attention_pattern = cache["pattern", layer][0, head]
            # queries are all the tokens T which have been repeated before
            q_indices = t.arange(seq_len+1, 2*seq_len+1) # quer
            # keys are the tokens AFTER the FIRST occurrence of T
            k_indices = t.arange(2, seq_len+2)
            # get induction score
            induction_attn_score = attention_pattern[q_indices, k_indices].mean()
            if induction_attn_score > 0.4:
                induction_attn_heads.append(f"{layer}.{head}")
    return induction_attn_heads


if MAIN:
    print("Induction heads = ", ", ".join(induction_attn_detector(rep_cache)))

# %%

if MAIN:
    embed = rep_cache["hook_embed"][0]
    l1_results = rep_cache["blocks.0.attn.hook_result"][0]
    l2_results = rep_cache["blocks.1.attn.hook_result"][0]
    first_half_tokens = rep_tokens[0, :seq_len+1]
    second_half_tokens = rep_tokens[0, seq_len:]

    seq_len = 50
    first_half_logit_attr = logit_attribution(embed[:1+seq_len], l1_results[:1+seq_len], l2_results[:1+seq_len], model.unembed.W_U, first_half_tokens)
    second_half_logit_attr = logit_attribution(embed[seq_len:], l1_results[seq_len:], l2_results[seq_len:], model.unembed.W_U, second_half_tokens)

    plot_logit_attribution(first_half_logit_attr, first_half_tokens)
    plot_logit_attribution(second_half_logit_attr, second_half_tokens)

# %%

if MAIN:
    ablation_scores = get_ablation_scores(model, rep_tokens)
    imshow(ablation_scores, xaxis="Head", yaxis="Layer", caxis="logit diff", title="Logit Difference After Ablating Heads (detecting induction heads)", text_auto=".2f")




# ===================================
# Reverse-engineering induction heads
# ===================================
# %%

if MAIN:
    head_index = 4
    layer = 1
    W_U = model.unembed.W_U
    W_O_all = model.blocks[1].attn.W_O
    W_V_all = model.blocks[1].attn.W_V
    W_E = model.embed.W_E
    OV_circuit_full = einsum("emb1 voc1, d_k emb1, emb2 d_k, voc2 emb2 -> voc1 voc2", W_U, W_O_all[4], W_V_all[4], W_E)

# %%

if MAIN:
    rand_indices = t.randperm(model.cfg.d_vocab)[:200]
    imshow(to_numpy(OV_circuit_full[rand_indices][:, rand_indices]))
# %%

def top_1_acc(OV_circuit):
    '''
    This should take the argmax of each column (ie over dim=0) and return the fraction of the time that's equal to the correct logit
    '''
    argmaxes = OV_circuit.argmax(dim=0)
    diag_indices = t.arange(OV_circuit.shape[0]).to(argmaxes.device)

    return (argmaxes == diag_indices).float().mean()
    
def top_5_acc(OV_circuit):
    '''
    This should take the argmax of each column (ie over dim=0) and return the fraction of the time that's equal to the correct logit
    '''
    argmaxes = t.topk(OV_circuit, k=5, dim=0).indices
    diag_indices = t.arange(OV_circuit.shape[0]).to(argmaxes.device)

    return (argmaxes == diag_indices).any(dim=0).float().mean()


if MAIN:
    print("Fraction of the time that the best logit is on the diagonal:")
    print(top_1_acc(OV_circuit_full))
    print("Fraction of the time that one of the best five logits is on the diagonal:")
    print(top_5_acc(OV_circuit_full))

# %%

if MAIN:
    try:
        del OV_circuit_full
    except:
        pass
    W_OV_full = einsum("d_k emb1, emb2 d_k -> emb1 emb2", W_O_all[4], W_V_all[4]) + einsum("d_k emb1, emb2 d_k -> emb1 emb2", W_O_all[10], W_V_all[10])
    OV_circuit_full_both = einsum("emb1 voc1, emb1 emb2, voc2 emb2 -> voc1 voc2", W_U, W_OV_full, W_E)
    print("Top 1 accuracy for the full OV circuit:", top_1_acc(OV_circuit_full_both))
    print("Top 5 accuracy for the full OV circuit:", top_5_acc(OV_circuit_full_both))
    try:
        del OV_circuit_full_both
    except:
        pass

# %%

def mask_scores(
    attn_scores: TT["query_d_model", "key_d_model"]
):
    """Mask the attention scores so that tokens don't attend to previous tokens."""
    mask = t.tril(t.ones_like(attn_scores)).bool()
    neg_inf = t.tensor(-1.0e6).to(attn_scores.device)
    masked_attn_scores = t.where(mask, attn_scores, neg_inf)
    return masked_attn_scores

if MAIN:
    W_pos = model.pos_embed.W_pos
    W_Q_all = model.blocks[0].attn.W_Q
    W_K_all = model.blocks[0].attn.W_K
    pos_by_pos_scores = einsum(
        "ctx1 emb1, emb1 d_k, emb2 d_k, ctx2 emb2 -> ctx1 ctx2",
        W_pos,
        W_Q_all[7],
        W_K_all[7],
        W_pos
    )
    pos_by_pos_pattern = mask_scores(pos_by_pos_scores / model.cfg.d_head ** 0.5).softmax(-1)
    imshow(to_numpy(pos_by_pos_pattern[:200, :200]), xaxis="Key", yaxis="Query")

# %%

if MAIN:
    seq_len = rep_tokens.shape[1]
    n_heads = model.cfg.n_heads
    d_model = model.cfg.d_model
    d_head = model.cfg.d_head

@typechecked
def decompose_qk_input(cache: ActivationCache) -> TT[2+n_heads, seq_len, d_model]:
    '''
    Output is decomposed_qk_input, with shape [2+num_heads, position, d_model]
    '''
    emb: TT[seq_len, d_model] = cache["embed"][0]
    pos_embed: TT[seq_len, d_model] = cache["pos_embed"][0]
    layer0_head_output: TT[n_heads, seq_len, d_model] = rearrange(cache["result", 0][0], "pos nhead d_model -> nhead pos d_model")

    return t.concat([emb.unsqueeze(0), pos_embed.squeeze(0).unsqueeze(0), layer0_head_output], dim=0)

@typechecked
def decompose_q(decomposed_qk_input: TT[2+n_heads, seq_len, d_model], ind_head_index: int) -> TT[2+n_heads, seq_len, d_head]:
    '''
    Output is decomposed_q with shape [2+num_heads, position, d_head] (such that sum along axis 0 is just q)
    '''
    W_Q: TT[d_model, d_head] = model.blocks[1].attn.W_Q[ind_head_index]
    return einsum("component seq d_model, d_model d_k -> component seq d_k", decomposed_qk_input, W_Q)

@typechecked
def decompose_k(decomposed_qk_input: t.Tensor, ind_head_index: int) -> TT[2+n_heads, seq_len, d_head]:
    '''
    Output is decomposed_k with shape [2+num_heads, position, d_head] (such that sum along axis 0 is just k) - exactly analogous as for q
    '''
    return einsum("component seq emb, emb d_k -> component seq d_k", decomposed_qk_input, model.blocks[1].attn.W_K[ind_head_index])


if MAIN:
    ind_head_index = 4
    decomposed_qk_input = decompose_qk_input(rep_cache)
    t.testing.assert_close(decomposed_qk_input.sum(0), rep_cache["resid_pre", 1][0] + rep_cache["pos_embed"][0], rtol=0.01, atol=1e-05)
    decomposed_q = decompose_q(decomposed_qk_input, ind_head_index)
    t.testing.assert_close(decomposed_q.sum(0), rep_cache["blocks.1.attn.hook_q"][0, :, ind_head_index], rtol=0.01, atol=0.001)
    decomposed_k = decompose_k(decomposed_qk_input, ind_head_index)
    t.testing.assert_close(decomposed_k.sum(0), rep_cache["blocks.1.attn.hook_k"][0, :, ind_head_index], rtol=0.01, atol=0.01)
    component_labels = ["Embed", "PosEmbed"] + [f"L0H{h}" for h in range(model.cfg.n_heads)]
    imshow(to_numpy(decomposed_q.pow(2).sum([-1])), xaxis="Pos", yaxis="Component", title="Norms of components of query")
    imshow(to_numpy(decomposed_k.pow(2).sum([-1])), xaxis="Pos", yaxis="Component", title="Norms of components of key")

# %%

def decompose_attn_scores(
    decomposed_q: TT["q_component": 2+n_heads, "q_pos": seq_len, "d_k": d_head], 
    decomposed_k: TT["k_component": 2+n_heads, "k_pos": seq_len, "d_k": d_head], 
) -> TT["q_component": 2+n_heads, "k_component": 2+n_heads, "q_pos": seq_len, "k_pos": seq_len]:
    '''
    Output is decomposed_attn_scores with shape [2+num_heads, position, position]
    '''
    return einsum("q_component q_pos d_k, k_component k_pos d_k -> q_component k_component q_pos k_pos", decomposed_q, decomposed_k)

if MAIN:
    decomposed_scores = decompose_attn_scores(decomposed_q, decomposed_k)
    decomposed_stds = reduce(
        decomposed_scores, "query_decomp key_decomp query_pos key_pos -> query_decomp key_decomp", t.std
    )
    # First plot: std dev over query and key positions, shown by component
    imshow(to_numpy(t.tril(decomposed_scores[0, 9])), title="Attention Scores for component from Q=Embed and K=Prev Token Head")
    # Second plot: attention score contribution from (query_component, key_component) = (Embed, L0H7)
    imshow(to_numpy(decomposed_stds), xaxis="Key Component", yaxis="Query Component", title="Standard deviations of components of scores", x=component_labels, y=component_labels)

# %%

def find_K_comp_full_circuit(prev_token_head_index, ind_head_index):
    '''
    Returns a vocab x vocab matrix, with the first dimension being the query side and the second dimension being the key side (going via the previous token head)
    '''
    W_E = model.embed.W_E.half()

    W_Q = model.blocks[1].attn.W_Q[ind_head_index].half()
    W_K = model.blocks[1].attn.W_K[ind_head_index].half()
    QK_circuit = einsum("emb_Q d_model, emb_K d_model -> emb_Q emb_K", W_Q, W_K)
    
    W_V = model.blocks[0].attn.W_V[prev_token_head_index].half()
    W_O = model.blocks[0].attn.W_O[prev_token_head_index].half()
    OV_circuit = einsum("d_model emb_O, emb_V d_model -> emb_O emb_V", W_O, W_V)
    
    return einsum(
        "voc1 emb1, emb1 emb2, emb2 emb3, voc2 emb3 -> voc1 voc2",
        W_E, QK_circuit, OV_circuit, W_E
    )

if MAIN:
    ind_head_index = 4
    prev_token_head_index = 7
    K_comp_circuit = find_K_comp_full_circuit(prev_token_head_index, ind_head_index)
    print("Fraction of tokens where the highest activating key is the same token", top_1_acc(K_comp_circuit.T).item())
    del K_comp_circuit
# %%

def find_K_comp_full_full_circuit(prev_token_head_index, ind_head_indices):
    '''
    Returns a vocab x vocab matrix, with the first dimension being the query side and the second dimension being the key side (going via the previous token head)
    '''
    W_E = model.embed.W_E.half()

    W_Qs = [model.blocks[1].attn.W_Q[i].half() for i in ind_head_indices]
    W_Ks = [model.blocks[1].attn.W_K[i].half() for i in ind_head_indices]
    QK_circuits = [einsum("emb_Q d_model, emb_K d_model -> emb_Q emb_K", W_Q, W_K) for (W_Q, W_K) in zip(W_Qs, W_Ks)]
    QK_circuit = sum(QK_circuits)
    
    W_V = model.blocks[0].attn.W_V[prev_token_head_index].half()
    W_O = model.blocks[0].attn.W_O[prev_token_head_index].half()
    OV_circuit = einsum("d_model emb_O, emb_V d_model -> emb_O emb_V", W_O, W_V)
    
    return einsum(
        "voc1 emb1, emb1 emb2, emb2 emb3, voc2 emb3 -> voc1 voc2",
        W_E, QK_circuit, OV_circuit, W_E
    )

if MAIN:
    ind_head_indices = [4, 10]
    prev_token_head_index = 7
    K_comp_circuit = find_K_comp_full_full_circuit(prev_token_head_index, ind_head_indices)
    print("Fraction of tokens where the highest activating key is the same token", top_1_acc(K_comp_circuit.T).item())
    del K_comp_circuit





# FURTHER EXPLORATION

def frobenius_norm(tensor):
    """
    Implicitly allows batch dimensions
    """
    return tensor.pow(2).sum([-2, -1])


def get_q_comp_scores(W_QK, W_OV):
    """
    Returns a layer_1_index x layer_0_index tensor, where the i,j th entry is the Q-Composition score from head L0Hj to L1Hi
    """
    "SOLUTION"
    q_full = t.einsum("Imn,imM->IinM", W_QK, W_OV)
    comp_scores = frobenius_norm(q_full) / frobenius_norm(W_QK)[:, None] / frobenius_norm(W_OV)[None, :]
    return comp_scores


def get_k_comp_scores(W_QK, W_OV):
    """
    Returns a layer_1_index x layer_0_index tensor, where the i,j th entry is the K-Composition score from head L0Hj to L1Hi
    """
    "SOLUTION"
    k_full = t.einsum("Inm,imM->IinM", W_QK, W_OV)
    comp_scores = frobenius_norm(k_full) / frobenius_norm(W_QK)[:, None] / frobenius_norm(W_OV)[None, :]
    return comp_scores


def get_v_comp_scores(W_OV_1, W_OV_0):
    """
    Returns a layer_1_index x layer_0_index tensor, where the i,j th entry is the V-Composition score from head L0Hj to L1Hi
    """
    "SOLUTION"
    v_full = t.einsum("Inm,imM->IinM", W_OV_1, W_OV_0)
    comp_scores = frobenius_norm(v_full) / frobenius_norm(W_OV_1)[:, None] / frobenius_norm(W_OV_0)[None, :]
    return comp_scores


if MAIN:
    W_O = model.blocks[0].attn.W_O
    W_V = model.blocks[0].attn.W_V
    W_OV_0 = t.einsum("imh,ihM->imM", W_O, W_V)
    W_Q = model.blocks[1].attn.W_Q
    W_K = model.blocks[1].attn.W_K
    W_V = model.blocks[1].attn.W_V
    W_O = model.blocks[1].attn.W_O
    W_QK = t.einsum("ihm,ihM->imM", W_Q, W_K)
    W_OV_1 = t.einsum("imh,ihM->imM", W_O, W_V)

    q_comp_scores = get_q_comp_scores(W_QK, W_OV_0)
    k_comp_scores = get_k_comp_scores(W_QK, W_OV_0)
    v_comp_scores = get_v_comp_scores(W_OV_1, W_OV_0)
    px.imshow(
        to_numpy(q_comp_scores),
        y=[f"L1H{h}" for h in range(cfg["n_heads"])],
        x=[f"L0H{h}" for h in range(cfg["n_heads"])],
        labels={"x": "Layer 0", "y": "Layer 1"},
        title="Q Composition Scores",
        color_continuous_scale="Blues",
        zmin=0.0,
    ).show()
    px.imshow(
        to_numpy(k_comp_scores),
        y=[f"L1H{h}" for h in range(cfg["n_heads"])],
        x=[f"L0H{h}" for h in range(cfg["n_heads"])],
        labels={"x": "Layer 0", "y": "Layer 1"},
        title="K Composition Scores",
        color_continuous_scale="Blues",
        zmin=0.0,
    ).show()
    px.imshow(
        to_numpy(v_comp_scores),
        y=[f"L1H{h}" for h in range(cfg["n_heads"])],
        x=[f"L0H{h}" for h in range(cfg["n_heads"])],
        labels={"x": "Layer 0", "y": "Layer 1"},
        title="V Composition Scores",
        color_continuous_scale="Blues",
        zmin=0.0,
    ).show()

# %%

def generate_single_random_comp_score() -> float:
    """
    Write a function which generates a single composition score for random matrices
    """
    "SOLUTION"
    matrices = [t.empty((cfg["d_head"], cfg["d_model"])) for i in range(4)]
    for mat in matrices:
        nn.init.kaiming_uniform_(mat, a=np.sqrt(5))
    W1 = matrices[0].T @ matrices[1]
    W2 = matrices[2].T @ matrices[3]
    W3 = W1 @ W2
    return (frobenius_norm(W3) / frobenius_norm(W1) / frobenius_norm(W2)).item()


if MAIN:
    comp_scores_baseline = np.array([generate_single_random_comp_score() for i in range(200)])
    print("Mean:", comp_scores_baseline.mean())
    print("Std:", comp_scores_baseline.std())
    px.histogram(comp_scores_baseline, nbins=50).show()
# %%

if MAIN:
    px.imshow(
        to_numpy(q_comp_scores),
        y=[f"L1H{h}" for h in range(cfg["n_heads"])],
        x=[f"L0H{h}" for h in range(cfg["n_heads"])],
        labels={"x": "Layer 0", "y": "Layer 1"},
        title="Q Composition Scores",
        color_continuous_scale="RdBu",
        color_continuous_midpoint=comp_scores_baseline.mean(),
    ).show()
    px.imshow(
        to_numpy(k_comp_scores),
        y=[f"L1H{h}" for h in range(cfg["n_heads"])],
        x=[f"L0H{h}" for h in range(cfg["n_heads"])],
        labels={"x": "Layer 0", "y": "Layer 1"},
        title="K Composition Scores",
        color_continuous_scale="RdBu",
        color_continuous_midpoint=comp_scores_baseline.mean(),
    ).show()
    px.imshow(
        to_numpy(v_comp_scores),
        y=[f"L1H{h}" for h in range(cfg["n_heads"])],
        x=[f"L0H{h}" for h in range(cfg["n_heads"])],
        labels={"x": "Layer 0", "y": "Layer 1"},
        title="V Composition Scores",
        color_continuous_scale="RdBu",
        color_continuous_midpoint=comp_scores_baseline.mean(),
    ).show()

# %%

def stranded_svd(A: t.Tensor, B: t.Tensor) -> tuple[t.Tensor, t.Tensor, t.Tensor]:
    """
    Returns the SVD of AB in the torch format (ie (U, S, V^T))
    """
    "SOLUTION"
    UA, SA, VhA = t.svd(A)
    UB, SB, VhB = t.svd(B)
    intermed = SA.diag() @ VhA.T @ UB @ SB.diag()
    UC, SC, VhC = t.svd(intermed)
    return (UA @ UC), SC, (VhB @ VhC).T


def stranded_composition_score(W_A1: t.Tensor, W_A2: t.Tensor, W_B1: t.Tensor, W_B2: t.Tensor):
    """
    Returns the composition score for W_A = W_A1 @ W_A2 and W_B = W_B1 @ W_B2, with the entries in a low-rank factored form
    """
    "SOLUTION"
    UA, SA, VhA = stranded_svd(W_A1, W_A2)
    UB, SB, VhB = stranded_svd(W_B1, W_B2)
    normA = SA.pow(2).sum()
    normB = SB.pow(2).sum()
    intermed = SA.diag() @ VhA.T @ UB @ SB.diag()
    UC, SC, VhC = t.svd(intermed)
    return SC.pow(2).sum() / normA / normB


# %%

def ablation_induction_score(prev_head_index: int, ind_head_index: int) -> t.Tensor:
    """
    Takes as input the index of the L0 head and the index of the L1 head, and then runs with the previous token head ablated and returns the induction score for the ind_head_index now.
    """

    def ablation_hook(v, hook):
        v[:, :, prev_head_index] = 0.0
        return v

    def induction_pattern_hook(attn, hook):
        hook.ctx[prev_head_index] = attn[0, ind_head_index].diag(-(seq_len - 1)).mean()

    model.run_with_hooks(
        rep_tokens,
        fwd_hooks=[
            ("blocks.0.attn.hook_v", ablation_hook),
            ("blocks.1.attn.hook_attn", induction_pattern_hook),
        ],
    )
    return model.blocks[1].attn.hook_attn.ctx[prev_head_index]


if MAIN:
    for i in range(cfg["n_heads"]):
        print(f"Ablation effect of head {i}:", ablation_induction_score(i, 4).item())


# %%

if MAIN:

    text = "Hello world"
    input_tokens = model.to_tokens(text)

    head_index = 5
    layer = 4

    def ablation_hook(value, hook):
        # value is the value activation for this attention layer
        # It has shape [batch, position, head_index, d_head]
        value[:, :, head_index, :] = 0.0
        return value

    logits = model.run_with_hooks(input_tokens, fwd_hooks=[(f"blocks.{layer}.attn.hook_v", ablation_hook)])
