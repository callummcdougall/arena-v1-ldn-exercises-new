# %%
import functools
import json
import os
from typing import Any, List, Tuple, Union

os.chdir(r"C:\Users\calsm\Documents\AI Alignment\ARENA\arena-v1-ldn-exercises-restructured\w5_chapter5_interpretability")

import einops
import matplotlib.pyplot as plt
import torch
import torch as t
import torch.nn.functional as F
from fancy_einsum import einsum
from sklearn.linear_model import LinearRegression
from torch import nn

import tests
from w5d1_transformer import MultiheadAttention, ParenTransformer, SimpleTokenizer

# %%

MAIN = __name__ == "__main__"
DEVICE = t.device("cpu")

# %%

if MAIN:
    model = ParenTransformer(ntoken=5, nclasses=2, d_model=56, nhead=2, d_hid=56, nlayers=3).to(DEVICE)
    state_dict = t.load("w2d5_state_dict.pt")
    model.to(DEVICE)
    model.load_simple_transformer_state_dict(state_dict)
    model.eval()
    tokenizer = SimpleTokenizer("()")
    with open("w2d5_data.json") as f:
        data_tuples: List[Tuple[str, bool]] = json.load(f)
        print(f"loaded {len(data_tuples)} examples")
    assert isinstance(data_tuples, list)


class DataSet:
    """A dataset containing sequences, is_balanced labels, and tokenized sequences"""

    def __init__(self, data_tuples: list):
        """
        data_tuples is List[Tuple[str, bool]] signifying sequence and label
        """
        self.strs = [x[0] for x in data_tuples]
        self.isbal = t.tensor([x[1] for x in data_tuples]).to(device=DEVICE, dtype=t.bool)
        self.toks = tokenizer.tokenize(self.strs).to(DEVICE)

        self.open_proportion = t.tensor([s.count("(") / len(s) for s in self.strs])
        self.starts_open = t.tensor([s[0] == "(" for s in self.strs]).bool()

    def __len__(self) -> int:
        return len(self.strs)

    def __getitem__(self, idx) -> Union["DataSet", Tuple[str, t.Tensor, t.Tensor]]:
        if type(idx) == slice:
            return self.__class__(list(zip(self.strs[idx], self.isbal[idx])))
        return self.strs[idx], self.isbal[idx], self.toks[idx]

    @property
    def seq_length(self) -> int:
        return self.toks.size(-1)

    @classmethod
    def with_length(cls, data_tuples: List[Tuple[str, bool]], selected_len: int) -> "DataSet":
        return cls([(s, b) for (s, b) in data_tuples if len(s) == selected_len])

    @classmethod
    def with_start_char(cls, data_tuples: List[Tuple[str, bool]], start_char: str) -> "DataSet":
        return cls([(s, b) for (s, b) in data_tuples if s[0] == start_char])


if MAIN:
    N_SAMPLES = 5000
    data_tuples = data_tuples[:N_SAMPLES]
    data = DataSet(data_tuples)

    if "SOLUTION":
        fig, ax = plt.subplots()
        ax.hist([len(x) for x in data.strs], bins=list(range(43)))
        ax.set(xlabel="Length", ylabel="Count")


# %%
def is_balanced_forloop(parens: str) -> bool:
    """Return True if the parens are balanced.
    Parens is just the ( and ) characters, no begin or end tokens.
    """
    "SOLUTION"
    i = 0
    for c in parens:
        if c == "(":
            i += 1
        elif c == ")":
            i -= 1
            if i < 0:
                return False
        else:
            raise ValueError(parens)
    return i == 0


if MAIN:
    examples = [
        "()",
        "))()()()()())()(())(()))(()(()(()(",
        "((()()()()))",
        "(()()()(()(())()",
        "()(()(((())())()))",
    ]
    labels = [True, False, True, False, True]

    for parens, expected in zip(examples, labels):
        actual = is_balanced_forloop(parens)
        assert expected == actual, f"{parens}: expected {expected} got {actual}"
    print("is_balanced_forloop ok!")

# %%
def is_balanced_vectorized(tokens: t.Tensor) -> bool:
    """
    tokens: sequence of tokens including begin, end and pad tokens - recall that 3 is '(' and 4 is ')'
    """
    "SOLUTION"
    table = t.tensor([0, 0, 0, 1, -1])
    change = table[tokens]
    cumsum = t.cumsum(change, -1)
    return (cumsum >= 0).all().item() and (cumsum[-1].item() == 0)  # type: ignore


if MAIN:
    for tokens, expected in zip(tokenizer.tokenize(examples), labels):
        actual = is_balanced_vectorized(tokens)
        assert expected == actual, f"{tokens}: expected {expected} got {actual}"
    print("is_balanced_vectorized ok!")


# TBD lowpri: implement the model's solution manually to make sure they understand the suffixes and right to left thing
# TBD lowpri: make plot of elevation


# %%


if MAIN:
    toks = tokenizer.tokenize(examples).to(DEVICE)
    out = model(toks)
    print("Model confidence: ", [f"{p:.4%}" for p in out.exp()[:, 1]])


def run_model_on_data(model: ParenTransformer, data: DataSet, batch_size: int = 200) -> t.Tensor:
    """Return probability that each example is balanced"""
    ln_probs = []
    for i in range(0, len(data.strs), batch_size):
        toks = data.toks[i : i + batch_size]
        # note model outputs in a weird shape, [seqlen, batch, 2 (unbal, bal)]
        with t.no_grad():
            out = model(toks)
        ln_probs.append(out)
    out = t.cat(ln_probs).exp()
    assert out.shape == (len(data), 2)
    return out


if MAIN:
    test_set = data
    n_correct = t.sum((run_model_on_data(model, test_set).argmax(-1) == test_set.isbal).float())
    print(f"Model got {n_correct} out of {len(data)} training examples correct!")

# %%

# CM: I felt confused here and thought it was asking for something more complicated - maybe docstring here?


def get_post_final_ln_dir(model: ParenTransformer) -> t.Tensor:
    return (model.decoder.weight[0] - model.decoder.weight[1]).cpu()


def get_inputs(model: ParenTransformer, data: DataSet, module: nn.Module) -> t.Tensor:
    """
    Get the inputs to a particular submodule of the model when run on the data.
    Returns a tensor of size (data_pts, seq_pos, emb_size).
    """
    acts = []
    fn = lambda m, i, o: acts.append(i[0].detach().clone())
    h = module.register_forward_hook(fn)
    run_model_on_data(model, data)
    h.remove()
    out = t.concat(acts, dim=0)
    assert out.shape == (len(data), data.seq_length, model.d_model)
    return out.clone()


def get_outputs(model: ParenTransformer, data: DataSet, module: nn.Module) -> t.Tensor:
    """
    Get the outputs from a particular submodule of the model when run on the data.
    Returns a tensor of size (data_pts, seq_pos, emb_size).
    """
    acts = []
    fn = lambda m, i, o: acts.append(o.detach().clone())
    h = module.register_forward_hook(fn)
    run_model_on_data(model, data)
    h.remove()
    out = t.concat(acts, dim=0)
    assert out.shape == (len(data), data.seq_length, model.d_model)
    return out.clone()

if MAIN:
    tests.test_get_inputs(get_inputs, model, data)
    tests.test_get_outputs(get_outputs, model, data)

# %%

def get_ln_fit(
    model: ParenTransformer,
    data: DataSet,
    ln_module: nn.LayerNorm,
    seq_pos: Union[None, int],
) -> Tuple[LinearRegression, t.Tensor]:
    """
    if seq_pos is None, find best fit for all sequence positions. Otherwise, fit only for given seq_pos.
    Returns: A tuple of a (fitted) sklearn LinearRegression object and a dimensionless tensor containing the r^2 of the fit (hint: wrap a value in torch.tensor() to make a dimensionless tensor)
    """
    "SOLUTION"

    inputs = get_inputs(model, data, ln_module)
    outputs = get_outputs(model, data, ln_module)

    if seq_pos is None:
        inputs = inputs.reshape(-1, 56)
        outputs = outputs.reshape(-1, 56)
    else:
        inputs = inputs[:, seq_pos, :]
        outputs = outputs[:, seq_pos, :]

    fitted = LinearRegression().fit(inputs, outputs)
    return fitted, t.tensor(fitted.score(inputs, outputs))


if MAIN:
    final_ln_fit, r2 = get_ln_fit(model, data, model.norm, seq_pos=0)  # type: ignore
    print("r^2: ", r2)
    tests.test_final_ln_fit(model, data, get_ln_fit)

# %%
def get_pre_final_ln_dir(model: ParenTransformer, data: DataSet) -> t.Tensor:
    """SOLUTION"""
    final_ln_fit, _ = get_ln_fit(model, data, model.norm, seq_pos=0)  # type: ignore
    unbalanced_d_post_ln = get_post_final_ln_dir(model)
    unbalanced_d_pre_final_ln = t.tensor(final_ln_fit.coef_.T) @ unbalanced_d_post_ln
    return unbalanced_d_pre_final_ln


if MAIN:
    tests.test_pre_final_ln_dir(model, data, get_pre_final_ln_dir)


# %%
def get_out_by_head(model: ParenTransformer, data: DataSet, layer: int) -> t.Tensor:
    """
    Get the output of the heads in a particular layer when the model is run on the data.
    Returns a tensor of shape (batch, num_heads, seq, embed_width)
    """

    "SOLUTION"
    out_proj: nn.Linear = model.layers[layer].self_attn.project_output
    combined_values = get_inputs(model, data, out_proj).to(DEVICE)
    num_heads = model.nhead
    # pytorch stores weight matricies in shape [out_features, in_features]
    o_mats_by_head = einops.rearrange(out_proj.weight, " out (head head_size) -> out head head_size", head=num_heads)
    head_value = einops.rearrange(combined_values, "b seq (head head_size)-> b seq head head_size", head=num_heads)
    out_by_head = einsum(
        "out head head_size, batch seq head head_size -> batch head seq out",
        o_mats_by_head,
        head_value,
    )
    assert out_by_head.shape == (len(data), num_heads, data.seq_length, model.d_model)
    return out_by_head.clone()


if MAIN:
    tests.test_get_out_by_head(get_out_by_head, model, data)

# %%

def get_out_by_components(model: ParenTransformer, data: DataSet) -> t.Tensor:
    """
    Computes a tensor of shape [10, dataset_size, seq_pos, emb] representing the output of the model's components when run on the data.
    The first dimension is  [embeddings, head 0.0, head 0.1, mlp 0, head 1.0, head 1.1, mlp 1, head 2.0, head 2.1, mlp 2]
    """
    "SOLUTION"
    components = []
    components.append(get_outputs(model, data, model.pos_encoder))
    for l in (0, 1, 2):
        by_head = get_out_by_head(model, data, l)
        mlp = get_outputs(model, data, model.layers[l].linear2)
        components.extend([by_head[:, 0], by_head[:, 1], mlp])
    out = t.stack(components)
    assert out.shape == (10, len(data), data.seq_length, model.d_model)
    return out.clone()


if MAIN:
    tests.test_get_out_by_component(get_out_by_components, model, data)

# %%
"""
Now, confirm that input to the final layer norm is the sum of the output of each component and the output projection biases.
"""

# %%
if MAIN:
    biases = sum([model.layers[l].self_attn.project_output.bias for l in (0, 1, 2)]).clone()  # type: ignore
    if "SOLUTION":
        out_by_components = get_out_by_components(model, data)
        summed_terms = out_by_components.sum(0) + biases
        pre_final_ln = get_inputs(model, data, model.norm)
    else:
        out_by_components = "YOUR CODE"
        summed_terms = "YOUR CODE"
        pre_final_ln = "YOUR CODE"
    t.testing.assert_close(summed_terms, pre_final_ln, atol=1e-4, rtol=0)


# %%
"""
### Which heads write in this direction? On what sorts of inputs?
To figure out which components are directly important for the the model's output being "unbalanced", we can see which components tend to output a vector to the position-0 residual stream with higher dot product in the unbalanced direction for actually unbalanced inputs.
Compute a (10, N_SAMPLES) tensor containing, for each sample, for each component, the dot product of the component's output with the unbalanced direction on this sample. Then normalize it by subtracting the mean of the dot product of this component's output with the unbalanced direction on balanced samples. This gives us a metric of how much, for each sample, this component was contributing to the unbalanced direction more than it usually does on a balanced sequence. You can use the data.isbal tensor to help with this (see DataSet definition). Then use our pre-provided graphing method to see which components are important.
"""
# %%
def hists_per_comp(magnitudes, isbal):
    """
    magnitudes: a (10, N_SAMPLES) tensor containing, for each sample, a scalar for each component of the model (e.g. head 2.0).
    isbal: a (N_SAMPLES,) boolean tensor indicating which samples are balanced.
    Makes 10 histograms, each of which displays a histogram of some component's values for the balanced and unbalanced sequences.
    """

    num_comps = magnitudes.shape[0]
    titles = [
        "embeddings",
        "head 0.0",
        "head 0.1",
        "mlp 0",
        "head 1.0",
        "head 1.1",
        "mlp 1",
        "head 2.0",
        "head 2.1",
        "mlp 2",
    ]
    assert num_comps == len(titles)
    _, axs = plt.subplots(num_comps, 1, figsize=(6, 1 + 2 * num_comps), sharex=True)
    for i, title in enumerate(titles):
        ax: plt.Axes = axs[i]  # type: ignore
        ax.hist(magnitudes[i, isbal].numpy(), alpha=0.5, bins=75, range=(-10, 20), label="bal")  # type: ignore
        ax.hist(magnitudes[i, ~data.isbal].numpy(), alpha=0.5, bins=75, range=(-10, 20), label="unbal")  # type: ignore
        ax.title.set_text(title)
        ax.legend(loc="upper right")
    plt.show()


if MAIN:
    if "SOLUTION":
        in_d = t.inner(out_by_components[:, :, 0, :], get_pre_final_ln_dir(model, data).clone()).detach()
        normed = in_d - in_d[:, data.isbal].mean(-1, keepdim=True)
        hists_per_comp(normed, data.isbal)
"""
<details>
<summary>Which heads are important, and how we'll procede with this analysis (read after making the graphs)</summary>
If all went well with the graphing, you'll see that heads 2.0, and 2.1 stand out.
</details>
"""

# %%
"""
### Head influence by type of failures
Those histograms showed us which heads were important, but it doesn't tell us what these heads are doing, however. In order to get some indication of that, let's focus in on the two heads in layer 2 and see how much they write in our chosen direction on different types of inputs. In particular, we can classify inputs by if they pass the 'overall elevation' and 'nowhere negative' tests.
We'll also ignore sentences that start with a close paren, as the behaviour is somewhat different on them.
Define, so that the graphing works:
* negative_failure: a (N_SAMPLES,) boolean vector that is true for sequences whose elevation (when reading from right to left) ever dips negative, i.e. there's an open paren that is never closed.
* total_elevation_failure: a (N_SAMPLES,) boolean vector that is true for sequences whose total elevation (i.e. the elevation at position 1, the first paren) is not exactly 0. In other words, for sentences with uneven numbers of open and close parens.
* h20_in_d: a (N_SAMPLES,) float vector equal to head 2.0's contribution to the position-0 residual stream in the unbalanced direction, normalized by subtracting its average unbalancedness contribution to this stream over _balanced sequences_.
* h21_in_d: same as above but head 2.1
"""

if MAIN:
    if "SOLUTION":
        count_open_after = t.flip(t.flip(data.toks == tokenizer.t_to_i["("], (1,)).cumsum(-1), (1,)).clone()
        count_close_after = t.flip(t.flip(data.toks == tokenizer.t_to_i[")"], (1,)).cumsum(-1), (1,)).clone()
        p_open_after = count_open_after / (count_open_after + count_close_after)
        max_open, _ = t.nan_to_num(p_open_after, nan=-1).max(1)
        negative_failure = max_open > 0.5
        total_elevation_failure = p_open_after[:, 0] != 0.5

        h20_in_d = in_d[7] - in_d[7, data.isbal].mean(0)
        h21_in_d = in_d[8] - in_d[8, data.isbal].mean(0)
    else:
        negative_failure = None
        total_elevation_failure = None
        h20_in_d = None
        h21_in_d = None

    plt.scatter(
        h20_in_d[data.starts_open & negative_failure & total_elevation_failure],
        h21_in_d[data.starts_open & negative_failure & total_elevation_failure],
        s=2,
        label="both failures",
    )
    plt.scatter(
        h20_in_d[data.starts_open & negative_failure & ~total_elevation_failure],
        h21_in_d[data.starts_open & negative_failure & ~total_elevation_failure],
        s=2,
        label="just neg failure",
    )
    plt.scatter(
        h20_in_d[data.starts_open & ~negative_failure & total_elevation_failure],
        h21_in_d[data.starts_open & ~negative_failure & total_elevation_failure],
        s=2,
        label="just total elevation failure",
    )
    plt.scatter(
        h20_in_d[data.starts_open & ~negative_failure & ~total_elevation_failure],
        h21_in_d[data.starts_open & ~negative_failure & ~total_elevation_failure],
        s=2,
        label="balanced",
    )
    plt.legend()
    plt.xlabel("Head 2.0 contribution")
    plt.ylabel("Head 2.1 contribution")
    plt.show()


# %%

"""
Look at the above graph and think about what the roles of the different heads are!
<details>
<summary>Nix's thoughts -- Read after thinking for yourself</summary>
The primary thing to take away is that 2.0 is responsible for checking the overall counts of open and close parentheses, and that 2.1 is responsible for making sure that the elevation never goes negative.
Aside: the actual story is a bit more complicated than that. Both heads will often pick up on failures that are not their responsibility, and output in the 'unbalanced' direction. This is in fact incentived by log-loss: the loss is slightly lower if both heads unanimously output 'unbalanced' on unbalanced sequences rather than if only the head 'responsible' for it does so. The heads in layer one do some logic that helps with this, although we'll not cover it today.
One way to think of it is that the heads specialized on being very reliable on their class of failures, and then sometimes will sucessfully pick up on the other type.
</details>
In most of the rest of these exercises, we'll focus on the overall elevation circuit as implemented by head 2.0. As an additional way to get intuition about what head 2.0 is doing, let's graph its output against the overall proportion of the sequence that is an open-paren.
"""

if MAIN:
    plt.scatter(data.open_proportion, h20_in_d, alpha=0.2, s=2)
    plt.xlabel("open-proportion of sequence")
    plt.ylabel("Amount 2.0 outputs in unbalanced direction")
    plt.show()

# %%

"""
Think about how this fits in with your understanding of what 2.0 is doing.
## Understanding the total elevation circuit
### Attention pattern of the responsible head
Which tokens is 2.0 paying attention to when the query is an open paren at token 0? Recall that we focus on sequences that start with an open paren because sequences that don't can be ruled out immediately, so more sophisticated behavior is unnecessary.
Write a function that extracts the attention patterns for a given head when run on a batch of inputs. Our code will show you the average attention pattern paid by the query for residual stream 0 when that position is an open paren.
Specifically:
* Use get_inputs from earlier, on the self-attention module in the layer in question.
* You can use the `attention_pattern_pre_softmax` function to get the pattern, then mask the padding (elements of the batch might be different lengths, and thus be suffixed with padding).
<details>
<summary> How do I find the padding?</summary>
`data.toks == tokenizer.PAD_TOKEN` will give you a boolean matrix of which positions in which batch elements are padding and which aren't.
</details>
"""

# %%
def get_attn_probs(model: ParenTransformer, tokenizer: SimpleTokenizer, data: DataSet, layer, head) -> t.Tensor:

    """
    Returns: (N_SAMPLES, max_seq_len, max_seq_len) tensor that sums to 1 over the last dimension.
    """

    """SOLUTION"""
    with t.no_grad():
        self_attn: BertSelfAttention = model.layers[layer].self_attn  # type: ignore
        # print(data.toks.device, model.layers[0].linear1.weight.device, self_attn.project_query.weight.device)
        attn_inputs = get_inputs(model, data, self_attn).to(DEVICE)
        attn_scores = self_attn.attention_pattern_pre_softmax(attn_inputs)
        additive_pad_mask = t.where(data.toks == tokenizer.PAD_TOKEN, -10000, 0)[:, None, None, :]
        attn_scores += additive_pad_mask
        attn_probs = attn_scores.softmax(dim=-1)
        return attn_probs[:, head, :, :].clone()


if MAIN:
    attn_probs = get_attn_probs(model, tokenizer, data, 2, 0)
    attn_probs_open = attn_probs[data.starts_open].mean(0)[[0]]
    plt.plot(attn_probs_open.squeeze().numpy())
    plt.xlabel("Key Position")
    plt.ylabel("Probability")
    plt.title("Avg Attention Probabilities for ( query from query 0")


# %%
"""
You should see an average attention of around 0.5 on position 1, and an average of about 0 for all other tokens. So 2.0 is just copying information from residual stream 1 to residual stream 0. In other words, 2.0 passes residual stream 1 through its W_OV circuit (after LayerNorming, of course), weighted by some amount which we'll pretend is constant. The plot thickens. Now we can ask, "What is the direction in residual stream 1 that, when passed through 2.0's W_OV, creates a vector in the unbalanced direction in residual stream 0?"
### Identifying meaningful direction before this head
We again need to propagate the direction back, this time through the OV matrix of 2.0 and a linear fit to the layernorm. This will require functions that can find these matrices for us.
"""


def get_WV(model: ParenTransformer, layer: int, head: int) -> t.Tensor:
    """
    Returns the W_V matrix of a head. Should be a CPU tensor of size (d_model / num_heads, d_model)
    """

    "SOLUTION"
    value_proj: nn.Linear = model.layers[layer].self_attn.project_value
    num_heads = model.nhead
    v_mats_by_head = einops.rearrange(value_proj.weight, "(head head_size) in -> head head_size in", head=num_heads)
    v_mat = v_mats_by_head[head]
    assert v_mat.shape == (model.d_model // model.nhead, model.d_model)
    return v_mat.clone()


def get_WO(model: ParenTransformer, layer: int, head: int) -> t.Tensor:
    """
    Returns the W_O matrix of a head. Should be a CPU tensor of size (d_model, d_model / num_heads)
    """

    "SOLUTION"
    out_proj: nn.Linear = model.layers[layer].self_attn.project_output
    num_heads = model.nhead
    o_mats_by_head = einops.rearrange(out_proj.weight, "out (head head_size) -> head out head_size", head=num_heads)
    o_mat = o_mats_by_head[head]
    assert o_mat.shape == (model.d_model, model.d_model // model.nhead)
    return o_mat.clone()


def get_WOV(model: ParenTransformer, layer: int, head: int) -> t.Tensor:
    return get_WO(model, layer, head) @ get_WV(model, layer, head)


def get_pre_20_dir(model, data):
    """
    Returns the direction propagated back through the OV matrix of 2.0 and then through the layernorm before the layer 2 attention heads.
    """
    "SOLUTION"
    d = get_pre_final_ln_dir(model, data)
    wLN, r2 = get_ln_fit(model, data, model.layers[2].norm1, seq_pos=1)
    wOV = get_WOV(model, 2, 0)
    print("r^2", r2)
    # print(wLN.coef_.shape, wOV.shape)
    return (t.tensor(wLN.coef_).T @ wOV.T) @ d


if MAIN:
    tests.test_get_WV(model, get_WV)
    tests.test_get_WO(model, get_WO)
    tests.test_get_pre_20_dir(model, data, get_pre_20_dir)

# %%

if MAIN:
    pre_20_d = get_pre_20_dir(model, data)

    in_d_20 = t.inner(out_by_components[:7, :, 1, :], pre_20_d).detach()

    titles = [
        "embeddings",
        "head 0.0",
        "head 0.1",
        "mlp 0",
        "head 1.0",
        "head 1.1",
        "mlp 1",
        "head 2.0",
        "head 2.1",
        "mlp 2",
    ]

    fig, axs = plt.subplots(7, 1, figsize=(6, 11), sharex=True)
    for i in range(7):
        ax: plt.Axes = axs[i]  # type: ignore
        normed = in_d_20[i] - in_d_20[i, data.isbal].mean(0)
        ax.hist(normed[data.starts_open & data.isbal.clone()].numpy(), alpha=0.5, bins=75, label="bal")  # type: ignore
        ax.hist(normed[data.starts_open & ~data.isbal.clone()].numpy(), alpha=0.5, bins=75, label="unbal")  # type: ignore
        ax.title.set_text(titles[i])

    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.show()


# %%

if MAIN:
    plt.scatter(data.open_proportion[data.starts_open], in_d_20[3, data.starts_open], s=2)
    plt.ylabel("amount mlp 0 writes in the unbalanced direction for head 2.0")
    plt.xlabel("open-proprtion of sequence")
    plt.show()

# %%

if MAIN:
    plt.scatter(data.open_proportion[data.starts_open], in_d_20[6, data.starts_open], s=2)
    plt.ylabel("amount mlp 1 writes in the unbalanced direction for head 2.0")
    plt.xlabel("open-proprtion of sequence")
    plt.show()

# %%

def out_by_neuron(model, data, layer):
    """
    Return shape: [len(data), seq_len, neurons, out]
    """
    "SOLUTION"
    lin2 = model.layers[layer].linear2
    neuron_acts = get_inputs(model, data, lin2)
    return einsum(
        "batch seq neuron, out neuron-> batch seq neuron out",
        neuron_acts,
        lin2.weight.cpu(),
    )


@functools.cache
def out_by_neuron_in_20_dir(model, data, layer):
    "SOLUTION"
    by_neuruon = out_by_neuron(model, data, layer)
    direction = get_pre_20_dir(model, data)
    return einsum("batch seq neuron out, out -> batch seq neuron", by_neuruon, direction)


# %%

def plot_neuron(model, data, layer, neuron_number):
    neurons_in_d = out_by_neuron_in_20_dir(model, data, layer)
    plt.scatter(
        data.open_proportion[data.starts_open],
        neurons_in_d[data.starts_open, 1, neuron_number].detach(),
        s=2,
    )
    plt.xlabel("open-proportion")
    plt.ylabel("output in 2.0 direction")
    plt.show()


if MAIN:
    if "SOLUTION":
        neuron_in_d_mlp0 = out_by_neuron_in_20_dir(model, data, 0)
        importance_mlp0 = neuron_in_d_mlp0[~data.isbal & data.starts_open, 1].mean(0) - neuron_in_d_mlp0[
            data.isbal, 1
        ].mean(0)

        neuron_in_d_mlp1 = out_by_neuron_in_20_dir(model, data, 1)
        importance_mlp1 = neuron_in_d_mlp1[~data.isbal & data.starts_open, 1].mean(0) - neuron_in_d_mlp1[
            data.isbal, 1
        ].mean(0)

        # most_important = torch.argmax(importance)
        print(torch.topk(importance_mlp0, k=20))
        # l0 - tensor([43, 33, 12, 10, 21,  3, 34, 39, 50, 42]))
        # l1 - tensor([10,  3, 53, 18, 31, 39,  9,  6, 19,  8]))

        print(torch.topk(importance_mlp1, k=20))
        plot_neuron(model, data, 1, 10)

# %%

def get_Q_and_K(model: ParenTransformer, layer: int, head: int) -> Tuple[t.Tensor, t.Tensor]:
    """
    Get the Q and K weight matrices for the attention head at the given indices.
    Return: Tuple of two tensors, both with shape (embedding_size, head_size)
    """
    "SOLUTION"
    q_proj: nn.Linear = model.layers[layer].self_attn.project_query
    k_proj: nn.Linear = model.layers[layer].self_attn.project_key
    num_heads = model.nhead
    q_mats_by_head = einops.rearrange(q_proj.weight, "(head head_size) out -> out head head_size", head=num_heads)
    k_mats_by_head = einops.rearrange(k_proj.weight, "(head head_size) out -> out head head_size", head=num_heads)
    q_mat = q_mats_by_head[:, head]
    assert q_mat.shape == (model.d_model, model.d_model // model.nhead)
    k_mat = k_mats_by_head[:, head]
    assert k_mat.shape == (model.d_model, model.d_model // model.nhead)
    return q_mat, k_mat


def qk_calc_termwise(
    model: ParenTransformer,
    layer: int,
    head: int,
    q_embedding: t.Tensor,
    k_embedding: t.Tensor,
) -> t.Tensor:
    """
    Get the pre-softmax attention scores that would be calculated by the given attention head from the given embeddings.
    q_embedding: tensor of shape (seq_len, embedding_size)
    k_embedding: tensor of shape (seq_len, embedding_size)
    Returns: tensor of shape (seq_len, seq_len)
    """
    "SOLUTION"
    q_mat, k_mat = get_Q_and_K(model, layer, head)
    qs = einsum("i o, x i -> x o", q_mat, q_embedding)
    ks = einsum("i o, y i -> y o", k_mat, k_embedding)
    scores = einsum("x o, y o -> x y", qs, ks)
    return scores.squeeze()


if MAIN:
    tests.qk_test(model, get_Q_and_K)
    tests.test_qk_calc_termwise(model, tokenizer, qk_calc_termwise)

# CM: is there a reason we run the model instead of just model.encoder.weight[tokenizer.t_to_i["("]]

def embedding(model: ParenTransformer, tokenizer: SimpleTokenizer, char: str) -> torch.Tensor:
    assert char in ("(", ")")
    "SOLUTION"
    input_id = tokenizer.t_to_i[char]
    input = t.tensor([input_id]).to(DEVICE)
    return model.encoder(input).clone()


if MAIN:
    open_emb = embedding(model, tokenizer, "(")
    closed_emb = embedding(model, tokenizer, ")")
    tests.embedding_test(model, tokenizer, embedding)

if MAIN:
    open_emb = embedding(model, tokenizer, "(")
    closed_emb = embedding(model, tokenizer, ")")

    pos_embeds = model.pos_encoder.pe
    open_emb_ln_per_seqpos = model.layers[0].norm1(open_emb.to(DEVICE) + pos_embeds[1:41])
    close_emb_ln_per_seqpos = model.layers[0].norm1(closed_emb.to(DEVICE) + pos_embeds[1:41])
    attn_score_open_open = qk_calc_termwise(model, 0, 0, open_emb_ln_per_seqpos, open_emb_ln_per_seqpos)
    attn_score_open_close = qk_calc_termwise(model, 0, 0, open_emb_ln_per_seqpos, close_emb_ln_per_seqpos)

    attn_score_open_avg = (attn_score_open_open + attn_score_open_close) / 2
    attn_prob_open = (attn_score_open_avg / (28**0.5)).softmax(-1).detach().clone().numpy()
    plt.matshow(attn_prob_open, cmap="magma")
    plt.xlabel("Key Position")
    plt.ylabel("Query Position")
    plt.title("Predicted Attention Probabilities for ( query")

    plt.gcf().set_size_inches(8, 6)
    plt.colorbar()
    plt.tight_layout()


#%%
def avg_attn_probs_0_0(
    model: ParenTransformer, data: DataSet, tokenizer: SimpleTokenizer, query_token: int
) -> t.Tensor:
    """
    Calculate the average attention probs for the 0.0 attention head for the provided data when the query is the given query token.
    Returns a tensor of shape (seq, seq)
    """
    "SOLUTION"
    attn_probs = get_attn_probs(model, tokenizer, data, 0, 0)
    assert attn_probs.shape == (len(data), data.seq_length, data.seq_length)
    is_open = data.toks == query_token
    assert is_open.shape == (len(data), data.seq_length)
    attn_probs_masked = t.where(is_open[:, :, None], attn_probs.double(), t.nan)
    out = t.nanmean(attn_probs_masked, dim=0)
    assert out.shape == (data.seq_length, data.seq_length)
    return out


if MAIN:
    data_len_40 = DataSet.with_length(data_tuples, 40)
    for paren in ("(", ")"):
        tok = tokenizer.t_to_i[paren]
        attn_probs_mean = avg_attn_probs_0_0(model, data_len_40, tokenizer, tok).detach().clone()
        plt.matshow(attn_probs_mean, cmap="magma")
        plt.ylabel("query position")
        plt.xlabel("key position")
        plt.title(f"with query = {paren}")
        plt.show()

if MAIN:
    tok = tokenizer.t_to_i["("]
    attn_probs_mean = avg_attn_probs_0_0(model, data_len_40, tokenizer, tok).detach().clone()
    plt.plot(range(42), attn_probs_mean[1])
    plt.ylim(0, None)
    plt.xlim(0, 42)
    plt.xlabel("Sequence position")
    plt.ylabel("Average attention")
    plt.show()


def embedding_V_0_0(model, emb_in: t.Tensor) -> t.Tensor:
    "SOLUTION"
    return emb_in @ get_WV(model, 0, 0).T.clone()


def embedding_OV_0_0(model, emb_in: t.Tensor) -> t.Tensor:
    "SOLUTION"
    return emb_in @ get_WOV(model, 0, 0).T.clone()


if MAIN:
    if "SOLUTION":
        data_start_open = DataSet.with_start_char(data_tuples, "(")
        attn0_ln_fit, r2 = get_ln_fit(model, data_start_open, model.layers[0].norm1, seq_pos=None)
        attn0_ln_fit = t.tensor(attn0_ln_fit.coef_)
        print("r^2: ", r2)
        open_v = embedding_OV_0_0(model, model.layers[0].norm1(open_emb))
        closed_v = embedding_OV_0_0(model, model.layers[0].norm1(closed_emb))
        print(torch.linalg.norm(open_v), torch.linalg.norm(closed_v))
        sim = F.cosine_similarity(open_v, closed_v, dim=1).item()
        print("Cosine Similarity of 0.0 OV", sim)



# %%

if MAIN:
    print("Update the examples list below below find adversarial examples")
    examples = ["()", "(()"]
    toks = tokenizer.tokenize(examples).to(DEVICE)
    out = model(toks)
    print("\n".join([f"{ex}: {p:.4%} balanced confidence" for ex, p in zip(examples, out.exp()[:, 1])]))

# %%
