# %%

import torch as t
from torch.utils.data import Dataset

# %%
# ============================== Creating dataset for the brackets task ==============================

# This involves the following:
#   - an algorithm to generate balanced and unbalanced bracket strings
#       - generate unbalanced strings by just randomly generating brackets and checking if they're balanced
#           - lines 15 - 50 check if a string is balanced
#       - generate balanced strings by the method of S -> (S)|SS (each with equal probability)
#           - the reason we can't generate these randomly too is that a random long bracket string is almost certainly unbalanced
#           - lines 55 - 133 deal with generating balanced / unbalanced strings, and adding [CLS] and [PAD] tokens
#           - lines 138 - 151 actually create a balanced bracket dataset (and have an "inspect" function to check it's working)


def is_balanced(bracket_arr: list):
    """
    Takes a string of tokens representing brackets (0 = left, 1 = right, >1 = other tokens) and checks if it's balanced

    Uses the altitude method:
        left brackets are converted into +1, right brackets into -1
        altitute is the cumulative sum
        bracket is balanced iff altitude is never negative, AND altitude is zero at the end

        examples:
            "(())"  ==> altitude is [1, 2, 1, 0]     ==> balanced
            "()())" ==> altitude is [1, 0, 1, 0, -1] ==> unbalanced
    """

    bracket_arr_altitude = t.full_like(bracket_arr, 1)
    bracket_arr_altitude[bracket_arr == 1] = -1
    bracket_arr_altitude = bracket_arr_altitude.cumsum(-1)

    is_balanced = (bracket_arr_altitude.min() >= 0) and (bracket_arr_altitude[-1] == 0)

    return is_balanced

def bracket_string_to_arr(bracket_string):
    assert set(bracket_string) <= set("()")
    bracket_arr = t.tensor([0 if s=="(" else 1 for s in bracket_string])
    return bracket_arr

def bracket_arr_to_string(bracket_arr):
    bracket_arr_ = bracket_arr[bracket_arr <= 1]
    if len(bracket_arr_) == 0:
        return ""
    else:
        bracket_string = "".join(["(" if n == 0 else ")" for n in bracket_arr_])
        return bracket_string

testcases = [
    ("()()", True),
    ("(())", True),
    ("(", False),
    (")", False),
    ("())(", False)
]
for bracket_string, expected_is_balanced in testcases:
    actual_is_balanced = is_balanced(bracket_string_to_arr(bracket_string))
    assert actual_is_balanced == expected_is_balanced, f"Failed on case {repr(bracket_string)}"

# %%

def randomly_evolve_bracket_string(s, n_brackets_added):
    """
    If `s` is something like "SS(S)((S))", this will randomly replace one of the S's with either SS or (S).
    """
    S_positions = [i for i, char in enumerate(s) if char == "S"]
    S_position = S_positions[t.randint(low=0, high=len(S_positions), size=(1,))]
    if t.rand(1) < 0.7:
        return s[:S_position] + "SS" + s[S_position+1:], n_brackets_added
    else:
        return s[:S_position] + "(S)" + s[S_position+1:], n_brackets_added + 1
    

def generate_balanced_brackets(max_length, length_distribution, return_type="tensor"):
    """
    Generates balanced bracket string of length between 0 and max_length

    Does this by randomly applying max_length//2 operations of the form S -> (S)|SS, then removing all S occurrences
    """
    assert max_length % 2 == 0
    half_length = length_distribution.sample().item()
    n_brackets_added = 0
    s = "S"
    while n_brackets_added < half_length:
        s, n_brackets_added = randomly_evolve_bracket_string(s, n_brackets_added)
    s = s.replace("S", "")
    if return_type == "tensor":
        return t.tensor([0 if char=="(" else 1 for char in s])
    elif return_type == "str":
        return s

def generate_unbalanced_brackets(max_length, length_distribution):
    assert max_length % 2 == 0
    length = length_distribution.sample() * 2
    while True:
        bracket_array = t.randint(low=0, high=2, size=(length,))
        if not is_balanced(bracket_array):
            return bracket_array

def pad_bracket_array(bracket_array: t.Tensor, final_length: t.Tensor) -> tuple[t.Tensor, t.Tensor]:
    """
    Takes an array like [0, 0, 1, 1] representing brackets, and turns it into [2, 0, 0, 1, 1, 3, 3, ...]

    (i.e. adding a classifier and padding tokens)
    """
    cls_token = t.tensor([2])
    end_token = t.tensor([3])
    n_pad_tokens = final_length - 2 - bracket_array.size(0)
    pad_tokens = 4 * t.ones(n_pad_tokens)

    padded_bracket_array = t.concat([cls_token, bracket_array, end_token, pad_tokens])

    one_zero_attention_mask = t.full_like(padded_bracket_array, 0)
    one_zero_attention_mask[:-n_pad_tokens] = 1
    
    return padded_bracket_array.to(t.long), one_zero_attention_mask

def generate_bracket_array(max_length, length_distribution, p_balanced):
    balanced_label = t.rand(1).item() < p_balanced
    if balanced_label:
        bracket_array = generate_balanced_brackets(max_length, length_distribution)
    else:
        bracket_array = generate_unbalanced_brackets(max_length, length_distribution)

    padded_bracket_array, one_zero_attention_mask = pad_bracket_array(bracket_array, final_length=max_length+1)

    return padded_bracket_array, one_zero_attention_mask, balanced_label

# %%

class BalancedBracketsDataset(Dataset):
    def __init__(self, size, max_length, length_distribution_function, p_balanced):

        self.max_length = max_length
        self.length_probs = length_distribution_function(t.arange(1, (max_length//2) + 1)).to(t.float)
        self.length_probs /= self.length_probs.sum()
        self.length_distribution = t.distributions.categorical.Categorical(probs=self.length_probs)
        self.p_balanced= p_balanced
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return generate_bracket_array(self.max_length, self.length_distribution, self.p_balanced)

trainset = BalancedBracketsDataset(
    size = 100,
    max_length = 10,
    length_distribution_function = lambda x: x-1,
    p_balanced = 0.5
)


# %%

def inspect_balanced_brackets_dataset(dataset):
    print(f"Length = {len(dataset)}")
    for n in range(10):
        padded_bracket_array, one_zero_attention_mask, balanced_label = dataset[t.randint(high=len(dataset), size=(1,)).item()]
        bracket_string = bracket_arr_to_string(padded_bracket_array)
        print(f"{repr(bracket_string):10}  |  {int(balanced_label)}  |  {padded_bracket_array}")

if __name__ == "__main__":
    inspect_balanced_brackets_dataset(trainset)

