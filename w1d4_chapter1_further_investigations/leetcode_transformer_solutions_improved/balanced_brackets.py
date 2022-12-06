# %%

import torch as t
import torch.nn.functional as F
from torch.utils.data import Dataset
from typing import Optional, Tuple, Union, List
from einops import repeat, rearrange

device = t.device("cuda" if t.cuda.is_available() else "cpu")

MAIN = __name__ == "__main__"

# %%
# ============================== Creating dataset for the brackets task ==============================

def bracket_tensor_is_balanced(bracket_tensor: t.Tensor) -> bool:
    """Checks if a tensor of brackets is balanced"""

    bracket_tensor = t.where(bracket_tensor <= 2, 0, bracket_tensor)
    bracket_tensor[bracket_tensor == 3] = 1
    bracket_tensor[bracket_tensor == 4] = -1
    final_altitude_is_zero = (bracket_tensor == 1).sum(1) == (bracket_tensor == -1).sum(1)
    altitude_is_non_negative = (bracket_tensor.cumsum(1) >= 0).all(1)

    return final_altitude_is_zero & altitude_is_non_negative


def generate_balanced_bracket_strings(num_strings: int, length: int, padded_length: int) -> t.Tensor:
    """
    Uses the altitude method to generate a random bracket string of a certain length.

    Altitude method:
        take N left and N right brackets, and randomly permute them in some order
        convert left brackets to +1, and right brackets to -1
        calculate the altitude (cumulative sum), and keep flipping until altitude is non-neg everywhere

    Returns:
        tensor with 0=start, 1=pad, 2=end, 3=left, 4=right
    """
    # We have start and end tokens, so we need at least 2 space in between
    assert padded_length >= length
    
    arr = t.randperm(length * num_strings).reshape(length, num_strings)
    arr = t.where(arr <= arr.median(dim=0).values, 1, -1).T

    num_strings_idx = repeat(t.arange(num_strings), "n -> n l", l=length)

    for n in range(length):
        arr_altitude = arr.cumsum(dim=1)
        mask = (num_strings_idx >= n) & (arr_altitude[:, n] < 0).unsqueeze(1)
        arr = t.where(mask, -arr, arr)

    return t.concat([
        t.full((num_strings, 1), fill_value=0, dtype=t.int),
        t.where(arr == 1, 3, 4),
        t.full((num_strings, 1), fill_value=2, dtype=t.int),
        t.full((num_strings, padded_length - length), fill_value=1, dtype=t.int),
    ], dim=1)


def generate_balanced_bracket_strings_variable_length(num_strings_per_length: int, max_length: int) -> t.Tensor:
    """
    Same as function above, but can generate strings of variable length.
    
    Each possible even length has the same probability of being generated.

    0 is left, 1 is right, 2 is padding token.
    """
    # Generate strings of each length, and concatenate them
    return t.concat([
        generate_balanced_bracket_strings(num_strings_per_length, length, max_length)
        for length in range(2, max_length, 2)
    ], dim=0)

def generate_unbalanced_bracket_strings_variable_length(num_strings: int, max_length: int) -> t.Tensor:

    # Initialise array with random brackets
    arr = t.randint(low=0, high=2, size=(num_strings, max_length+1))
    arr = t.where(arr == 0, 3, 4)
    random_lengths = 2 * t.randint(1, max_length // 2, (num_strings,))

    # Set end token and pad tokens
    for n, l in enumerate(random_lengths):
        arr[n, l] = 2
        arr[n, l+1:] = 1

    # Pad with start tokens
    return t.concat([t.full((num_strings, 1), fill_value=0, dtype=t.int), arr], dim=1)


class SimpleTokenizer:
    # Define special tokens
    START_TOKEN = 0
    PAD_TOKEN = 1
    END_TOKEN = 2

    base_d = {"[start]": START_TOKEN, "[pad]": PAD_TOKEN, "[end]": END_TOKEN}

    def __init__(self, alphabet: str):
        self.alphabet = alphabet
        # The 3 is because there are 3 special tokens (defined just above)
        self.t_to_i = {**{c: i + 3 for i, c in enumerate(alphabet)}, **self.base_d}
        self.i_to_t = {i: c for c, i in self.t_to_i.items()}

    # Function to encode a string (or strings) into a tensor (raises ValueError if it encounters an unknown character)
    def tokenize(self, strs: Union[str, List[str]], max_len: Optional[int] = None) -> t.Tensor:
        def c_to_int(c: str) -> int:
            if c in self.t_to_i:
                return self.t_to_i[c]
            else:
                raise ValueError(c)

        if max_len is None:
            max_len = max((max(len(s) for s in strs), 1))

        if isinstance(strs, str):
            strs = [strs]
        ints = [
            [self.START_TOKEN] + [c_to_int(c) for c in s] + [self.END_TOKEN] + [self.PAD_TOKEN] * (max_len - len(s))
            for s in strs
        ]
        tokenized = t.tensor(ints)
        one_zero_attention_mask = (tokenized != self.PAD_TOKEN).to(dtype=t.float)
        return tokenized, one_zero_attention_mask

    # Function to decode a list of token indices into a string (raises ValueError if it encounters an unknown token index)
    def decode(self, tokens: t.Tensor) -> Union[str, List[str]]:
        def int_to_c(c: int) -> str:
            if c < len(self.i_to_t):
                return self.i_to_t[c]
            else:
                raise ValueError(c)

        single_token = (tokens.ndim == 1)
        if single_token:
            tokens = tokens.unsqueeze(0)
        decoded = [
            "".join(int_to_c(i.item()) for i in seq[1:] if i != self.PAD_TOKEN and i != self.END_TOKEN)
            for seq in tokens
        ]
        return decoded[0] if single_token else decoded

class BracketsDataset(Dataset):
    """A dataset containing sequences, is_balanced labels, and tokenized sequences"""

    def __init__(self, tokenizer: SimpleTokenizer, size: int, max_length: int, fraction_balanced: int):
        """
        bracket_tensors are lists of outputs from `generate_bracket_string` function
        """
        self.max_length = max_length
        self.size = size

        num_balanced = int(size * fraction_balanced)
        num_balanced_per_length = num_balanced // (max_length // 2 - 1)
        num_balanced = num_balanced_per_length * (max_length // 2 - 1)
        num_unbalanced = size - num_balanced
        
        balanced_brackets = generate_balanced_bracket_strings_variable_length(num_balanced_per_length, max_length)
        random_brackets = generate_unbalanced_bracket_strings_variable_length(num_unbalanced, max_length)

        self.brackets = t.cat([balanced_brackets, random_brackets], dim=0)
        self.attn_mask = (self.brackets != tokenizer.PAD_TOKEN).to(dtype=t.float)
        self.bracket_strings = tokenizer.decode(self.brackets)
        self.isbalanced = bracket_tensor_is_balanced(self.brackets)

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx) -> Tuple[t.Tensor, t.Tensor, str, t.Tensor]:
        return self.brackets[idx], self.attn_mask[idx], self.bracket_strings[idx], self.isbalanced[idx]

    def random_sample(self) -> Tuple[t.Tensor, t.Tensor, str, t.Tensor]:
        return self.__getitem__(t.randint(0, self.size, (1,)).item())

# %%

if MAIN:
    max_length=10
    tokenizer = SimpleTokenizer("()")
    trainset = BracketsDataset(tokenizer, size=10_000, max_length=max_length, fraction_balanced=0.5)


    for i in range(10):
        brackets, attn_mask, bracket_str, is_balanced = trainset.random_sample()
        print(f"{brackets},  {bracket_str:{max_length}}, {is_balanced}")
        assert is_balanced.item() == bracket_tensor_is_balanced(brackets.unsqueeze(0)).item()
        assert tokenizer.decode(brackets) == bracket_str
        t.testing.assert_close(tokenizer.tokenize(bracket_str, max_len=max_length)[0].squeeze(), brackets)

# %%

