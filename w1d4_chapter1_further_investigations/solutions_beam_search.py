import torch as t
import transformers
from typing import List, Tuple

MAIN = __name__ == "__main__"

device = t.device("cuda" if t.cuda.is_available() else "cpu")

if MAIN:
    assert str(device) == "cuda"
    tokenizer = transformers.AutoTokenizer.from_pretrained("gpt2")
    gpt = transformers.AutoModelForCausalLM.from_pretrained("gpt2").to(device).train()

@t.inference_mode()
def beam_search(
    model, input_ids: t.Tensor, num_return_sequences: int, num_beams: int, max_new_tokens: int, tokenizer, verbose=False
) -> List[Tuple[float, t.Tensor]]:
    """
    input_ids: (seq, ) - the prompt

    max_new_tokens: stop after this many new tokens are generated, even if no EOS is generated. In this case, the best incomplete sequences should also be returned.
    verbose: if True, print the current (unfinished) completions after each iteration for debugging purposes

    Return list of length num_return_sequences. Each element is a tuple of (logprob, tokens) where the tokens include both prompt and completion, sorted by descending logprob.
    """
    assert num_return_sequences <= num_beams

    model.eval()
    
    # Create list to store the sequences to return
    # We only add to this when we generate an EOS token, or at the very end
    final_logitsums_and_completions = []

    # Create list to store the current best completions and their logit scores
    best_logitsums_and_completions = [(0, input_ids.tolist())]

    for n in range(max_new_tokens):
        
        # Create a list to store the completions at this stage
        new_best_logitsums_and_completions = []

        # This section loops through all completions so far, and get the next words
        for (logitsum, completion) in best_logitsums_and_completions:

            # Get output (we only care about the vector of logits for the next token)
            output = model(t.tensor(completion).unsqueeze(0).to(device, t.long))
            output = (output if isinstance(output, t.Tensor) else output.logits)[0, -1, :].log_softmax(-1)

            # Find the top `num_beams` (because this is the maximum we might need)
            topk_logits, topk_indices = t.topk(output, k=num_beams)

            # Append to the new best completions list
            for logit, idx in zip(topk_logits, topk_indices):
                new_completion_and_logit = (logitsum + logit.item(), completion + [idx.item(),])
                new_best_logitsums_and_completions.append(new_completion_and_logit)

        # This section updates (and sorts) the list of best completions, and also updates `final_logitsums_and_completions` if EOS was produced
        best_logitsums_and_completions = []
        for (logitsum, completion) in sorted(new_best_logitsums_and_completions, key=lambda x: x[0], reverse=True):
            
            # If token is eos then add it to final_logitsums_and_completions
            if completion[-1] == getattr(tokenizer, "eos_token_id", None):
                final_logitsums_and_completions.append((logitsum, completion))
            
            # Else add it to best_logitsums_and_completions until that list is full up, then we break out of for loop
            else:
                best_logitsums_and_completions.append((logitsum, completion))
                if len(best_logitsums_and_completions) == num_beams:
                    break

        # Add `best_logitsums_and_completions` to our final list, if necessary
        # Also sort the final completions list, and print output if necessary
        if n == max_new_tokens - 1:
            final_logitsums_and_completions.extend(best_logitsums_and_completions)
            final_logitsums_and_completions = sort_by_logits_and_crop(final_logitsums_and_completions, max_size=num_return_sequences)
            if verbose: print_sequences(f"Returning best {num_return_sequences=} completions:", final_logitsums_and_completions, tokenizer)
        else:
            final_logitsums_and_completions = sort_by_logits_and_crop(final_logitsums_and_completions, max_size=num_beams)
            if verbose: print_sequences(f"Printing {num_beams=} best completions:", best_logitsums_and_completions, tokenizer)


    return final_logitsums_and_completions

def print_sequences(name, logitsums_and_completions, tokenizer):
    if len(logitsums_and_completions) == 0:
        return
    print("\n" + name + "\n")
    print("logitsum | completion")
    for logit_sum, completion in logitsums_and_completions:
        text = tokenizer.decode(completion)
        print(f"{logit_sum:>8.3f} | {text}")

def sort_by_logits_and_crop(logitsums_and_completions, max_size):
    logitsums_and_completions = sorted(logitsums_and_completions, key=lambda x: x[0], reverse=True)
    logitsums_and_completions = logitsums_and_completions[:min(max_size, len(logitsums_and_completions))]
    return logitsums_and_completions
