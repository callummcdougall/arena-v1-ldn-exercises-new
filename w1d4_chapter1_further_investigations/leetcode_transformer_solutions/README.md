`balanced_brackets.py` just contains code to check if a bracket string is balanced, and generate balanced brackets. 

The most complicated part of this was how to generate balanced brackets. I based this on the fact that the set of balanced brackets can be represented by the context-free grammar with ruleset:

    S -> e | (S) | SS

`my_tranformer.py` defines all the internal modules of the bidirectional transformer model used which I used for this task. These parts are virtually identical to the modules for BERT.

`training.py` is where I define my actual transformer, and train it.