`balanced_brackets.py` just contains code to check if a bracket string is balanced, and generate balanced brackets. 

The messiest part of this was how to generate balanced brackets. In the first version of this code, I based this on the fact that the set of balanced brackets can be represented by the context-free grammar with ruleset:

    S -> e | (S) | SS

But in the second part, I based it on a conceptually simpler set of rules involving altitude. You convert left brackets to `+1` and right to `-1`, then:
* A bracket string is balanced iff the sum is zero and the cumulative sum is non-negative
* A random balanced bracket string can be generated via the following method:
    * Use permutations to generate a random sequence containing an equal number of `+1` and `-1`
    * While the cumulative sum is non-negative, flip the signs of the subsequent brackets. An example of this algorithm:
        * Start with `)((())`
        * Cumulative sum is negative at 0th index, so flip everything: `()))((`
        * Cumulative sum is negative at 3rd index, so flip everything after that: `()(())`
        * Bracket string is balanced

`my_tranformer.py` defines all the internal modules of the bidirectional transformer model used which I used for this task. These parts are virtually identical to the modules for BERT.

`training.py` is where I define my actual transformer, and train it.