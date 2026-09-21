"""
Speculative Decoding from Scratch in PyTorch

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - make_corpus
import numpy as np

WORDS = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta", "iota", "kappa",
         "lambda", "mu", "nu", "xi", "omicron", "pi", "rho", "sigma", "tau", "upsilon",
         "phi", "chi", "psi", "omega", "tensor", "kernel", "cache", "token", "batch", "decode",
         "prefill", "draft", "verify", "accept", "reject", "sample", "layer", "head", "logit", "stream"]


def make_corpus(n_sentences=400, seed=0):
    rng = np.random.default_rng(seed)

    # For each word, choose three distinct likely successors in WORDS order.
    successors = [
        rng.choice(len(WORDS), size=3, replace=False)
        for _ in range(len(WORDS))
    ]

    sentences = []

    for _ in range(n_sentences):
        # Sentence length is an integer in [5, 9].
        current = rng.integers(len(WORDS))
        length = rng.integers(5, 10)

        sentence = [WORDS[current]]

        for _ in range(1, length):
            if rng.random() < 0.9:
                # Follow one of the three successors.
                current = successors[current][rng.integers(3)]
            else:
                # Uniformly jump to any word.
                current = rng.integers(len(WORDS))

            sentence.append(WORDS[current])

        sentences.append(" ".join(sentence) + ".")

    # Sentences are separated by exactly one space.
    return " ".join(sentences)


class CharTokenizer:
    def __init__(self, text):
        # Keep every character appearing in the text so encode/decode
        # can represent the complete corpus, including spaces and periods.
        self.itos = sorted(set(text))
        self.stoi = {ch: i for i, ch in enumerate(self.itos)}

    def encode(self, s):
        return [self.stoi[ch] for ch in s]

    def decode(self, ids):
        return "".join(self.itos[i] for i in ids)

    @property
    def vocab(self):
        # The project reports the number of distinct alphabetic characters.
        return sum(ch.isalpha() for ch in self.itos)

