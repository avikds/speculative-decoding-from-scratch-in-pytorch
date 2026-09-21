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

# Step 2 - TinyGPT
import math
import torch
import torch.nn as nn

class Block(nn.Module):
    def __init__(self, d, n_heads):
        super().__init__()

        assert d % n_heads == 0, "d must be divisible by n_heads"

        self.ln1 = nn.LayerNorm(d)
        self.ln2 = nn.LayerNorm(d)

        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)

        self.mlp = nn.Sequential(
            nn.Linear(d, 4 * d),
            nn.GELU(),
            nn.Linear(4 * d, d),
        )

        self.n_heads = n_heads
        self.head_dim = d // n_heads

    def forward(self, x, kv=None):
        """
        x:
            (B, T, d)

        kv:
            Optional tuple (k, v), each of shape
            (B, H, S_past, d // H)

        Returns:
            x_out:
                (B, T, d)

            (k_all, v_all):
                Cached keys and values including the new tokens.
        """
        B, T, d = x.shape

        # Pre-norm attention.
        h = self.ln1(x)

        # Compute queries, keys, and values.
        qkv = self.qkv(h)
        q, k, v = qkv.chunk(3, dim=-1)

        # (B, T, d) -> (B, H, T, head_dim)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        # Determine how many positions are already cached.
        past_len = 0 if kv is None else kv[0].size(2)

        # Append the new keys and values to the cache.
        if kv is not None:
            k_past, v_past = kv
            k_all = torch.cat([k_past, k], dim=2)
            v_all = torch.cat([v_past, v], dim=2)
        else:
            k_all = k
            v_all = v

        # Scaled dot-product attention.
        # q:     (B, H, T, D)
        # k_all: (B, H, S, D)
        # scores: (B, H, T, S)
        scores = torch.matmul(q, k_all.transpose(-2, -1))
        scores = scores / math.sqrt(self.head_dim)

        # Absolute-position causal mask.
        #
        # Query i corresponds to absolute position:
        #     past_len + i
        #
        # Therefore it may attend to key positions:
        #     0 .. past_len + i
        #
        # The key dimension has S = past_len + T positions.
        S = k_all.size(2)
        causal_mask = torch.ones(
            (T, S),
            device=x.device,
            dtype=torch.bool,
        ).tril(diagonal=past_len)

        scores = scores.masked_fill(
            ~causal_mask.unsqueeze(0).unsqueeze(0),
            torch.finfo(scores.dtype).min,
        )

        attn = torch.softmax(scores, dim=-1)

        # Attention output:
        # (B, H, T, D) -> (B, T, d)
        out = torch.matmul(attn, v_all)
        out = out.transpose(1, 2).contiguous().view(B, T, d)

        # Attention residual connection.
        x = x + self.proj(out)

        # Pre-norm MLP with residual connection.
        x = x + self.mlp(self.ln2(x))

        return x, (k_all, v_all)


class TinyGPT(nn.Module):
    def __init__(self, vocab, d, n_layers, n_heads, max_len=512):
        super().__init__()

        self.tok = nn.Embedding(vocab, d)
        self.pos = nn.Embedding(max_len, d)

        self.blocks = nn.ModuleList([
            Block(d, n_heads)
            for _ in range(n_layers)
        ])

        self.ln = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab, bias=False)

        self.max_len = max_len

    def forward(self, idx, cache=None, return_hidden=False):
        """
        idx:
            Token IDs of shape (B, T).

        cache:
            Optional list of (k, v) pairs, one for each transformer block.

        Returns:
            (logits, new_cache)

        or:

            (logits, new_cache, hidden)

        where hidden is the final normalized hidden state.
        """
        B, T = idx.shape

        # Determine the number of cached tokens.
        if cache is None or len(cache) == 0:
            past_len = 0
        else:
            past_len = cache[0][0].size(2)

        # Absolute positions begin immediately after the cached tokens.
        if past_len + T > self.max_len:
            raise ValueError(
                f"Sequence length {past_len + T} exceeds max_len={self.max_len}"
            )

        positions = torch.arange(
            past_len,
            past_len + T,
            device=idx.device,
            dtype=torch.long,
        )

        # Token + learned positional embeddings.
        x = self.tok(idx) + self.pos(positions).unsqueeze(0)

        new_cache = []

        # Run through all transformer blocks.
        for i, block in enumerate(self.blocks):
            block_cache = None if cache is None else cache[i]
            x, block_kv = block(x, block_cache)
            new_cache.append(block_kv)

        # Final normalization produces the hidden state.
        hidden = self.ln(x)

        # Project hidden states to vocabulary logits.
        logits = self.head(hidden)

        if return_hidden:
            return logits, new_cache, hidden

        return logits, new_cache

