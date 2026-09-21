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

# Step 3 - train_lm
def get_batch(data, batch, seq_len, gen):
    # Draw random starting offsets using the supplied generator.
    starts = torch.randint(
        0,
        len(data) - seq_len - 1,
        (batch,),
        generator=gen,
    )

    # Input tokens.
    x = torch.stack([
        data[start:start + seq_len]
        for start in starts
    ])

    # Targets are shifted by exactly one token.
    y = torch.stack([
        data[start + 1:start + seq_len + 1]
        for start in starts
    ])

    return x, y


def _ensure_model_vocab(model, data):
    """
    Ensure the model can represent every token ID occurring in data.

    Step 1 reports the alphabetic vocabulary size through tok.vocab,
    while encode() also assigns IDs to space and period. Therefore the
    actual number of token IDs can be larger than model.head.out_features.
    """
    required_vocab = int(data.max().item()) + 1
    current_vocab = model.tok.num_embeddings

    if required_vocab <= current_vocab:
        return

    # Expand the token embedding while preserving existing weights.
    old_tok = model.tok
    new_tok = nn.Embedding(
        required_vocab,
        old_tok.embedding_dim,
        device=old_tok.weight.device,
        dtype=old_tok.weight.dtype,
    )

    with torch.no_grad():
        new_tok.weight[:current_vocab].copy_(old_tok.weight)

    model.tok = new_tok

    # Expand the output projection so all token IDs are valid targets.
    old_head = model.head
    new_head = nn.Linear(
        old_head.in_features,
        required_vocab,
        bias=False,
        device=old_head.weight.device,
        dtype=old_head.weight.dtype,
    )

    with torch.no_grad():
        new_head.weight[:current_vocab].copy_(old_head.weight)

    model.head = new_head


def train_lm(model, data, steps, lr=3e-3, batch=32, seq_len=64, seed=0):
    # Make sure the model vocabulary covers every encoded token.
    # This must happen before constructing the optimizer.
    _ensure_model_vocab(model, data)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
    )

    # Dedicated generator for reproducible batch sampling.
    gen = torch.Generator()
    gen.manual_seed(seed)

    losses = []

    model.train()

    for _ in range(steps):
        x, y = get_batch(
            data,
            batch,
            seq_len,
            gen,
        )

        optimizer.zero_grad(set_to_none=True)

        logits, _ = model(x)

        # Cross-entropy over all positions in the batch.
        loss = nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            y.reshape(-1),
        )

        loss.backward()

        # Required gradient clipping.
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            1.0,
        )

        optimizer.step()

        losses.append(loss.item())

    return losses


def make_models(seed=0, target_steps=300, draft_steps=300):
    # Build the deterministic synthetic corpus.
    text = make_corpus(400, seed)

    # Build the character tokenizer.
    tok = CharTokenizer(text)

    # Encode the complete corpus as a 1-D long tensor.
    data = torch.tensor(
        tok.encode(text),
        dtype=torch.long,
    )

    # Construct the target model after setting the requested seed.
    torch.manual_seed(seed)
    target = TinyGPT(
        tok.vocab,
        64,
        2,
        4,
    )

    # Construct the draft model with seed + 1.
    torch.manual_seed(seed + 1)
    draft = TinyGPT(
        tok.vocab,
        32,
        1,
        2,
    )

    # Train both models using the same batch RNG seed.
    train_lm(
        target,
        data,
        target_steps,
        seed=seed,
    )

    train_lm(
        draft,
        data,
        draft_steps,
        seed=seed,
    )

    # Both models must be in evaluation mode after training.
    target.eval()
    draft.eval()

    return tok, data, target, draft

# Step 4 - generate
def sample_from(probs, gen):
    # Draw one token index from the probability distribution.
    return int(torch.multinomial(probs, 1, generator=gen).item())


def next_probs(logits_row, temperature):
    if temperature > 0:
        # Temperature-scaled softmax.
        return torch.softmax(logits_row / temperature, dim=-1)

    # At temperature zero, use a deterministic argmax as a one-hot vector.
    idx = torch.argmax(logits_row)
    probs = torch.zeros_like(logits_row)
    probs[idx] = 1.0
    return probs


@torch.no_grad()
def generate(model, prompt, n, temperature=1.0, gen=None):
    # Nothing to generate means no model forward calls.
    if n <= 0:
        return [], 0

    # TinyGPT expects a batch dimension.
    idx = prompt.unsqueeze(0)

    # First forward pass processes the complete prompt and creates the cache.
    logits, cache = model(idx)

    tokens = []
    passes = 1

    # Sample the first generated token directly from the final prompt position.
    probs = next_probs(logits[0, -1], temperature)
    token = sample_from(probs, gen)
    tokens.append(token)

    # Generate the remaining tokens one at a time using the KV cache.
    while len(tokens) < n:
        next_idx = torch.tensor(
            [[token]],
            dtype=prompt.dtype,
            device=prompt.device,
        )

        logits, cache = model(next_idx, cache)

        probs = next_probs(logits[0, -1], temperature)
        token = sample_from(probs, gen)

        tokens.append(token)
        passes += 1

    return tokens, passes

# Step 5 - draft_tokens
@torch.no_grad()
def draft_tokens(draft, prefix, gamma, temperature=1.0, gen=None):
    # Keep the growing sequence as a 1-D tensor.
    current = prefix

    draft_ids = []
    draft_probs = []

    for _ in range(gamma):
        # The draft model processes the entire growing prefix.
        logits, _ = draft(current.unsqueeze(0))

        # Distribution for the next token.
        probs = next_probs(logits[0, -1], temperature)

        # Sample the next token from that distribution.
        token = sample_from(probs, gen)

        draft_ids.append(token)
        draft_probs.append(probs)

        # Feed the sampled token back into the next forward pass.
        next_token = torch.tensor(
            [token],
            dtype=current.dtype,
            device=current.device,
        )
        current = torch.cat((current, next_token), dim=0)

    # Convert sampled IDs and distributions to tensors.
    ids = torch.tensor(
        draft_ids,
        dtype=torch.long,
        device=prefix.device,
    )

    probs = torch.stack(draft_probs, dim=0)

    return ids, probs


def model_draft_fn(draft, temperature=1.0, gen=None):
    # Return a generic drafter function so later speculative
    # decoding code can replace the drafting strategy.
    def f(prefix, gamma):
        return draft_tokens(
            draft,
            prefix,
            gamma,
            temperature=temperature,
            gen=gen,
        )

    return f

# Step 6 - verify_tokens
def verify_tokens(target_probs, draft_ids, draft_probs, gen=None):
    """
    Verify draft tokens using exact speculative-decoding rejection sampling.

    target_probs:
        (gamma + 1, vocab), with one target distribution for each
        proposed token plus a final distribution for the bonus token.

    draft_ids:
        (gamma,), the tokens proposed by the draft model.

    draft_probs:
        (gamma, vocab), the draft distributions used to sample draft_ids.

    Returns:
        (tokens, n_accepted)
    """
    tokens = []
    n_accepted = 0

    gamma = draft_ids.numel()

    for i in range(gamma):
        token = int(draft_ids[i].item())

        # Target and draft probabilities for the proposed token.
        p = target_probs[i, token]
        q = draft_probs[i, token]

        # Acceptance probability is min(1, p / q).
        # A zero q cannot occur for a token sampled from q in exact
        # arithmetic, but handle it safely for robustness.
        if q.item() == 0.0:
            accept_prob = 1.0 if p.item() > 0.0 else 0.0
        else:
            accept_prob = min(1.0, (p / q).item())

        u = torch.rand((), generator=gen, device=target_probs.device)

        if u.item() < accept_prob:
            # Draft token accepted.
            tokens.append(token)
            n_accepted += 1
        else:
            # First rejection: sample from the residual distribution
            # max(0, target_probs[i] - draft_probs[i]).
            residual = torch.clamp(
                target_probs[i] - draft_probs[i],
                min=0.0,
            )

            total = residual.sum()

            # Normalize the residual before sampling.
            if total.item() > 0.0:
                residual = residual / total
            else:
                # This should not occur for a valid rejection event,
                # but prevents an invalid probability distribution
                # from reaching torch.multinomial due to floating-point
                # round-off.
                residual = target_probs[i]

                residual_sum = residual.sum()
                if residual_sum.item() > 0.0:
                    residual = residual / residual_sum
                else:
                    residual = torch.zeros_like(residual)
                    residual[torch.argmax(target_probs[i])] = 1.0

            replacement = sample_from(residual, gen)

            tokens.append(replacement)
            return tokens, n_accepted

    # Every draft token was accepted, so draw one bonus token from
    # the target distribution after all proposed tokens.
    bonus = sample_from(target_probs[gamma], gen)
    tokens.append(bonus)

    return tokens, n_accepted

# Step 7 - speculative_generate
@torch.no_grad()
def speculative_generate(
    target,
    draft_fn,
    prompt,
    n,
    gamma=4,
    temperature=1.0,
    gen=None,
):
    # Generated output tokens only; prompt is kept separately in ids.
    tokens = []
    ids = prompt.clone()

    stats = {
        "target_passes": 0,
        "rounds": 0,
        "drafted": 0,
        "examined": 0,
        "accepted": 0,
    }

    # Nothing to generate.
    if n <= 0:
        return tokens[:n], stats

    while len(tokens) < n:
        # Draft gamma candidate tokens from the current prefix.
        draft_ids, draft_probs = draft_fn(ids, gamma)

        stats["rounds"] += 1
        stats["drafted"] += int(draft_ids.numel())

        # One target forward pass on the prefix followed by all drafts.
        target_input = torch.cat(
            [
                ids,
                draft_ids.to(device=ids.device, dtype=ids.dtype),
            ],
            dim=0,
        ).unsqueeze(0)

        logits, _ = target(target_input)
        stats["target_passes"] += 1

        # For draft token i, the relevant target distribution is the
        # next-token distribution at position len(ids) - 1 + i.
        prefix_len = ids.numel()

        target_probs = torch.stack([
            next_probs(
                logits[0, prefix_len - 1 + i],
                temperature,
            )
            for i in range(gamma + 1)
        ])

        # Verify the proposed tokens using exact rejection sampling.
        verified_tokens, n_accepted = verify_tokens(
            target_probs,
            draft_ids,
            draft_probs,
            gen=gen,
        )

        # The verifier examines all accepted drafts and, if necessary,
        # the first rejected draft. It does not examine later drafts.
        stats["accepted"] += n_accepted

        if n_accepted < gamma:
            # A rejection occurred. The rejected draft itself was examined.
            stats["examined"] += n_accepted + 1
        else:
            # Every proposed draft was examined and accepted.
            stats["examined"] += gamma

        # Append the verified tokens to both the generated sequence
        # and the running prefix used by the next speculation round.
        verified_tensor = torch.tensor(
            verified_tokens,
            dtype=ids.dtype,
            device=ids.device,
        )

        tokens.extend(verified_tokens)
        ids = torch.cat(
            [ids, verified_tensor],
            dim=0,
        )

    return tokens[:n], stats


def tokens_per_pass(tokens, stats):
    return len(tokens) / stats["target_passes"]


def acceptance_rate(stats):
    return stats["accepted"] / stats["examined"]

# Step 8 - check_distribution
@torch.no_grad()
def exact_next_probs(model, prefix, temperature=1.0):
    # Run the model on the complete prefix and take the distribution
    # corresponding to the next token after the final prefix position.
    logits, _ = model(prefix.unsqueeze(0))
    return next_probs(logits[0, -1], temperature)


def total_variation(p, q):
    # TV distance = 1/2 * L1 distance.
    return 0.5 * torch.sum(torch.abs(p - q)).item()


@torch.no_grad()
def first_token_histogram(
    target,
    draft,
    prefix,
    n_samples,
    gen,
    accept_all=False,
):
    # The histogram covers the model vocabulary.
    vocab = target.head.out_features
    counts = torch.zeros(
        vocab,
        dtype=torch.float32,
        device=prefix.device,
    )

    for _ in range(n_samples):
        # Draft exactly one token from the current prefix.
        draft_ids, draft_probs = draft_tokens(
            draft,
            prefix,
            1,
            gen=gen,
        )

        draft_token = int(draft_ids[0].item())

        if accept_all:
            # Skip verification and keep the draft token directly.
            token = draft_token
        else:
            # Run the target once on prefix + the single draft token.
            target_input = torch.cat(
                [
                    prefix,
                    draft_ids.to(
                        device=prefix.device,
                        dtype=prefix.dtype,
                    ),
                ],
                dim=0,
            ).unsqueeze(0)

            logits, _ = target(target_input)

            # Row 0 predicts the draft token; row 1 is the distribution
            # after that token and acts as the bonus distribution.
            target_probs = torch.stack([
                next_probs(logits[0, -2], 1.0),
                next_probs(logits[0, -1], 1.0),
            ])

            verified, _ = verify_tokens(
                target_probs,
                draft_ids,
                draft_probs,
                gen=gen,
            )

            # gamma=1 always produces exactly one output token.
            token = verified[0]

        counts[token] += 1.0

    # Convert counts to the empirical probability distribution.
    return counts / n_samples


@torch.no_grad()
def hardest_prefix(target, draft, data, length=20, stride=50):
    # Search only windows whose starting offsets are multiples of stride.
    max_start = len(data) - length
    best_start = 0
    best_tv = -1.0

    for start in range(0, max_start + 1, stride):
        prefix = data[start:start + length]

        target_probs = exact_next_probs(target, prefix)
        draft_probs = exact_next_probs(draft, prefix)

        tv = total_variation(target_probs, draft_probs)

        if tv > best_tv:
            best_tv = tv
            best_start = start

    return data[best_start:best_start + length]


@torch.no_grad()
def check_distribution(
    target,
    draft,
    prefix,
    n_samples=2000,
    gen=None,
):
    # Compute the target's exact next-token distribution.
    target_probs = exact_next_probs(target, prefix)

    # Empirical distribution from exact speculative verification.
    spec_hist = first_token_histogram(
        target,
        draft,
        prefix,
        n_samples,
        gen,
        accept_all=False,
    )

    # Empirical distribution when every draft token is accepted.
    accept_all_hist = first_token_histogram(
        target,
        draft,
        prefix,
        n_samples,
        gen,
        accept_all=True,
    )

    # The draft model's own next-token distribution.
    draft_probs = exact_next_probs(draft, prefix)

    return {
        "tv_speculative": total_variation(
            spec_hist,
            target_probs,
        ),
        "tv_accept_all": total_variation(
            accept_all_hist,
            target_probs,
        ),
        "tv_draft": total_variation(
            draft_probs,
            target_probs,
        ),
    }

# Step 9 - expected_tokens_per_pass
def expected_tokens_per_pass(alpha, gamma):
    if alpha == 1:
        return gamma + 1

    return (1 - alpha ** (gamma + 1)) / (1 - alpha)


def acceptance_by_gamma(target, draft, prompt, n, gammas, gen_seed=0):
    results = {}

    for gamma in gammas:
        # Use the same deterministic generator for both drafting and
        # verification during this gamma's complete generation run.
        g = torch.Generator().manual_seed(gen_seed)

        draft_fn = model_draft_fn(
            draft,
            gen=g,
        )

        tokens, stats = speculative_generate(
            target,
            draft_fn,
            prompt,
            n,
            gamma=gamma,
            gen=g,
        )

        alpha = acceptance_rate(stats)
        measured = tokens_per_pass(tokens, stats)
        formula = expected_tokens_per_pass(alpha, gamma)

        results[gamma] = {
            "alpha": alpha,
            "measured": measured,
            "formula": formula,
        }

    return results

# Step 10 - NgramModel
class NgramModel:
    def __init__(self, n, vocab, k=0.1):
        self.n = n
        self.vocab = vocab
        self.k = k
        self.counts = {}

    def fit(self, ids):
        # Accept either a Python list or a 1-D tensor.
        if isinstance(ids, torch.Tensor):
            ids = ids.detach().cpu().tolist()
        else:
            ids = list(ids)

        ids = [int(x) for x in ids]

        # The tokenizer's reported vocab is 24, while the actual
        # encoded character IDs also include space and period.
        # Expand to cover every token that occurs in the data.
        if ids:
            required_vocab = max(ids) + 1

            if required_vocab > self.vocab:
                old_vocab = self.vocab
                self.vocab = required_vocab

                # Expand all previously-created count vectors.
                for context, old_counts in self.counts.items():
                    new_counts = torch.zeros(
                        self.vocab,
                        dtype=torch.float32,
                    )
                    new_counts[:old_vocab] = old_counts
                    self.counts[context] = new_counts

        context_len = self.n - 1

        # Count every n-token window:
        # (n-1)-token context -> following token.
        for i in range(len(ids) - self.n + 1):
            context = tuple(ids[i:i + context_len])
            next_token = ids[i + context_len]

            if context not in self.counts:
                self.counts[context] = torch.zeros(
                    self.vocab,
                    dtype=torch.float32,
                )

            self.counts[context][next_token] += 1.0

        return self

    def probs(self, context):
        # Convert the context to a Python list.
        if isinstance(context, torch.Tensor):
            context = context.detach().cpu().tolist()
        else:
            context = list(context)

        context = [int(x) for x in context]
        context_len = self.n - 1

        # A context shorter than n-1 has no usable history.
        if len(context) < context_len:
            return torch.full(
                (self.vocab,),
                1.0 / self.vocab,
                dtype=torch.float32,
            )

        # Only the final n-1 tokens matter.
        key = tuple(context[-context_len:])

        counts = self.counts.get(key)

        # Unseen context -> uniform distribution.
        if counts is None:
            return torch.full(
                (self.vocab,),
                1.0 / self.vocab,
                dtype=torch.float32,
            )

        # Add-k smoothing:
        #     P(x | context) = (count(x) + k) /
        #                      (total + k * vocab)
        probs = counts + self.k
        probs = probs / (
            counts.sum() + self.k * self.vocab
        )

        return probs

    @torch.no_grad()
    def draft(self, prefix, gamma, gen=None, greedy=False):
        # Work with a Python list for sequential context updates.
        if isinstance(prefix, torch.Tensor):
            current = prefix.detach().cpu().tolist()
            device = prefix.device
        else:
            current = list(prefix)
            device = None

        current = [int(x) for x in current]

        ids = []
        probs = []

        for _ in range(gamma):
            # Get the next-token distribution from the N-gram table.
            p = self.probs(current)

            if greedy:
                token = int(torch.argmax(p).item())
            else:
                token = int(
                    torch.multinomial(
                        p,
                        1,
                        generator=gen,
                    ).item()
                )

            ids.append(token)
            probs.append(p)

            # Extend the context for the next prediction.
            current.append(token)

        draft_ids = torch.tensor(
            ids,
            dtype=torch.long,
            device=device,
        )

        if len(probs) > 0:
            draft_probs = torch.stack(probs, dim=0)
        else:
            draft_probs = torch.empty(
                (0, self.vocab),
                dtype=torch.float32,
            )

        if device is not None:
            draft_probs = draft_probs.to(device)

        return draft_ids, draft_probs

    def draft_fn(self, gen=None, greedy=False):
        # Match the same f(prefix, gamma) interface used by
        # model_draft_fn() in the draft-target implementation.
        def f(prefix, gamma):
            return self.draft(
                prefix,
                gamma,
                gen=gen,
                greedy=greedy,
            )

        return f

