"""
Speculative Decoding from Scratch in PyTorch scaffold.

Run this with: python scaffold.py
Uses functions defined in model.py.
"""

from model import *  # noqa: F401, F403 (pulls in your solution functions)

"""Speculative Decoding from Scratch in PyTorch (Inference Engineering, chapter 5.2).

Story: train a target and a draft on a synthetic language; decode plainly at one
token per pass; add draft-target speculation with exact verification and prove the
output distribution is unchanged; swap in an n-gram table as a free drafter; put
Medusa heads on the target so it drafts for itself; then fill the book's speedup
table with measured acceptance rates.
"""
import torch


def main() -> None:
    tok, data, target, draft = make_models(seed=0)
    prompt = data[:20]
    n = 150

    # ---- 1. The baseline ----
    tokens, passes = generate(target, prompt, n, gen=torch.Generator().manual_seed(0))
    print(f"plain decoding: {len(tokens)} tokens in {passes} target passes, 1.00 tokens per pass")
    print(f"  sample: {tok.decode(tokens[:60])!r}")

    # ---- 2. Draft-target speculation ----
    g = torch.Generator().manual_seed(0)
    tokens, stats = speculative_generate(target, model_draft_fn(draft, gen=g), prompt, n, gamma=4, gen=g)
    print(f"\ndraft-target (gamma 4): {len(tokens)} tokens in {stats['target_passes']} target passes, "
          f"{tokens_per_pass(tokens, stats):.2f} tokens per pass, acceptance {acceptance_rate(stats):.2f}")
    prefix = hardest_prefix(target, draft, data)
    r = check_distribution(target, draft, prefix, n_samples=1500, gen=torch.Generator().manual_seed(0))
    print(f"  distribution check at the prefix where draft and target disagree most (TV draft vs target {r['tv_draft']:.3f}):")
    print(f"    verified speculative samples vs target: TV {r['tv_speculative']:.3f}   accept-everything: TV {r['tv_accept_all']:.3f}")
    table = acceptance_by_gamma(target, draft, prompt, n, [1, 2, 4, 8])
    print("  tokens per pass by draft length: " + "  ".join(f"gamma {g}: {table[g]['measured']:.2f} (formula {table[g]['formula']:.2f})" for g in table))

    # ---- 3. N-gram speculation ----
    ng = NgramModel(4, tok.vocab)
    ng.fit(data)
    g = torch.Generator().manual_seed(0)
    tokens, stats = ngram_speculation(target, ng, prompt, n, gamma=4, gen=g)
    print(f"\nn-gram draft (4-gram table, gamma 4): {tokens_per_pass(tokens, stats):.2f} tokens per pass, acceptance {acceptance_rate(stats):.2f}")

    # ---- 4. Medusa heads ----
    torch.manual_seed(0)
    heads = MedusaHeads(64, tok.vocab, 3)
    losses = train_medusa(target, heads, data, 150)
    g = torch.Generator().manual_seed(0)
    tokens, stats = medusa_generate(target, heads, prompt, n, gen=g)
    print(f"medusa (3 heads, loss {losses[0]:.2f} -> {losses[-1]:.2f}): {tokens_per_pass(tokens, stats):.2f} tokens per pass, acceptance {acceptance_rate(stats):.2f}")

    # ---- 5. The speedup table ----
    rows = run_all_methods(target, draft, ng, heads, prompt, n, gamma=4)
    lines = speedup_table(rows, {"plain": 0.0, "draft-target": 0.08, "ngram": 0.0, "medusa": 0.02})
    print("\nsimulated speedup with per-token draft costs 0.08 (model), 0.00 (n-gram), 0.02 (medusa):")
    for line in lines:
        print("  " + line)
    print(f"  best method on this workload: {best_method(lines)}")


if __name__ == "__main__":
    main()

