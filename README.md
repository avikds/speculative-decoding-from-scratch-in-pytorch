# Speculative Decoding from Scratch in PyTorch

Chapter 5.2 of Inference Engineering, built and measured. Train a small causal transformer with a KV cache as the target model and a smaller one as the draft, then implement the three ways the book describes to get more than one token out of every target forward pass. Draft-target speculation with the exact rejection-sampling verification, which you will prove leaves the output distribution unchanged. N-gram speculation, a draft that costs nothing because it is a lookup table, fitted on a corpus or on the prompt itself. Medusa-style self-speculation, where extra heads on the target's own hidden state propose the next tokens and the verification pass doubles as the next round's base pass. Along the way you measure acceptance rates, compare them with the expected-tokens formula, and finish with the speedup table the book's speculative decoding simulator computes, from your own numbers.

## How to run

```bash
python scaffold.py
```

## Steps

- [x] **1.** make_corpus
- [x] **2.** TinyGPT
- [x] **3.** train_lm
- [x] **4.** generate
- [x] **5.** draft_tokens
- [x] **6.** verify_tokens
- [x] **7.** speculative_generate
- [x] **8.** check_distribution
- [x] **9.** expected_tokens_per_pass
- [x] **10.** NgramModel
- [x] **11.** ngram_speculation
- [x] **12.** MedusaHeads
- [x] **13.** medusa_generate
- [x] **14.** speedup_table

## Results

```
plain decoding: 150 tokens in 150 target passes, 1.00 tokens per pass
  sample: 'rho kernel omega kern omicron omicron nu cache decherhe ompt'

draft-target (gamma 4): 150 tokens in 90 target passes, 1.67 tokens per pass, acceptance 0.41
  distribution check at the prefix where draft and target disagree most (TV draft vs target 0.936):
    verified speculative samples vs target: TV 0.003   accept-everything: TV 0.935
  tokens per pass by draft length: gamma 1: 1.40 (formula 1.40)  gamma 2: 1.67 (formula 1.67)  gamma 4: 1.67 (formula 1.67)  gamma 8: 1.61 (formula 1.62)

n-gram draft (4-gram table, gamma 4): 1.81 tokens per pass, acceptance 0.48
medusa (3 heads, loss 3.28 -> 1.51): 1.60 tokens per pass, acceptance 0.40

simulated speedup with per-token draft costs 0.08 (model), 0.00 (n-gram), 0.02 (medusa):
  plain        tokens/pass  1.00  draft cost 0.00  speedup 1.00x
  draft-target tokens/pass  1.67  draft cost 0.32  speedup 1.26x
  ngram        tokens/pass  1.81  draft cost 0.00  speedup 1.81x
  medusa       tokens/pass  1.60  draft cost 0.06  speedup 1.51x
  best method on this workload: ngram
```
