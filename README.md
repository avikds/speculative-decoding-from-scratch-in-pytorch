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

---

Built on Deep-ML.
