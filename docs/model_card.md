# Model card — `lmops` reference model

A model card for a deliberately small research model. It is here because the habit
matters more than the model does: anything trained and served ought to ship with a
written statement of what it is for, what it was trained on, and where it fails.

---

## Model details

| | |
| --- | --- |
| **Developed by** | [Ashu2621](https://github.com/Ashu2621) |
| **Model type** | Decoder-only transformer language model (causal LM) |
| **Language** | English (Early Modern English, specifically) |
| **License** | MIT |
| **Repository** | <https://github.com/Ashu2621/lmops> |
| **Training config** | [`configs/small.yaml`](../configs/small.yaml) |
| **Framework** | PyTorch, implemented from scratch — no `transformers`, no `tokenizers` |

### Architecture

| Component | Value |
| --- | --- |
| Layers | 4 |
| Model width (`d_model`) | 256 |
| Attention heads | 8 query / 8 key-value (multi-head) |
| Head dimension | 32 |
| Feed-forward | SwiGLU, inner width 704 |
| Normalisation | RMSNorm, pre-norm placement |
| Positions | Rotary (RoPE), θ = 10000 |
| Context length | 192 tokens |
| Vocabulary | 1024 (byte-level BPE, trained from scratch) |
| Embedding tying | Yes (`lm_head.weight is token_emb.weight`) |
| Dropout | 0.1 |

Design rationale for each choice: [`architecture.md`](architecture.md).

---

## Intended use

**In scope.** Demonstrating and testing an end-to-end language-model pipeline —
tokenizer training, model training, evaluation, graph export, and low-latency serving.
It is a reference implementation and a test fixture. It is also a reasonable starting
point for anyone who wants to understand a modern decoder stack by reading ~1,500
lines rather than a framework.

**Out of scope.** This model must not be used for anything that matters. It is far too
small to be useful as a general-purpose language model, it was trained on a single
1.1 MB literary corpus, and it has had no instruction tuning, no safety tuning and no
alignment work of any kind. Specifically not for: factual question answering, advice
of any kind, content moderation, user-facing products, or any decision affecting a
person.

**Foreseeable misuse.** Its output is fluent-looking pastiche. Presenting it as
authored text, or as information, would mislead — the model has no notion of truth and
no knowledge beyond the statistics of one book of plays.

---

## Training data

The concatenated works of Shakespeare (`data/raw/tinyshakespeare.txt`), ~1.1M
characters / ~1.1 MB of UTF-8 text. **Public domain.** This is the standard small-scale
language-modelling corpus popularised by Karpathy's `char-rnn`.

- **Split**: the final 10% of the corpus by position is held out for validation. The
  split is positional rather than random because training windows overlap heavily, and
  a random split lets near-duplicate neighbouring windows land on both sides and
  flatter the validation loss.
- **Tokenisation**: byte-level BPE with a 1024-token vocabulary, merges learned from
  the first 500,000 characters. Achieved compression: **2.40 characters per token**.
- **Preprocessing**: none. The text is used as-is; no filtering, deduplication or
  normalisation was applied.

**Bias.** The corpus is the writing of one author, from one culture, in the late 16th
and early 17th centuries. Every social attitude embedded in that text — about gender,
class, nationality, religion — is in the training signal and will surface in
generations. Nothing was done to measure or mitigate this, because the model has no
use case where it would matter. Do not treat that as a template for a model that does.

---

## Evaluation

Reported on the held-out validation shard using non-overlapping sequential windows, so
the numbers are exactly reproducible:

```bash
lmops eval --checkpoint runs/small/best.pt --data data/processed
```

Measured results: [`benchmarks.md`](benchmarks.md).

**Metric note.** Perplexity is reported because it is expected, but it is *not*
comparable across tokenizers — a vocabulary that packs more characters into each token
earns a lower perplexity for free. **Bits-per-byte** normalises by the underlying UTF-8
bytes and is the number to use when comparing runs with different vocabularies.

**What was not evaluated.** No downstream task benchmarks, no factuality evaluation, no
bias or toxicity measurement, no robustness testing. For a model of this size trained
on this corpus, those evaluations would be measuring noise.

---

## Limitations

- **Scale.** A few million parameters trained on ~0.4M tokens for ~3.7M token-updates.
  It produces text with the *shape* of Shakespearean dialogue — speaker headings, line
  breaks, plausible morphology — and essentially no coherence beyond a clause.
- **Context.** 192 tokens (~460 characters). Anything earlier is gone.
- **Domain.** Any prompt outside its corpus is out of distribution. The tokenizer will
  encode it losslessly; the model will ignore it.
- **No knowledge.** No facts, no dates, no arithmetic, no reasoning.
- **Determinism.** Reproducible only when a seed is supplied (`--seed`, or `"seed"` in
  an API request). Unseeded sampling is genuinely random.

---

## Environmental and compute footprint

Trained on a **laptop CPU** — no GPU was used at any point. That is a deliberate
constraint rather than a limitation to apologise for: a baseline that requires hardware
the reader does not have is not a baseline they can check. Exact wall-clock time and
throughput are in [`benchmarks.md`](benchmarks.md).

---

## How to reproduce

```bash
pip install --index-url https://download.pytorch.org/whl/cpu "torch>=2.2"
pip install -e ".[dev,serve,export]"

lmops prepare --input data/raw/tinyshakespeare.txt --out data/processed --vocab-size 1024
lmops train   --config configs/small.yaml
lmops eval    --checkpoint runs/small/best.pt --data data/processed
```

Training is seeded (`train.seed: 1337`). Run-to-run variation on identical hardware
comes only from non-deterministic CPU kernel scheduling; set `train.deterministic: true`
to constrain that further, at some cost in speed.
