# Architecture and design decisions

This document explains *why* the code looks the way it does. The code itself explains
what it does.

---

## 1. Shape of the system

```
data/raw/*.txt
      │
      │  lmops prepare        byte-level BPE trained from scratch
      ▼
data/processed/               tokenizer.json + train.bin + val.bin (uint16, memmapped)
      │
      │  lmops train          AdamW, cosine+warmup, AMP, grad accumulation
      ▼
runs/<name>/                  best.pt / last.pt / metrics.jsonl / config.yaml
      │
      ├── lmops eval          loss, perplexity, bits-per-byte, top-1 accuracy
      ├── lmops bench         TTFT, per-token p50/p95, KV-cache speedup
      ├── lmops export ──────▶ artifacts/model.pt (TorchScript), model.onnx  [+ parity check]
      └── lmops serve ───────▶ FastAPI: /v1/generate, SSE stream, /healthz, /readyz, /metrics
```

Each stage reads the previous stage's artifacts from disk and nothing else. That is
what makes the CI `pipeline` job meaningful: it runs the stages in order against real
files, so a mismatch between what training writes and what serving expects fails the
build rather than production.

---

## 2. The model

A decoder-only transformer, ~3.5M parameters in the reference config.

| Choice | Why |
| --- | --- |
| **Pre-norm residual blocks** | Post-norm needs careful warmup to train at all past a handful of layers. Pre-norm keeps a clean gradient path through the residual stream. |
| **RMSNorm** | Same stabilising effect as LayerNorm without the mean subtraction, so one fewer reduction per call. Computed in float32 even under autocast — `rsqrt` of a half-precision mean-square is a well-known source of blow-ups. |
| **RoPE** | Relative position encoded by rotating queries and keys, so attention scores depend only on *distance*. No learned position table to bound or extrapolate, and the cache resumes at an offset for free. |
| **SwiGLU feed-forward** | Consistently beats ReLU/GELU MLPs at equal parameter count. Inner width is `8/3 · d_model` rounded up, the usual rule for matching a plain MLP's parameter count given three projections instead of two. |
| **Grouped-query attention** | `n_kv_head < n_head` shrinks the KV cache linearly, which is the memory that actually bounds concurrent sequences at serving time. Set `n_kv_head == n_head` for plain MHA. See the measured ablation in [benchmarks.md](benchmarks.md). |
| **Tied embeddings** | Removes `vocab_size × d_model` parameters and usually helps at small scale, where the embedding table is a large fraction of the model. |
| **Scaled residual init** | Output projections are initialised at `init_std / sqrt(2·n_layer)`, so residual-stream variance does not grow linearly with depth. |

### The KV cache is the part worth reading

Cached incremental decoding must produce *exactly* the logits a full forward pass
would. Two invariants make that true, and both fail silently when broken — the model
keeps producing fluent text, it just stops producing the *right* text.

1. **Rotary positions must use absolute offsets.** At decode step *t* the model is fed
   a single token, but its rotary embedding has to be taken at position *t*, not
   position 0 of the one-token slice. `RotaryEmbedding.forward(seq_len, offset=...)`
   threads that through.

2. **`is_causal=True` is wrong when a cache is present.** PyTorch's
   `scaled_dot_product_attention` aligns its built-in causal mask to the *top-left* of
   the attention matrix. With a cache, queries sit at the bottom of a longer key axis,
   so the flag masks the wrong entries. Cached passes build an explicit offset mask
   (`build_cache_mask`) instead; the flag is only used when query and key axes start at
   the same absolute position.

`tests/test_attention.py` asserts single-token decoding *and* chunked prefill both
match a full forward pass. Those two tests are what keep the invariants honest.

---

## 3. Tokenizer

Byte-level BPE, trained from scratch in `src/lmops/data/tokenizer.py`.

**Byte-level** means the 256 single-byte tokens are always present, so there is no
out-of-vocabulary case and `decode(encode(s)) == s` for *every* string — emoji, CJK,
malformed-looking input, scripts the corpus never contained. The tokenizer tests
assert exactly that on text outside the training distribution.

**Training keeps incremental pair counts.** The textbook algorithm rescans every word
once per merge, which is quadratic in the merge count. This implementation maintains
`pair_counts` plus a `pair -> {word ids}` index and touches only the words a merge
actually affects. `test_incremental_training_matches_naive_reference` reimplements the
naive version in a dozen lines and asserts both learn identical merges, so the
optimisation cannot drift away from the definition of BPE.

**Pre-tokenisation** splits on a GPT-2-style regex before BPE runs, which prevents
merges that straddle a space or punctuation boundary.

**Streaming needs `token_bytes`.** A BPE token can end halfway through a multi-byte
character, so a server that decodes each token independently emits mojibake.
`IncrementalDecoder` buffers bytes and releases only complete characters.

---

## 4. Data pipeline

Token shards are flat `uint16` binaries read with `np.memmap`. Start-up cost is
independent of corpus size, and several processes share one page cache.

**The train/val split is positional, not random.** Windows overlap heavily, so a
random split lets near-duplicate neighbours land on both sides and quietly flatters
the validation loss. The validation shard is the tail of the corpus.
`test_prepare_splits_by_position_not_at_random` pins this down.

**Evaluation uses non-overlapping sequential windows**, so the same checkpoint scores
identically on every run. Training uses random windows.

---

## 5. Training loop

- **AdamW** with parameters split into decayed and non-decayed groups. One-dimensional
  parameters — norm gains, biases — are never decayed; pulling a norm gain towards
  zero fights the normalisation it exists to perform.
- **Cosine decay with linear warmup**, floored at `min_lr_ratio · lr` rather than at
  zero, so the last stretch of a run still contributes.
- **Gradient accumulation** divides each micro-batch loss by `grad_accum_steps` so the
  accumulated gradient equals the mean over the full batch. Getting this wrong
  silently scales the effective learning rate;
  `test_gradient_accumulation_matches_a_single_large_step` asserts the equivalence
  against a single large batch.
- **Mixed precision** picks bf16 only where it is natively supported. `auto` resolves
  to float32 on CPU — emulated bf16 there is slower *and* less accurate than float32,
  so "half precision is always faster" is the wrong default.
- **Checkpoints** carry model, optimizer, GradScaler, step counter and RNG state, so a
  resumed run continues rather than restarts. Writes go to a temp file and are
  atomically renamed, so an interrupted save cannot destroy the previous `best.pt`.
  The model config is stored as a plain dict, which keeps checkpoints loadable under
  `torch.load(weights_only=True)` — the PyTorch ≥ 2.6 default and the only safe way to
  open a checkpoint you did not produce.

---

## 6. Evaluation

Perplexity is reported because everyone expects it, but it is **not comparable across
tokenizers**: a vocabulary that packs more characters into each token earns a lower
perplexity for free. **Bits-per-byte** normalises by the underlying UTF-8 bytes and is
the number to use when comparing runs with different vocabularies.

Losses are summed in float32 regardless of the autocast dtype — accumulating many
half-precision log-probabilities moves the third decimal place of a perplexity.

---

## 7. Export

Only the **full-sequence scoring graph** (`token ids -> logits`) is exported to
TorchScript and ONNX. This is a real limitation, stated rather than hidden: exporting
cached incremental decode would make the KV cache tensors graph inputs and outputs,
turning a small exporter into a stateful-session contract that every runtime
implements differently. The exported graph is the right artifact for scoring, ranking
and embedding services; the Python server keeps the cache for interactive generation.

**Every exporter is paired with a parity check**, verified at a *different* batch and
sequence length than the one used for tracing. An export that loads is not the same as
an export that is correct — a silently wrong opset conversion still produces a
perfectly valid `.onnx` file, and a trace with a baked-in shape still loads fine.

---

## 8. Serving

- **`/healthz` and `/readyz` are different endpoints.** Liveness answers "is the
  process wedged?" and never touches the model; readiness answers "should traffic be
  routed here?" and depends on a checkpoint being loaded. Wiring both to the same
  handler means a saturated server gets killed by its own orchestrator mid-request.
- **Admission control** is a bounded semaphore with a timeout. Overload becomes a fast
  `503` rather than an unbounded queue in which every request times out and every
  client retries into the same wall.
- **Blocking work runs in a thread pool.** Torch releases the GIL inside its kernels,
  so generation in a worker thread leaves the event loop free to answer health probes
  while the model is busy.
- **Metrics** cover what a generation SLO is actually written against: time-to-first
  token and total latency as separate histograms, tokens in and out as counters,
  in-flight requests as a gauge. Histogram buckets are sized for this workload rather
  than copied from a web-framework default, because buckets that all saturate say
  nothing about the tail.
- The Kubernetes HPA in `deploy/k8s/hpa.yaml` scales on `lmops_requests_in_flight`
  rather than CPU. A pod pinned at 100% CPU serving one long request looks identical
  to one serving eight short requests, but only the second is saturated.

---

## 9. What is deliberately not here

Being explicit about the boundary is more useful than pretending there isn't one.

- **Distributed training.** The loop is single-process. DDP would be a wrapper plus a
  sampler change, but untested distributed code is worse than none.
- **Continuous batching.** Requests are served one at a time per slot. Real throughput
  serving needs a paged KV cache and a scheduler that admits new sequences into a
  running batch — a different project.
- **Instruction tuning / RLHF.** This is a base language model.
- **A learned tokenizer beyond BPE.** No unigram or WordPiece implementation.
