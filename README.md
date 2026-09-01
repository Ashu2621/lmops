# lmops

**Train, evaluate, export, serve and observe a decoder-only language model — built from scratch in PyTorch.**

[![CI](https://github.com/Ashu2621/lmops/actions/workflows/ci.yml/badge.svg)](https://github.com/Ashu2621/lmops/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Checked with mypy](https://img.shields.io/badge/mypy-checked-blue.svg)](https://mypy-lang.org/)
[![Code style: ruff](https://img.shields.io/badge/style-ruff-black.svg)](https://github.com/astral-sh/ruff)

The transformer, the tokenizer and the training loop are implemented here rather than
imported — no `transformers`, no `tokenizers`, no `lightning`. Around them sits the
part that usually gets skipped in a from-scratch project: reproducible configs, graph
export with numerical parity checks, a production-shaped inference server, Prometheus
instrumentation, container and Kubernetes manifests, and CI that runs the whole
pipeline end to end on every push.

---

## Why this exists

Most "build a transformer from scratch" repos stop at a training loop that prints a
falling loss. Most "MLOps template" repos wrap a model somebody else trained. The
interesting engineering is in the seam between them:

- Does cached incremental decoding produce **exactly** the logits a full forward pass
  would? (It is easy to be fast and quietly wrong — see [the KV cache section](#the-kv-cache-is-the-part-worth-reading).)
- Does the exported ONNX graph agree with the eager model at a shape it was **not**
  traced at?
- Does the server return `422` for a request the model cannot satisfy, and `503` when
  it is saturated — instead of `500` for both?
- Does the streamed text match the non-streamed text, byte for byte, when a token ends
  halfway through a multi-byte character?

Every one of those questions is a test in this repository.

---

## Quickstart

```bash
git clone https://github.com/Ashu2621/lmops.git
cd lmops
pip install --index-url https://download.pytorch.org/whl/cpu "torch>=2.2"   # CPU wheel
pip install -e ".[dev,serve,export]"
```

The corpus ships with the repo (public-domain Shakespeare), so there is no download
step:

```bash
lmops prepare                              # train a byte-level BPE tokenizer, write token shards
lmops train  --config configs/small.yaml   # train the reference model
lmops eval   --checkpoint runs/small/best.pt --data data/processed
lmops sample --checkpoint runs/small/best.pt --prompt "KING RICHARD:" --tokens 200
lmops export --checkpoint runs/small/best.pt --out artifacts   # TorchScript + ONNX, both parity-checked
lmops bench  --checkpoint runs/small/best.pt                   # TTFT, per-token p50/p95, KV-cache speedup
lmops serve  --checkpoint runs/small/best.pt --tokenizer data/processed/tokenizer.json
```

Or via `make`: `make prepare train eval export bench serve`. `make help` lists every
target.

---

## Architecture

```
data/raw/*.txt
      │
      │  lmops prepare        byte-level BPE trained from scratch
      ▼
data/processed/               tokenizer.json + train.bin + val.bin (uint16, memmapped)
      │
      │  lmops train          AdamW · cosine+warmup · AMP · grad accumulation
      ▼
runs/<name>/                  best.pt · last.pt · metrics.jsonl · config.yaml
      │
      ├── lmops eval          loss · perplexity · bits-per-byte · top-1 accuracy
      ├── lmops bench         TTFT · per-token p50/p95 · KV-cache speedup
      ├── lmops export ──────▶ artifacts/model.pt · model.onnx      [parity-verified]
      └── lmops serve ───────▶ FastAPI: /v1/generate · SSE stream · /healthz · /readyz · /metrics
```

Each stage reads the previous stage's artifacts from disk and nothing else, which is
what makes the CI `pipeline` job meaningful: it runs the stages in order against real
files, so a mismatch between what training writes and what serving expects fails the
build rather than production.

### The model

| Component | Choice |
| --- | --- |
| Blocks | Pre-norm residual, **RMSNorm** (computed in fp32 even under autocast) |
| Positions | **RoPE** — relative position by rotating Q/K, no learned table |
| Attention | Causal, **grouped-query** (`n_kv_head ≤ n_head`), SDPA-backed |
| Feed-forward | **SwiGLU**, inner width `8/3 · d_model` rounded to a multiple of 64 |
| Head | Tied to the embedding table by default |
| Init | Normal(0, 0.02); residual projections scaled by `1/√(2·n_layer)` |
| Decoding | Pre-allocated **KV cache**, temperature / top-k / top-p / repetition penalty |

Full rationale for each choice: [`docs/architecture.md`](docs/architecture.md).

---

## The KV cache is the part worth reading

Cached incremental decoding must reproduce a full forward pass exactly. Two invariants
make that true, and **both fail silently** — the model keeps generating fluent text, it
just stops generating the *right* text.

**1. Rotary positions must use absolute offsets.** At decode step *t* the model sees a
single token, but its rotary embedding has to be taken at position *t*, not position 0
of the one-token slice.

**2. `is_causal=True` is wrong when a cache is present.** PyTorch's
`scaled_dot_product_attention` aligns its built-in causal mask to the *top-left* of the
attention matrix. With a cache, queries sit at the bottom of a longer key axis, so the
flag masks the wrong entries:

```
no cache, is_causal=True          with cache, q_len=3, offset=5
   k0 k1 k2                          k0 k1 k2 k3 k4 k5 k6 k7
q0  ✓  ·  ·                      q5   ✓  ✓  ✓  ✓  ✓  ✓  ·  ·
q1  ✓  ✓  ·                      q6   ✓  ✓  ✓  ✓  ✓  ✓  ✓  ·
q2  ✓  ✓  ✓                      q7   ✓  ✓  ✓  ✓  ✓  ✓  ✓  ✓
                                 ↑ is_causal would mask this as if it started at k0
```

So cached passes build an explicit offset mask; the flag is used only when the query
and key axes start at the same absolute position.

The tests that keep this honest assert that **single-token decoding** and **chunked
prefill followed by single-token decode** both match a full forward pass to `1e-5`:

```python
def test_kv_cache_matches_full_forward(tiny_model):
    reference, _ = tiny_model(ids)  # one full pass
    cache = KVCache.allocate(tiny_model.config, batch_size=2, max_seq_len=seq_len)
    cached = torch.cat(
        [tiny_model(ids[:, t : t + 1], cache=cache)[0] for t in range(seq_len)], dim=1
    )  # one token at a time
    torch.testing.assert_close(reference, cached, rtol=1e-4, atol=1e-5)
```

---

## Testing

Correctness properties, not coverage theatre. A selection of what is actually asserted:

| Test | What breaks without it |
| --- | --- |
| `test_kv_cache_matches_full_forward` | Fast, silently wrong generation |
| `test_kv_cache_matches_full_forward_with_chunked_prefill` | Multi-token prefill mis-aligning the causal mask |
| `test_no_information_leaks_from_the_future` | A non-causal model that looks great in training and cannot generate |
| `test_incremental_training_matches_naive_reference` | The optimised BPE trainer drifting from the definition of BPE |
| `test_roundtrip_survives_text_the_corpus_never_contained` | Mojibake on any input outside the training distribution |
| `test_model_can_overfit_a_single_batch` | A training loop that cannot learn at all |
| `test_gradient_accumulation_matches_a_single_large_step` | A silently doubled effective learning rate |
| `test_targets_are_inputs_shifted_by_one` | Training on the wrong objective |
| `test_prepare_splits_by_position_not_at_random` | A validation loss that flatters itself |
| `test_onnx_axes_are_dynamic` | An export that only works at the traced shape |
| `test_streaming_text_matches_the_non_streaming_response` | Streamed output diverging from batch output |
| `test_incremental_decoder_reassembles_split_characters` | Broken emoji/CJK in the token stream |
| `test_healthz_does_not_depend_on_the_model` | An orchestrator killing pods that are merely busy |
| `test_slots_are_released_after_a_failure` | A semaphore leak that wedges the server after N errors |
| `test_unknown_keys_are_rejected_not_ignored` | A typo'd config key silently training the wrong model |

Two tests deserve a note because they document *behaviour*, not just guard it:

- `test_tied_embeddings_bias_an_untrained_model_toward_the_input_token` — with weight
  tying, `logits = x·Eᵀ` and the residual stream at init is dominated by the input
  token's own embedding, so an untrained model scores "predict the token you were just
  given" far better than chance. That is why the initialisation check has to use
  targets drawn independently of the inputs.
- `test_repeat_kv_expands_each_head_contiguously` — GQA head expansion must repeat each
  KV head as a block, not interleave. Interleaving still trains; it just pairs every
  query head with the wrong keys.

```bash
make check        # ruff + ruff format --check + mypy + pytest
make test-cov     # coverage report
```

---

## Serving

```bash
lmops serve --checkpoint runs/small/best.pt --tokenizer data/processed/tokenizer.json
```

| Endpoint | Purpose |
| --- | --- |
| `POST /v1/generate` | Completion in a single JSON body, with token usage and latency |
| `POST /v1/generate/stream` | Server-sent events: `{"delta": "..."}` then a final `{"done": true, ...}` |
| `GET /healthz` | Liveness — **never touches the model** |
| `GET /readyz` | Readiness — checkpoint loaded, device, parameter count, training step |
| `GET /metrics` | Prometheus exposition |
| `GET /docs` | Auto-generated OpenAPI UI |

```bash
curl -X POST http://127.0.0.1:8000/v1/generate \
  -H 'content-type: application/json' \
  -d '{"prompt":"KING RICHARD:","max_new_tokens":80,"temperature":0.8,"top_p":0.95,"seed":7}'
```

Operational behaviour worth noting:

- **Liveness ≠ readiness.** Wiring both to one handler means a saturated server gets
  killed by its own orchestrator mid-request.
- **Admission control** is a bounded semaphore with a timeout, so overload is a fast
  `503` rather than an unbounded queue where everything times out and every client
  retries into the same wall.
- **`422` vs `500`.** A request the model cannot satisfy (empty prompt, `max_new_tokens`
  larger than the context) is a client error and is reported as one.
- **Blocking work runs in a thread pool**, so the event loop still answers health
  probes while the model is busy.
- **Seeded requests are reproducible** — same prompt, same seed, same bytes out.

### Metrics

`lmops_requests_total{endpoint,status}` · `lmops_request_latency_seconds` ·
`lmops_ttft_seconds` · `lmops_tokens_generated_total` · `lmops_tokens_prompt_total` ·
`lmops_requests_in_flight` · `lmops_requests_rejected_total` · `lmops_model_info`

Time-to-first-token and total latency are separate histograms because a generation SLO
is written against both, and their buckets are sized for this workload rather than
copied from a web-framework default.

---

## Deployment

```bash
make docker                                  # multi-stage build, CPU-only torch, non-root
docker compose -f docker/docker-compose.yml up   # API + Prometheus
```

The image installs torch from the CPU wheel index; the default PyPI wheel drags in
several gigabytes of CUDA libraries a CPU serving container never loads. The runtime
stage copies only the virtualenv and the package — build tools, tests and the raw
corpus never ship.

[`deploy/k8s/`](deploy/k8s) holds a reference Deployment (separate startup/liveness/
readiness probes, non-root, read-only root filesystem, `OMP_NUM_THREADS` pinned inside
the CPU limit) and an HPA that scales on **`lmops_requests_in_flight` rather than CPU** —
a pod at 100% CPU serving one long request looks identical to one serving eight short
requests, but only the second is saturated.

---

## CI

Every push runs three jobs ([`.github/workflows/ci.yml`](.github/workflows/ci.yml)):

1. **quality** — `ruff check`, `ruff format --check`, `mypy`, `pytest` with coverage, on
   Python 3.10 and 3.12.
2. **pipeline** — the real thing, stage by stage against real files: prepare → train →
   eval → sample → export (with parity assertions) → bench → **resume from checkpoint** →
   serve, then probe `/healthz`, `/readyz`, `/v1/generate`, `/metrics`, and assert that a
   malformed request returns `422`.
3. **docker** — build the image, start it with **no model mounted**, and assert
   `/healthz` is `200` while `/readyz` is `503`.

---

## Project layout

```
src/lmops/
├── config.py            typed dataclass configs, YAML merge, unknown keys rejected
├── cli.py               prepare · train · eval · sample · export · bench · serve · info
├── model/
│   ├── layers.py        RMSNorm · RoPE · SwiGLU
│   ├── attention.py     causal GQA + KVCache + offset mask
│   └── transformer.py   blocks, LM head, cached sampling (top-k/top-p/penalty)
├── data/
│   ├── tokenizer.py     byte-level BPE, incremental pair counts
│   ├── dataset.py       memmapped uint16 shards, batching
│   └── prepare.py       corpus → tokenizer + shards (positional split)
├── training/
│   ├── optimizer.py     AdamW param groups, cosine+warmup schedule
│   ├── checkpoint.py    atomic writes, weights_only-safe payloads
│   └── trainer.py       AMP, accumulation, clipping, eval, resume
├── evaluation/
│   ├── metrics.py       perplexity, bits-per-byte, top-1 accuracy
│   ├── harness.py       deterministic checkpoint scoring
│   └── benchmark.py     TTFT, decode p50/p95, cache vs no-cache
├── export/graph.py      TorchScript + ONNX + parity checks + int8 quantisation
└── serving/
    ├── engine.py        incremental UTF-8 decode, admission control
    ├── app.py           FastAPI routes, SSE, health, metrics
    ├── schemas.py       request/response validation
    └── observability.py Prometheus collectors
```

---

## Limitations

Stated because the boundary is more useful than pretending there isn't one:

- **Single-process training.** No DDP/FSDP. Untested distributed code is worse than none.
- **No continuous batching.** Requests are served one per slot. Real throughput serving
  needs a paged KV cache and a scheduler that admits sequences into a running batch.
- **Export covers scoring only.** Cached decode is not exported; the reason is in
  [`docs/architecture.md`](docs/architecture.md#7-export).
- **Base model only.** No instruction tuning, no alignment.
- **The reference model is small** — sized to train on a laptop CPU in under an hour,
  because a baseline nobody can rerun is not a baseline. `configs/medium.yaml` is the
  starting point for a GPU.

---

## Documentation

- [`docs/architecture.md`](docs/architecture.md) — every design decision and its reason
- [`docs/benchmarks.md`](docs/benchmarks.md) — measured results and how to reproduce them
- [`docs/model_card.md`](docs/model_card.md) — intended use, training data, limitations

## License

[MIT](LICENSE). The bundled corpus is public domain — see [`data/raw/README.md`](data/raw/README.md).
