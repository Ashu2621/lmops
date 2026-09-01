# Benchmarks

Every number on this page was produced by the commands shown beside it, on the hardware
described below, from artifacts committed to `runs/`. Regenerate the tables with:

```bash
python scripts/report.py runs/small --eval runs/eval.json --bench runs/bench.json
```

**Hardware.** A 6-core / 12-thread laptop CPU (`torch.get_num_threads() == 6`),
Windows 11, PyTorch 2.8.0+cpu. **No GPU was used at any point.** The machine was not
otherwise idle during training, so throughput figures are conservative.

---

## 1. Training

`configs/small.yaml` — 3,475,712 parameters, 4 layers, `d_model` 256, 8 heads,
192-token context, 1024-token byte-level BPE vocabulary.

```bash
lmops prepare --input data/raw/tinyshakespeare.txt --out data/processed --vocab-size 1024
lmops train   --config configs/small.yaml
```

| step | train loss | val loss | val perplexity |
| ---: | ---: | ---: | ---: |
| 150 | 4.0832 | 4.0333 | 56.45 |
| 300 | 3.6562 | 3.7016 | 40.51 |
| 450 | 3.1255 | 3.4874 | 32.70 |
| 600 | 3.1651 | 3.3701 | 29.08 |
| 750 | 2.8883 | 3.3060 | 27.28 |
| 900 | 2.7405 | 3.2729 | 26.39 |
| 1050 | 2.7765 | 3.2516 | 25.83 |
| 1200 | 2.5926 | **3.2360** | **25.43** |

- **Wall clock**: ≈39 minutes for 1200 steps.
- **Throughput**: median **1,739 tokens/sec**, peak 2,099 tokens/sec.
- **Tokens seen**: 3,686,400 (≈8.9 epochs over the 415,532-token training shard).
- Train loss falls from 6.09 to 2.59; validation loss improves **monotonically** at
  every evaluation, so the run was still improving when it stopped rather than
  overfitting. The widening train/val gap (2.59 vs 3.24) is the beginning of
  memorisation on a 0.4M-token corpus and is the reason `best.pt` tracking exists.

### Tokenizer

| metric | value |
| --- | ---: |
| vocabulary | 1,024 (256 byte tokens + 767 merges + 1 special) |
| merges learned from | first 500,000 characters |
| compression | **2.40 characters/token** |
| training shard | 415,532 tokens |
| validation shard | 50,080 tokens |

---

## 2. Evaluation

```bash
lmops eval --checkpoint runs/small/best.pt --data data/processed --report runs/eval.json
```

Scored on the held-out validation shard with **non-overlapping sequential windows**, so
repeating the command reproduces these numbers exactly.

| metric | value |
| --- | ---: |
| parameters | 3,475,712 |
| eval tokens | 49,152 (109,561 UTF-8 bytes) |
| loss (nats/token) | 3.2360 |
| perplexity | 25.432 |
| **bits per byte** | **2.0944** |
| top-1 next-token accuracy | 33.45% |

**Read bits-per-byte, not perplexity, when comparing across tokenizers.** A vocabulary
that packs more characters into each token earns a lower perplexity for free; bits per
byte normalises that away. A third of next tokens predicted exactly, from 3.5M
parameters and 39 minutes of laptop CPU, is the honest headline here.

### Sample output

`lmops sample --checkpoint runs/small/best.pt --prompt "KING RICHARD:" --tokens 160 --seed 7`

```
KING RICHARD:
Even so favour! thou art not a cords;
Not dost thou wilt not a parade meaning to me.

NORFOLK:
O, then thy name is banished! Thou hast subjectd!

JULIET:
Three day thou wouldst be talk'd in thee?

JULIET:
Madam, what is thy father?

ROMEO:
'Tis that thy love is dead, that thy daughter's dead.

LADY CAPULET:
I should bend your grace as I speak.
```

The model has learned speaker headings, act structure, verse line breaks, and Early
Modern English morphology. It has not learned to mean anything — see the
[model card](model_card.md#limitations).

---

## 3. Export parity

```bash
lmops export --checkpoint runs/small/best.pt --out artifacts --seq-len 64
```

Both graphs are traced at `(batch 1, seq 64)` and then **verified at shapes they were
never traced at** — TorchScript at `(2, 24)`, ONNX at `(2, 24)` — against the eager
model.

| format | artifact | size | max abs deviation from eager | tolerance | verdict |
| --- | --- | ---: | ---: | ---: | --- |
| TorchScript | `artifacts/model.pt` | 13.9 MB | **0.000e+00** | 1e-4 | PASS |
| ONNX (opset 17) | `artifacts/model.onnx` | 15.1 MB | **1.907e-05** | 1e-3 | PASS |

TorchScript reproduces eager output bit-for-bit. ONNX does not, and is not expected to:
onnxruntime is free to fuse and reassociate floating-point operations, so the tolerance
is looser by design rather than by concession.

---

## 4. Inference latency

```bash
lmops bench --checkpoint runs/small/best.pt --report runs/bench.json
```

Five timed runs after one warm-up, single process, CPU. TTFT is prefill; the per-token
figures are steady-state decode.

### The KV cache, measured

| workload | decoding | TTFT p50 | per-token p50 | per-token p95 | throughput |
| --- | --- | ---: | ---: | ---: | ---: |
| 32 prompt + 64 new | KV cache | 11.50 ms | **7.55 ms** | 10.62 ms | 127.4 tok/s |
| 32 prompt + 64 new | no cache | 9.49 ms | 13.30 ms | 17.82 ms | 72.9 tok/s |
| 128 prompt + 64 new | KV cache | 21.88 ms | **7.18 ms** | 9.21 ms | 133.3 tok/s |
| 128 prompt + 64 new | no cache | 17.27 ms | 20.02 ms | 27.20 ms | 48.1 tok/s |
| 32 prompt + 64 new, batch 4 | KV cache | 18.95 ms | **7.24 ms** | 9.15 ms | 532.9 tok/s |
| 32 prompt + 64 new, batch 4 | no cache | 16.97 ms | 24.15 ms | 34.94 ms | 163.7 tok/s |

| workload | end-to-end speedup |
| --- | ---: |
| 32 prompt + 64 new | **1.75×** |
| 128 prompt + 64 new | **2.77×** |
| 32 prompt + 64 new, batch 4 | **3.25×** |

Three things the table shows that a single speedup number would hide:

1. **The cache does not help time-to-first-token** — it is marginally *slower* there
   (11.50 ms vs 9.49 ms), because prefill does the same work either way plus cache
   allocation and writes. All the benefit is in decode.
2. **Cached per-token latency is flat in context length** (7.55 ms at 32 tokens of
   prompt, 7.18 ms at 128) while uncached latency grows with it (13.30 → 20.02 ms).
   That is the whole point: without a cache, every step re-attends over the entire
   sequence, so cost per token is linear in what came before.
3. **The advantage compounds with batching** (3.25× at batch 4), because the redundant
   work the cache eliminates is duplicated across every sequence in the batch.

Extrapolating from the flat-vs-linear trend, the gap keeps widening with context; 192
tokens is this model's limit, so that is where the measurement stops.

---

## 5. Reproducing all of it

```bash
pip install --index-url https://download.pytorch.org/whl/cpu "torch>=2.2"
pip install -e ".[dev,serve,export]"

make prepare        # tokenizer + shards
make train          # ~39 min on a laptop CPU
make eval           # -> runs/eval.json
make export         # -> artifacts/, with parity assertions
make bench          # -> runs/bench.json
python scripts/report.py runs/small --eval runs/eval.json --bench runs/bench.json
```

Training is seeded (`train.seed: 1337`). Run-to-run variation on identical hardware
comes from non-deterministic CPU kernel scheduling only; `train.deterministic: true`
constrains that further at some cost in speed.

### Not measured here

- **Grouped-query attention.** `configs/small-gqa.yaml` is identical to the reference
  config except that eight query heads share two KV heads, so the difference is
  attributable to GQA alone. It is a one-command ablation
  (`lmops train --config configs/small-gqa.yaml`) but it was not run for this report,
  and no numbers for it are claimed.
- **GPU training.** `configs/medium.yaml` (~11M parameters, 256-token context) is the
  starting point, but every figure above is CPU-only.
- **Quantised inference.** `quantize_dynamic_int8` is implemented and tested for
  shape and top-1 agreement, but no end-to-end latency/quality trade-off was measured.
