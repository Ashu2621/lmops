"""Throughput with and without continuous batching, at several concurrency levels.

    python scripts/bench_batching.py --config configs/small.yaml --requests 32

Both modes serve the same requests (same prompts, same lengths, greedy) with the same
number allowed in flight:

* ``per-request``: what the server did before batching. Each in-flight request runs
  its own ``model.stream`` in its own thread (the old semaphore path).
* ``batched``: one scheduler thread decodes every in-flight request in one pass.

Weights are untrained: throughput does not depend on what the weights are.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from concurrent.futures import ThreadPoolExecutor

import torch

from lmops.config import load_config
from lmops.model import LanguageModel
from lmops.serving.scheduler import BatchScheduler


def per_request(model: LanguageModel, prompts: list[list[int]], new: int, concurrency: int):
    def one(ids: list[int]) -> float:
        t0 = time.perf_counter()
        for _ in model.stream(torch.tensor([ids]), new, temperature=0.0):
            pass
        return time.perf_counter() - t0

    with ThreadPoolExecutor(concurrency) as pool:
        return list(pool.map(one, prompts))


def batched(model: LanguageModel, prompts: list[list[int]], new: int, concurrency: int):
    scheduler = BatchScheduler(
        model, max_batch=concurrency, max_queue=len(prompts), queue_timeout_s=600
    )

    def one(ids: list[int]) -> float:
        t0 = time.perf_counter()
        job = scheduler.submit(
            {
                "ids": ids,
                "max_new_tokens": new,
                "temperature": 0.0,
                "top_k": None,
                "top_p": None,
                "repetition_penalty": 1.0,
                "generator": None,
            }
        )
        for _ in job.tokens():
            pass
        return time.perf_counter() - t0

    try:
        with ThreadPoolExecutor(concurrency) as pool:
            return list(pool.map(one, prompts))
    finally:
        scheduler.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/small.yaml")
    ap.add_argument("--requests", type=int, default=32)
    ap.add_argument("--prompt", type=int, default=32)
    ap.add_argument("--new", type=int, default=64)
    ap.add_argument("--concurrency", default="1,4,8,16")
    ap.add_argument("--threads", type=int, default=0, help="torch threads (0 = torch default)")
    args = ap.parse_args()

    if args.threads:
        torch.set_num_threads(args.threads)
    torch.manual_seed(0)
    config = load_config(args.config)
    config.model.dropout = 0.0
    model = LanguageModel(config.model).eval()
    gen = torch.Generator().manual_seed(1)
    prompts = [
        torch.randint(0, config.model.vocab_size, (args.prompt,), generator=gen).tolist()
        for _ in range(args.requests)
    ]
    # Warm up kernels and allocators so the first mode measured is not penalised.
    per_request(model, prompts[:2], 8, 1)

    rows = []
    for c in (int(x) for x in args.concurrency.split(",")):
        for name, fn in (("per-request", per_request), ("batched", batched)):
            t0 = time.perf_counter()
            latencies = fn(model, prompts, args.new, c)
            wall = time.perf_counter() - t0
            rows.append(
                {
                    "mode": name,
                    "concurrency": c,
                    "tokens_per_s": round(args.requests * args.new / wall, 1),
                    "request_p50_ms": round(statistics.median(latencies) * 1000),
                    "request_p95_ms": round(statistics.quantiles(latencies, n=20)[18] * 1000),
                    "wall_s": round(wall, 2),
                }
            )
            print(json.dumps(rows[-1]), flush=True)


if __name__ == "__main__":
    main()
