"""Inference latency and throughput measurement.

Reports the two numbers that actually govern a serving SLO -- time-to-first-token
(dominated by prefill) and steady-state decode throughput -- rather than a single
end-to-end figure that hides which half is slow.

Also measures decoding *without* the KV cache, which turns the cache from an
architectural claim into a measured speedup you can put in a README.
"""

from __future__ import annotations

import statistics
import time
from dataclasses import asdict, dataclass

import torch

from ..model import LanguageModel


@dataclass
class BenchmarkResult:
    """One benchmark configuration and its measured latencies (milliseconds)."""

    label: str
    batch_size: int
    prompt_tokens: int
    generated_tokens: int
    runs: int
    ttft_ms_p50: float
    decode_ms_p50: float
    decode_ms_p95: float
    tokens_per_sec: float
    total_ms_p50: float

    def to_dict(self) -> dict:
        return asdict(self)

    def render(self) -> str:
        return (
            f"{self.label:<22} batch={self.batch_size:<3} "
            f"ttft={self.ttft_ms_p50:7.2f}ms  "
            f"decode_p50={self.decode_ms_p50:6.2f}ms  "
            f"decode_p95={self.decode_ms_p95:6.2f}ms  "
            f"{self.tokens_per_sec:8.1f} tok/s"
        )


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    if len(values) == 1:
        return values[0]
    ordered = sorted(values)
    idx = min(len(ordered) - 1, round(q * (len(ordered) - 1)))
    return ordered[idx]


@torch.inference_mode()
def benchmark_generation(
    model: LanguageModel,
    prompt_tokens: int = 32,
    generated_tokens: int = 64,
    batch_size: int = 1,
    runs: int = 5,
    warmup: int = 1,
    use_cache: bool = True,
    device: torch.device | str = "cpu",
    seed: int = 0,
    label: str | None = None,
) -> BenchmarkResult:
    """Time prefill and per-token decode for a synthetic prompt."""
    model.eval().to(device)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    prompt = torch.randint(
        0, model.config.vocab_size, (batch_size, prompt_tokens), generator=generator
    ).to(device)

    ttfts: list[float] = []
    decode_steps: list[float] = []
    totals: list[float] = []

    for run in range(runs + warmup):
        recording = run >= warmup
        start = time.perf_counter()
        first_token_at: float | None = None
        step_times: list[float] = []
        last = start

        if use_cache:
            for _ in model.stream(prompt, generated_tokens, temperature=0.0):
                now = time.perf_counter()
                if first_token_at is None:
                    first_token_at = now
                else:
                    step_times.append((now - last) * 1000.0)
                last = now
        else:
            sequence = prompt
            for _ in range(generated_tokens):
                logits, _ = model(sequence, last_only=True)
                nxt = logits[:, -1, :].argmax(dim=-1, keepdim=True)
                sequence = torch.cat([sequence, nxt], dim=1)
                now = time.perf_counter()
                if first_token_at is None:
                    first_token_at = now
                else:
                    step_times.append((now - last) * 1000.0)
                last = now

        end = time.perf_counter()
        if recording and first_token_at is not None:
            ttfts.append((first_token_at - start) * 1000.0)
            decode_steps.extend(step_times)
            totals.append((end - start) * 1000.0)

    total_p50 = statistics.median(totals) if totals else float("nan")
    produced = batch_size * generated_tokens
    tok_per_s = produced / (total_p50 / 1000.0) if total_p50 and total_p50 == total_p50 else 0.0

    return BenchmarkResult(
        label=label or ("kv-cache" if use_cache else "no-cache"),
        batch_size=batch_size,
        prompt_tokens=prompt_tokens,
        generated_tokens=generated_tokens,
        runs=runs,
        ttft_ms_p50=statistics.median(ttfts) if ttfts else float("nan"),
        decode_ms_p50=_percentile(decode_steps, 0.50),
        decode_ms_p95=_percentile(decode_steps, 0.95),
        tokens_per_sec=tok_per_s,
        total_ms_p50=total_p50,
    )


def compare_cache(
    model: LanguageModel,
    prompt_tokens: int = 32,
    generated_tokens: int = 64,
    batch_size: int = 1,
    runs: int = 3,
    device: torch.device | str = "cpu",
) -> list[BenchmarkResult]:
    """Benchmark cached vs uncached decoding under identical settings."""
    return [
        benchmark_generation(
            model,
            prompt_tokens=prompt_tokens,
            generated_tokens=generated_tokens,
            batch_size=batch_size,
            runs=runs,
            use_cache=use_cache,
            device=device,
        )
        for use_cache in (True, False)
    ]
