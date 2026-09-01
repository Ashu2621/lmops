"""Turn a run directory into the Markdown tables used in docs/benchmarks.md.

Every number in this repository's documentation is produced by this script from
artifacts on disk -- ``metrics.jsonl``, ``eval.json``, ``bench.json`` -- rather than
copied by hand. That is the whole point: a benchmark table nobody can regenerate is
a claim, not a measurement.

    python scripts/report.py runs/small --eval runs/eval.json --bench runs/bench.json

Deliberately dependency-free (stdlib only) so it runs anywhere the package does,
including inside CI containers that never installed matplotlib.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

SPARK_UNICODE = "▁▂▃▄▅▆▇█"
SPARK_ASCII = "_.-~=+*#"


def _ramp() -> str:
    """Pick a sparkline ramp the current stdout can actually encode.

    Windows consoles still default to cp1252, which cannot represent the block
    characters. Crashing a reporting script over decoration would be absurd, so it
    degrades to ASCII instead.
    """
    encoding = getattr(sys.stdout, "encoding", None) or "ascii"
    try:
        SPARK_UNICODE.encode(encoding)
    except (UnicodeEncodeError, LookupError):
        return SPARK_ASCII
    return SPARK_UNICODE


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def sparkline(values: list[float]) -> str:
    """A one-line loss curve. Not a substitute for a real plot, but it travels well."""
    if not values:
        return ""
    ramp = _ramp()
    lo, hi = min(values), max(values)
    if hi - lo < 1e-12:
        return ramp[0] * len(values)
    return "".join(ramp[min(7, int((v - lo) / (hi - lo) * 7.999))] for v in values)


def training_table(records: list[dict[str, Any]]) -> str:
    evals = [r for r in records if r.get("event") == "eval"]
    trains = [r for r in records if r.get("event") == "train"]
    if not evals:
        return "_no evaluation records found_"

    lines = [
        "| step | train loss | val loss | val perplexity |",
        "| ---: | ---: | ---: | ---: |",
    ]
    for record in evals:
        step = record["step"]
        nearest = min(trains, key=lambda r: abs(r["step"] - step), default=None)
        train_loss = f"{nearest['loss']:.4f}" if nearest else "--"
        lines.append(
            f"| {step} | {train_loss} | {record['val_loss']:.4f} | {record['val_perplexity']:.2f} |"
        )
    return "\n".join(lines)


def throughput_summary(records: list[dict[str, Any]]) -> str:
    rates = [r["tokens_per_sec"] for r in records if r.get("event") == "train"]
    if not rates:
        return ""
    rates_sorted = sorted(rates)
    median = rates_sorted[len(rates_sorted) // 2]
    return f"median training throughput: **{median:,.0f} tokens/sec**"


def eval_table(report: dict[str, Any]) -> str:
    rows = [
        ("parameters", f"{report['n_parameters']:,}"),
        ("eval tokens", f"{report['n_tokens']:,}"),
        ("loss (nats/token)", f"{report['loss']:.4f}"),
        ("perplexity", f"{report['perplexity']:.3f}"),
        ("bits per byte", f"{report['bits_per_byte']:.4f}"),
        ("top-1 next-token accuracy", f"{report['top1_accuracy'] * 100:.2f}%"),
    ]
    lines = ["| metric | value |", "| --- | ---: |"]
    lines += [f"| {name} | {value} |" for name, value in rows]
    return "\n".join(lines)


def bench_table(results: list[dict[str, Any]]) -> str:
    lines = [
        "| decoding | TTFT p50 | per-token p50 | per-token p95 | throughput |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for result in results:
        lines.append(
            f"| {result['label']} | {result['ttft_ms_p50']:.1f} ms "
            f"| {result['decode_ms_p50']:.2f} ms | {result['decode_ms_p95']:.2f} ms "
            f"| {result['tokens_per_sec']:.1f} tok/s |"
        )
    if len(results) == 2 and results[0]["total_ms_p50"] > 0:
        speedup = results[1]["total_ms_p50"] / results[0]["total_ms_p50"]
        lines.append("")
        lines.append(f"KV cache speedup: **{speedup:.2f}x**")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path, help="Directory containing metrics.jsonl")
    parser.add_argument("--eval", type=Path, default=None, help="eval.json from `lmops eval`")
    parser.add_argument("--bench", type=Path, default=None, help="bench.json from `lmops bench`")
    args = parser.parse_args()

    records = read_jsonl(args.run_dir / "metrics.jsonl")
    print(f"## Training -- `{args.run_dir}`\n")
    print(training_table(records))
    losses = [r["loss"] for r in records if r.get("event") == "train"]
    if losses:
        print(f"\ntrain loss curve: `{sparkline(losses)}`  ({losses[0]:.2f} to {losses[-1]:.2f})")
    summary = throughput_summary(records)
    if summary:
        print(f"\n{summary}")

    if args.eval and args.eval.exists():
        print("\n## Evaluation\n")
        print(eval_table(json.loads(args.eval.read_text(encoding="utf-8"))))

    if args.bench and args.bench.exists():
        print("\n## Inference latency\n")
        print(bench_table(json.loads(args.bench.read_text(encoding="utf-8"))))


if __name__ == "__main__":
    main()
