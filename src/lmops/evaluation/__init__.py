"""Offline evaluation: quality metrics and inference benchmarks."""

from .benchmark import BenchmarkResult, benchmark_generation, compare_cache
from .harness import EvalReport, evaluate_checkpoint, sample_text, write_report
from .metrics import Accumulator, bits_per_byte, perplexity

__all__ = [
    "Accumulator",
    "BenchmarkResult",
    "EvalReport",
    "benchmark_generation",
    "bits_per_byte",
    "compare_cache",
    "evaluate_checkpoint",
    "perplexity",
    "sample_text",
    "write_report",
]
