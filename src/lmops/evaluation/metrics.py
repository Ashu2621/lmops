"""Language-model metrics.

Perplexity is reported because everyone expects it, but it is *not* comparable
across tokenizers: a vocabulary that packs more characters into each token gets a
lower perplexity for free. Bits-per-byte normalises by the underlying UTF-8 bytes
instead, which makes two runs with different vocabularies comparable.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor


def perplexity(mean_nll: float, cap: float = 20.0) -> float:
    """``exp`` of the mean negative log-likelihood, clamped so early training does
    not overflow to ``inf`` and poison a metrics dashboard."""
    return math.exp(min(mean_nll, cap))


def bits_per_byte(total_nll_nats: float, total_bytes: int) -> float:
    """Convert a summed nat-loss into bits per UTF-8 byte."""
    if total_bytes <= 0:
        return float("nan")
    return total_nll_nats / math.log(2) / total_bytes


@dataclass
class Accumulator:
    """Running totals for a streaming evaluation pass."""

    total_nll: float = 0.0
    total_tokens: int = 0
    total_bytes: int = 0
    correct_top1: int = 0

    def update(
        self,
        sum_nll: float,
        n_tokens: int,
        n_bytes: int = 0,
        correct: int = 0,
    ) -> None:
        self.total_nll += sum_nll
        self.total_tokens += n_tokens
        self.total_bytes += n_bytes
        self.correct_top1 += correct

    @property
    def mean_nll(self) -> float:
        return self.total_nll / self.total_tokens if self.total_tokens else float("nan")

    @property
    def perplexity(self) -> float:
        return perplexity(self.mean_nll)

    @property
    def bits_per_byte(self) -> float:
        return bits_per_byte(self.total_nll, self.total_bytes)

    @property
    def top1_accuracy(self) -> float:
        return self.correct_top1 / self.total_tokens if self.total_tokens else float("nan")


def batch_nll_and_accuracy(logits: Tensor, targets: Tensor) -> tuple[float, int, int]:
    """Return ``(summed_nll_nats, n_tokens, n_correct_top1)`` for one batch.

    The loss is computed in float32 regardless of the autocast dtype -- summing many
    half-precision log-probabilities loses accuracy fast enough to move the third
    decimal place of a perplexity number.
    """
    flat_logits = logits.reshape(-1, logits.size(-1)).float()
    flat_targets = targets.reshape(-1)
    nll = torch.nn.functional.cross_entropy(flat_logits, flat_targets, reduction="sum")
    correct = int((flat_logits.argmax(dim=-1) == flat_targets).sum().item())
    return float(nll.item()), int(flat_targets.numel()), correct
