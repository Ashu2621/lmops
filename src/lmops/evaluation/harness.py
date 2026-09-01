"""Offline evaluation of a checkpoint against a held-out shard."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from ..data import ByteBPETokenizer, TokenDataset
from ..model import LanguageModel
from ..training.checkpoint import build_model_from_checkpoint
from ..training.trainer import resolve_device
from .metrics import Accumulator, batch_nll_and_accuracy


@dataclass
class EvalReport:
    """Everything a reviewer needs to compare two checkpoints."""

    checkpoint: str
    split: str
    step: int
    n_tokens: int
    n_bytes: int
    loss: float
    perplexity: float
    bits_per_byte: float
    top1_accuracy: float
    n_parameters: int

    def to_dict(self) -> dict:
        return asdict(self)

    def render(self) -> str:
        return (
            f"checkpoint : {self.checkpoint}\n"
            f"split      : {self.split} ({self.n_tokens:,} tokens / {self.n_bytes:,} bytes)\n"
            f"step       : {self.step:,}\n"
            f"params     : {self.n_parameters / 1e6:.2f}M\n"
            f"loss       : {self.loss:.4f}\n"
            f"perplexity : {self.perplexity:.3f}\n"
            f"bits/byte  : {self.bits_per_byte:.4f}\n"
            f"top-1 acc  : {self.top1_accuracy * 100:.2f}%"
        )


@torch.no_grad()
def evaluate_checkpoint(
    checkpoint: str | Path,
    data_dir: str | Path,
    split: str = "val",
    batch_size: int = 8,
    max_batches: int | None = None,
    device: str = "auto",
    tokenizer_path: str | Path | None = None,
) -> EvalReport:
    """Score ``checkpoint`` on ``split`` and return a populated :class:`EvalReport`.

    Windows are non-overlapping and taken in corpus order, so the number is
    reproducible: two runs of this function on the same checkpoint agree exactly.
    """
    dev = resolve_device(device)
    model, meta = build_model_from_checkpoint(checkpoint, device=dev)
    seq_len = model.config.max_seq_len

    data_dir = Path(data_dir)
    shard = TokenDataset(data_dir / f"{'train' if split == 'train' else 'val'}.bin", seq_len)

    tokenizer = None
    tok_path = Path(tokenizer_path) if tokenizer_path else data_dir / "tokenizer.json"
    if tok_path.exists():
        tokenizer = ByteBPETokenizer.load(tok_path)

    acc = Accumulator()
    for i, (inputs, targets) in enumerate(shard.iter_sequential(batch_size, dev)):
        logits, _ = model(inputs)
        sum_nll, n_tokens, correct = batch_nll_and_accuracy(logits, targets)
        n_bytes = _decoded_bytes(tokenizer, targets) if tokenizer is not None else 0
        acc.update(sum_nll, n_tokens, n_bytes, correct)
        if max_batches is not None and i + 1 >= max_batches:
            break

    return EvalReport(
        checkpoint=str(checkpoint),
        split=split,
        step=int(meta.get("step", 0)),
        n_tokens=acc.total_tokens,
        n_bytes=acc.total_bytes,
        loss=acc.mean_nll,
        perplexity=acc.perplexity,
        bits_per_byte=acc.bits_per_byte,
        top1_accuracy=acc.top1_accuracy,
        n_parameters=model.num_parameters(),
    )


def _decoded_bytes(tokenizer: ByteBPETokenizer, targets: torch.Tensor) -> int:
    """UTF-8 byte length of the target tokens, for the bits-per-byte denominator."""
    total = 0
    for row in targets.tolist():
        total += len(tokenizer.decode(row).encode("utf-8"))
    return total


@torch.no_grad()
def sample_text(
    model: LanguageModel,
    tokenizer: ByteBPETokenizer,
    prompt: str,
    max_new_tokens: int = 128,
    temperature: float = 0.8,
    top_k: int | None = 40,
    top_p: float | None = 0.95,
    seed: int | None = None,
    device: torch.device | str = "cpu",
) -> str:
    """Generate a continuation of ``prompt`` and return prompt + continuation."""
    ids = tokenizer.encode(prompt) or [0]
    idx = torch.tensor([ids], dtype=torch.long, device=device)
    generator = None
    if seed is not None:
        generator = torch.Generator(device=str(device)).manual_seed(seed)
    out = model.generate(
        idx,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        generator=generator,
    )
    return tokenizer.decode(out[0].tolist())


def write_report(report: EvalReport, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(report.to_dict(), fh, indent=2)
    return path
