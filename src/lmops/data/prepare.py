"""Corpus -> tokenizer -> ``train.bin`` / ``val.bin``.

The split is taken by *position* rather than at random: a language model evaluated on
validation windows drawn from the same region as its training windows will report a
perplexity that flatters it, because neighbouring windows overlap heavily.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .dataset import write_shard
from .tokenizer import DEFAULT_SPECIAL_TOKENS, ByteBPETokenizer


@dataclass
class PrepareResult:
    """Summary of a preparation run, mirrored into ``meta.json``."""

    vocab_size: int
    train_tokens: int
    val_tokens: int
    total_chars: int
    tokenizer_path: str
    data_dir: str

    @property
    def compression_ratio(self) -> float:
        """Characters per token -- the headline number for a tokenizer's efficiency."""
        total = self.train_tokens + self.val_tokens
        return self.total_chars / total if total else 0.0

    def to_dict(self) -> dict:
        return {
            "vocab_size": self.vocab_size,
            "train_tokens": self.train_tokens,
            "val_tokens": self.val_tokens,
            "total_chars": self.total_chars,
            "chars_per_token": round(self.compression_ratio, 4),
            "tokenizer_path": self.tokenizer_path,
        }


def prepare_corpus(
    input_path: str | Path,
    data_dir: str | Path,
    vocab_size: int = 1024,
    val_fraction: float = 0.1,
    tokenizer_train_chars: int = 500_000,
    progress: bool = False,
) -> PrepareResult:
    """Train a tokenizer on ``input_path`` and write token shards into ``data_dir``.

    Args:
        vocab_size: total vocabulary, including the 256 byte tokens and specials.
        val_fraction: tail fraction of the corpus held out for validation.
        tokenizer_train_chars: cap on the prefix used for *learning merges*. The full
            corpus is still encoded; capping only bounds the cost of BPE training,
            which is superlinear in corpus size.
    """
    input_path = Path(input_path)
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    text = input_path.read_text(encoding="utf-8")
    if not text:
        raise ValueError(f"{input_path} is empty")

    if progress:
        print(f"corpus: {len(text):,} chars from {input_path}")

    sample = text[:tokenizer_train_chars]
    tokenizer = ByteBPETokenizer.train(
        sample, vocab_size=vocab_size, special_tokens=DEFAULT_SPECIAL_TOKENS, progress=progress
    )
    tokenizer_path = data_dir / "tokenizer.json"
    tokenizer.save(tokenizer_path)
    if progress:
        print(f"tokenizer: {tokenizer.vocab_size} tokens -> {tokenizer_path}")

    split_at = int(len(text) * (1.0 - val_fraction))
    train_ids = tokenizer.encode(text[:split_at])
    val_ids = tokenizer.encode(text[split_at:])

    n_train = write_shard(train_ids, data_dir / "train.bin")
    n_val = write_shard(val_ids, data_dir / "val.bin")

    result = PrepareResult(
        vocab_size=tokenizer.vocab_size,
        train_tokens=n_train,
        val_tokens=n_val,
        total_chars=len(text),
        tokenizer_path=str(tokenizer_path),
        data_dir=str(data_dir),
    )
    with open(data_dir / "meta.json", "w", encoding="utf-8") as fh:
        json.dump(result.to_dict(), fh, indent=2)

    if progress:
        print(
            f"shards: train={n_train:,} val={n_val:,} tokens "
            f"({result.compression_ratio:.2f} chars/token)"
        )
    return result
