"""Memory-mapped token shards and batch sampling.

Tokenised corpora are stored as flat ``uint16`` binaries and read with ``np.memmap``,
so start-up cost is independent of corpus size and several worker processes can share
one page cache. Batches are contiguous random windows, which is the standard sampling
scheme for causal LM pre-training.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

TOKEN_DTYPE = np.uint16
MAX_VOCAB_FOR_DTYPE = np.iinfo(TOKEN_DTYPE).max + 1


def write_shard(tokens: list[int] | np.ndarray, path: str | Path) -> int:
    """Write token ids to ``path`` as a flat ``uint16`` binary. Returns the count."""
    array = np.asarray(tokens, dtype=np.int64)
    if array.size and int(array.max()) >= MAX_VOCAB_FOR_DTYPE:
        raise ValueError(
            f"token id {int(array.max())} does not fit in {TOKEN_DTYPE.__name__}; "
            f"shards support vocabularies up to {MAX_VOCAB_FOR_DTYPE}"
        )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    array.astype(TOKEN_DTYPE).tofile(path)
    return int(array.size)


class TokenDataset:
    """A single memory-mapped ``uint16`` token shard."""

    def __init__(self, path: str | Path, seq_len: int) -> None:
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(
                f"{self.path} not found -- run `lmops prepare` to build the token shards"
            )
        self.seq_len = seq_len
        self.tokens = np.memmap(self.path, dtype=TOKEN_DTYPE, mode="r")
        if len(self.tokens) < seq_len + 1:
            raise ValueError(
                f"{self.path} holds {len(self.tokens)} tokens, which is fewer than the "
                f"seq_len + 1 = {seq_len + 1} needed for a single training window"
            )

    def __len__(self) -> int:
        """Number of distinct ``seq_len``-token windows with a next-token target."""
        return len(self.tokens) - self.seq_len

    def get_batch(
        self,
        batch_size: int,
        device: torch.device | str = "cpu",
        generator: torch.Generator | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Draw a random batch of ``(inputs, targets)``, each ``(batch_size, seq_len)``."""
        starts = torch.randint(len(self), (batch_size,), generator=generator, device="cpu").tolist()
        # int64 copy first: torch has no uint16, and the model needs int64 indices anyway.
        windows = np.stack([self.tokens[s : s + self.seq_len + 1].astype(np.int64) for s in starts])
        batch = torch.from_numpy(windows)
        inputs = batch[:, :-1].contiguous()
        targets = batch[:, 1:].contiguous()
        if str(device) != "cpu":
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
        return inputs, targets

    def iter_sequential(
        self, batch_size: int, device: torch.device | str = "cpu"
    ) -> Iterator[tuple[Tensor, Tensor]]:
        """Yield non-overlapping windows in order -- used for deterministic evaluation."""
        stride = self.seq_len
        starts = list(range(0, len(self) - stride, stride))
        for i in range(0, len(starts) - batch_size + 1, batch_size):
            chunk = starts[i : i + batch_size]
            windows = np.stack(
                [self.tokens[s : s + self.seq_len + 1].astype(np.int64) for s in chunk]
            )
            batch = torch.from_numpy(windows).to(device)
            yield batch[:, :-1].contiguous(), batch[:, 1:].contiguous()


class DataModule:
    """Holds the train/val shards described by a ``meta.json`` in ``data_dir``."""

    def __init__(self, data_dir: str | Path, seq_len: int) -> None:
        self.data_dir = Path(data_dir)
        self.seq_len = seq_len
        self.train = TokenDataset(self.data_dir / "train.bin", seq_len)
        self.val = TokenDataset(self.data_dir / "val.bin", seq_len)
        self.meta = self._load_meta()

    def _load_meta(self) -> dict:
        meta_path = self.data_dir / "meta.json"
        if not meta_path.exists():
            return {}
        with open(meta_path, encoding="utf-8") as fh:
            return json.load(fh)

    @property
    def vocab_size(self) -> int | None:
        size = self.meta.get("vocab_size")
        return int(size) if size is not None else None

    def split(self, name: str) -> TokenDataset:
        if name == "train":
            return self.train
        if name in ("val", "validation"):
            return self.val
        raise ValueError(f"unknown split {name!r}; expected 'train' or 'val'")
