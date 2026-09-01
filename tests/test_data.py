"""Token shards, batching and corpus preparation."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from lmops.data import DataModule, TokenDataset, prepare_corpus, write_shard


def test_write_shard_roundtrips(tmp_path: Path) -> None:
    tokens = [0, 1, 2, 3, 65535]
    path = tmp_path / "s.bin"
    assert write_shard(tokens, path) == 5
    assert np.fromfile(path, dtype=np.uint16).tolist() == tokens


def test_write_shard_rejects_ids_that_do_not_fit(tmp_path: Path) -> None:
    """Silently wrapping a token id modulo 65536 would corrupt a corpus invisibly."""
    with pytest.raises(ValueError, match="does not fit"):
        write_shard([1, 70_000], tmp_path / "s.bin")


def test_dataset_reports_window_count(tmp_path: Path) -> None:
    write_shard(list(range(100)), tmp_path / "s.bin")
    dataset = TokenDataset(tmp_path / "s.bin", seq_len=10)
    assert len(dataset) == 90


def test_dataset_rejects_a_shard_too_small_for_one_window(tmp_path: Path) -> None:
    write_shard(list(range(5)), tmp_path / "s.bin")
    with pytest.raises(ValueError, match="fewer than"):
        TokenDataset(tmp_path / "s.bin", seq_len=10)


def test_missing_shard_names_the_fix(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="lmops prepare"):
        TokenDataset(tmp_path / "nope.bin", seq_len=4)


def test_batch_shapes_and_dtype(tmp_path: Path) -> None:
    write_shard(list(range(1_000)), tmp_path / "s.bin")
    dataset = TokenDataset(tmp_path / "s.bin", seq_len=16)
    inputs, targets = dataset.get_batch(4)
    assert inputs.shape == targets.shape == (4, 16)
    assert inputs.dtype == torch.int64


def test_targets_are_inputs_shifted_by_one(tmp_path: Path) -> None:
    """The entire training signal depends on this off-by-one being right."""
    write_shard(list(range(500)), tmp_path / "s.bin")
    dataset = TokenDataset(tmp_path / "s.bin", seq_len=8)
    inputs, targets = dataset.get_batch(3)
    torch.testing.assert_close(inputs[:, 1:], targets[:, :-1])


def test_batches_are_reproducible_under_a_seeded_generator(tmp_path: Path) -> None:
    write_shard(list(range(1_000)), tmp_path / "s.bin")
    dataset = TokenDataset(tmp_path / "s.bin", seq_len=8)
    a = dataset.get_batch(4, generator=torch.Generator().manual_seed(11))[0]
    b = dataset.get_batch(4, generator=torch.Generator().manual_seed(11))[0]
    torch.testing.assert_close(a, b)


def test_sequential_iteration_does_not_overlap(tmp_path: Path) -> None:
    write_shard(list(range(1_000)), tmp_path / "s.bin")
    dataset = TokenDataset(tmp_path / "s.bin", seq_len=10)
    firsts = [int(inputs[0, 0]) for inputs, _ in dataset.iter_sequential(2)]
    assert firsts[:3] == [0, 20, 40]
    assert firsts == sorted(firsts)


def test_prepare_writes_every_artifact(tmp_path: Path, corpus_text: str) -> None:
    raw = tmp_path / "raw.txt"
    raw.write_text(corpus_text[:40_000], encoding="utf-8")
    out = tmp_path / "processed"
    result = prepare_corpus(raw, out, vocab_size=300, tokenizer_train_chars=20_000)

    for name in ("train.bin", "val.bin", "tokenizer.json", "meta.json"):
        assert (out / name).exists(), name

    assert result.train_tokens > result.val_tokens > 0
    assert result.compression_ratio > 1.5

    meta = json.loads((out / "meta.json").read_text(encoding="utf-8"))
    assert meta["vocab_size"] == result.vocab_size


def test_prepare_splits_by_position_not_at_random(tmp_path: Path) -> None:
    """The validation shard must come from the tail of the corpus.

    A random split lets near-duplicate neighbouring windows land on both sides and
    quietly flatters the validation loss.
    """
    raw = tmp_path / "raw.txt"
    raw.write_text("A" * 9_000 + "B" * 1_000, encoding="utf-8")
    out = tmp_path / "processed"
    prepare_corpus(raw, out, vocab_size=270, val_fraction=0.1, tokenizer_train_chars=5_000)

    from lmops.data import ByteBPETokenizer

    tok = ByteBPETokenizer.load(out / "tokenizer.json")
    val = np.fromfile(out / "val.bin", dtype=np.uint16).tolist()
    assert set(tok.decode(val)) == {"B"}


def test_empty_corpus_is_rejected(tmp_path: Path) -> None:
    raw = tmp_path / "raw.txt"
    raw.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="empty"):
        prepare_corpus(raw, tmp_path / "out", vocab_size=300)


def test_datamodule_exposes_both_splits(prepared_dir: Path) -> None:
    module = DataModule(prepared_dir, seq_len=32)
    assert module.split("train") is module.train
    assert module.split("val") is module.val
    assert module.vocab_size is not None
    with pytest.raises(ValueError, match="unknown split"):
        module.split("test")
