"""Shared fixtures.

The corpus is generated rather than downloaded so the whole suite is hermetic: no
network, no dependency on ``data/`` having been prepared, and identical bytes on every
machine. It is still *structured* text (a small closed vocabulary with punctuation and
line breaks), because BPE trained on random bytes learns nothing and would make the
tokenizer tests vacuous.
"""

from __future__ import annotations

import random
from pathlib import Path

import pytest
import torch

from lmops.config import Config, DataConfig, ModelConfig, OptimConfig, TrainConfig
from lmops.data import ByteBPETokenizer, prepare_corpus
from lmops.model import LanguageModel
from lmops.training.checkpoint import save_checkpoint

WORDS = [
    "the",
    "quick",
    "brown",
    "fox",
    "jumps",
    "over",
    "lazy",
    "dog",
    "and",
    "then",
    "runs",
    "into",
    "forest",
    "where",
    "river",
    "flows",
    "under",
    "old",
    "stone",
    "bridge",
    "while",
    "birds",
    "sing",
    "morning",
    "light",
    "falls",
    "through",
]


def _synthetic_corpus(n_lines: int = 4_000, seed: int = 7) -> str:
    rng = random.Random(seed)
    lines = []
    for i in range(n_lines):
        length = rng.randint(4, 12)
        sentence = " ".join(rng.choice(WORDS) for _ in range(length))
        lines.append(sentence.capitalize() + rng.choice([".", ",", "!", "?"]))
        if i % 7 == 0:
            lines.append("")
    return "\n".join(lines)


@pytest.fixture(scope="session")
def corpus_text() -> str:
    return _synthetic_corpus()


@pytest.fixture(scope="session")
def prepared_dir(tmp_path_factory: pytest.TempPathFactory, corpus_text: str) -> Path:
    """A fully prepared data directory: tokenizer.json, train.bin, val.bin, meta.json."""
    root = tmp_path_factory.mktemp("data")
    raw = root / "corpus.txt"
    raw.write_text(corpus_text, encoding="utf-8")
    out = root / "processed"
    prepare_corpus(raw, out, vocab_size=320, val_fraction=0.1, tokenizer_train_chars=60_000)
    return out


@pytest.fixture(scope="session")
def tokenizer(prepared_dir: Path) -> ByteBPETokenizer:
    return ByteBPETokenizer.load(prepared_dir / "tokenizer.json")


@pytest.fixture
def tiny_model_config() -> ModelConfig:
    """Two layers, grouped-query attention, deliberately small but not degenerate."""
    return ModelConfig(
        vocab_size=97,
        n_layer=2,
        n_head=4,
        n_kv_head=2,
        d_model=64,
        max_seq_len=64,
        dropout=0.0,
    )


@pytest.fixture
def tiny_model(tiny_model_config: ModelConfig) -> LanguageModel:
    torch.manual_seed(0)
    model = LanguageModel(tiny_model_config)
    model.eval()
    return model


@pytest.fixture
def tiny_config(prepared_dir: Path, tmp_path: Path, tokenizer: ByteBPETokenizer) -> Config:
    """A complete :class:`Config` wired to the prepared fixture data."""
    return Config(
        model=ModelConfig(
            vocab_size=tokenizer.vocab_size,
            n_layer=2,
            n_head=4,
            n_kv_head=2,
            d_model=64,
            max_seq_len=32,
            dropout=0.0,
        ),
        data=DataConfig(
            data_dir=str(prepared_dir),
            tokenizer_path=str(prepared_dir / "tokenizer.json"),
            batch_size=4,
            seq_len=32,
        ),
        optim=OptimConfig(lr=1e-3, warmup_steps=2),
        train=TrainConfig(
            max_steps=6,
            eval_interval=6,
            eval_batches=2,
            log_interval=3,
            checkpoint_interval=6,
            device="cpu",
            dtype="float32",
            out_dir=str(tmp_path / "run"),
        ),
    )


@pytest.fixture
def served_checkpoint(
    tmp_path: Path, prepared_dir: Path, tokenizer: ByteBPETokenizer
) -> tuple[Path, Path]:
    """A checkpoint whose vocab matches the fixture tokenizer, plus that tokenizer's path.

    The vocab-size match matters: :class:`~lmops.serving.engine.GenerationEngine`
    refuses to start when the checkpoint and tokenizer disagree, which is exactly the
    mismatch this fixture must not accidentally introduce.
    """
    torch.manual_seed(0)
    config = Config(
        model=ModelConfig(
            vocab_size=tokenizer.vocab_size,
            n_layer=2,
            n_head=4,
            n_kv_head=2,
            d_model=64,
            max_seq_len=96,
            dropout=0.0,
        )
    )
    model = LanguageModel(config.model)
    path = tmp_path / "ckpt.pt"
    save_checkpoint(path, model, step=1, config=config, metrics={"val_loss": 1.23})
    return path, prepared_dir / "tokenizer.json"
