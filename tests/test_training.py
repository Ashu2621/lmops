"""Training loop, schedule, parameter grouping and checkpoint round-trips."""

from __future__ import annotations

import json
from itertools import pairwise
from pathlib import Path

import pytest
import torch

from lmops.config import Config, ModelConfig, OptimConfig
from lmops.model import LanguageModel
from lmops.training import (
    Trainer,
    build_model_from_checkpoint,
    build_param_groups,
    load_checkpoint,
    lr_at_step,
    resolve_dtype,
    save_checkpoint,
    set_seed,
)

# -- learning-rate schedule ---------------------------------------------------


def test_warmup_rises_linearly_to_the_peak() -> None:
    cfg = OptimConfig(lr=1e-3, warmup_steps=10, min_lr_ratio=0.1)
    assert lr_at_step(0, cfg, 100) == pytest.approx(1e-4)
    assert lr_at_step(4, cfg, 100) == pytest.approx(5e-4)
    assert lr_at_step(9, cfg, 100) == pytest.approx(1e-3)


def test_cosine_decays_to_the_floor_and_stops() -> None:
    cfg = OptimConfig(lr=1e-3, warmup_steps=10, min_lr_ratio=0.1)
    assert lr_at_step(10, cfg, 100) == pytest.approx(1e-3, rel=1e-3)
    assert lr_at_step(100, cfg, 100) == pytest.approx(1e-4)
    # Past the end of the run the schedule must clamp, not go negative.
    assert lr_at_step(500, cfg, 100) == pytest.approx(1e-4)


def test_schedule_is_monotone_after_warmup() -> None:
    cfg = OptimConfig(lr=1e-3, warmup_steps=5, min_lr_ratio=0.05)
    values = [lr_at_step(s, cfg, 50) for s in range(5, 51)]
    assert all(a >= b - 1e-12 for a, b in pairwise(values))


def test_zero_warmup_does_not_divide_by_zero() -> None:
    cfg = OptimConfig(lr=1e-3, warmup_steps=0)
    assert lr_at_step(0, cfg, 10) > 0


# -- parameter groups ---------------------------------------------------------


def test_weight_decay_skips_one_dimensional_parameters() -> None:
    model = LanguageModel(ModelConfig(vocab_size=64, n_layer=2, n_head=4, d_model=32))
    decay, no_decay = build_param_groups(model, 0.1)
    assert decay["weight_decay"] == 0.1
    assert no_decay["weight_decay"] == 0.0
    assert all(p.dim() >= 2 for p in decay["params"])
    assert all(p.dim() == 1 for p in no_decay["params"])
    # Every RMSNorm gain must land in the no-decay group.
    n_norms = sum(1 for n, _ in model.named_parameters() if n.endswith("norm.weight"))
    assert len(no_decay["params"]) == n_norms


def test_tied_parameters_are_not_counted_twice() -> None:
    """``lm_head.weight is token_emb.weight`` under tying; AdamW must see it once."""
    model = LanguageModel(
        ModelConfig(vocab_size=64, n_layer=1, n_head=2, d_model=32, tie_embeddings=True)
    )
    groups = build_param_groups(model, 0.1)
    ids = [id(p) for g in groups for p in g["params"]]
    assert len(ids) == len(set(ids))


# -- dtype / seeding ----------------------------------------------------------


def test_auto_dtype_is_float32_on_cpu() -> None:
    assert resolve_dtype("auto", torch.device("cpu")) is torch.float32


def test_explicit_dtype_is_honoured() -> None:
    assert resolve_dtype("bfloat16", torch.device("cpu")) is torch.bfloat16


def test_unknown_dtype_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown dtype"):
        resolve_dtype("float8", torch.device("cpu"))


def test_seeding_makes_initialisation_reproducible() -> None:
    def build() -> torch.Tensor:
        set_seed(123)
        return LanguageModel(
            ModelConfig(vocab_size=32, n_layer=1, n_head=2, d_model=16)
        ).token_emb.weight.clone()

    torch.testing.assert_close(build(), build())


# -- checkpoints --------------------------------------------------------------


def test_checkpoint_roundtrip_preserves_weights(tmp_path: Path) -> None:
    config = Config(model=ModelConfig(vocab_size=48, n_layer=2, n_head=4, d_model=32))
    model = LanguageModel(config.model)
    path = save_checkpoint(tmp_path / "c.pt", model, step=7, config=config, metrics={"a": 1.0})

    restored, meta = build_model_from_checkpoint(path)
    assert meta["step"] == 7
    for (name, original), (_, loaded) in zip(
        model.state_dict().items(), restored.state_dict().items(), strict=True
    ):
        torch.testing.assert_close(original, loaded, msg=name)


def test_checkpoint_loads_under_weights_only(tmp_path: Path) -> None:
    """The config is stored as a plain dict so ``weights_only=True`` can open it."""
    config = Config(model=ModelConfig(vocab_size=48, n_layer=1, n_head=2, d_model=16))
    path = save_checkpoint(tmp_path / "c.pt", LanguageModel(config.model), 1, config)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    assert isinstance(payload["model_config"], dict)
    assert payload["model_config"]["n_layer"] == 1


def test_checkpoint_write_is_atomic(tmp_path: Path) -> None:
    """No ``.tmp`` file may survive a successful save."""
    config = Config(model=ModelConfig(vocab_size=32, n_layer=1, n_head=2, d_model=16))
    save_checkpoint(tmp_path / "c.pt", LanguageModel(config.model), 1, config)
    assert (tmp_path / "c.pt").exists()
    assert not list(tmp_path.glob("*.tmp"))


def test_missing_checkpoint_raises_a_clear_error(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="checkpoint not found"):
        load_checkpoint(tmp_path / "absent.pt")


def test_restored_model_produces_identical_logits(tmp_path: Path) -> None:
    config = Config(model=ModelConfig(vocab_size=48, n_layer=2, n_head=4, d_model=32))
    model = LanguageModel(config.model).eval()
    path = save_checkpoint(tmp_path / "c.pt", model, step=1, config=config)
    restored, _ = build_model_from_checkpoint(path)

    ids = torch.randint(0, 48, (2, 9))
    with torch.no_grad():
        torch.testing.assert_close(model(ids)[0], restored(ids)[0])


# -- the loop ----------------------------------------------------------------


def test_training_reduces_loss_and_writes_metrics(tiny_config: Config) -> None:
    trainer = Trainer(tiny_config)
    state = trainer.fit()

    assert state.step == tiny_config.train.max_steps
    assert state.tokens_seen == state.step * trainer.tokens_per_step

    metrics_path = Path(tiny_config.train.out_dir) / "metrics.jsonl"
    records = [json.loads(line) for line in metrics_path.read_text(encoding="utf-8").splitlines()]
    train_losses = [r["loss"] for r in records if r["event"] == "train"]
    assert len(train_losses) >= 2
    assert train_losses[-1] < train_losses[0]

    assert (Path(tiny_config.train.out_dir) / "last.pt").exists()
    assert (Path(tiny_config.train.out_dir) / "config.yaml").exists()


def test_trainer_aligns_vocab_size_to_the_shards(tiny_config: Config) -> None:
    """A model built with the wrong vocab trains fine and fails only at decode time,
    so the trainer corrects it from the shard metadata instead."""
    tiny_config.model.vocab_size = 999
    trainer = Trainer(tiny_config)
    assert trainer.model.config.vocab_size == trainer.data.vocab_size


def test_evaluation_is_deterministic(tiny_config: Config) -> None:
    trainer = Trainer(tiny_config)
    assert trainer.evaluate("val", max_batches=2) == trainer.evaluate("val", max_batches=2)


def test_resume_restores_step_and_weights(tiny_config: Config, tmp_path: Path) -> None:
    trainer = Trainer(tiny_config)
    trainer.fit()
    checkpoint = Path(tiny_config.train.out_dir) / "last.pt"

    resumed_config = Config(
        model=tiny_config.model,
        data=tiny_config.data,
        optim=tiny_config.optim,
        train=tiny_config.train,
    )
    resumed_config.train.resume = str(checkpoint)
    resumed_config.train.out_dir = str(tmp_path / "resumed")

    resumed = Trainer(resumed_config)
    assert resumed.state.step == trainer.state.step

    ids = torch.randint(0, trainer.model.config.vocab_size, (1, 8))
    trainer.model.eval()
    resumed.model.eval()
    with torch.no_grad():
        torch.testing.assert_close(trainer.model(ids)[0], resumed.model(ids)[0])


def test_gradient_accumulation_matches_a_single_large_step(tiny_config: Config) -> None:
    """Two micro-batches at accum=2 must give the same gradient as one full batch.

    This is the invariant the ``loss / grad_accum_steps`` scaling exists to preserve;
    getting it wrong silently doubles the effective learning rate.
    """
    torch.manual_seed(0)
    model = LanguageModel(tiny_config.model)
    ids = torch.randint(0, tiny_config.model.vocab_size, (4, 16))
    targets = torch.randint(0, tiny_config.model.vocab_size, (4, 16))

    model.zero_grad(set_to_none=True)
    _, loss = model(ids, targets=targets)
    loss.backward()
    full = [p.grad.clone() for p in model.parameters()]

    model.zero_grad(set_to_none=True)
    for half in range(2):
        chunk = slice(half * 2, half * 2 + 2)
        _, part = model(ids[chunk], targets=targets[chunk])
        (part / 2).backward()
    accumulated = [p.grad.clone() for p in model.parameters()]

    for a, b in zip(full, accumulated, strict=True):
        torch.testing.assert_close(a, b, rtol=1e-4, atol=1e-6)
