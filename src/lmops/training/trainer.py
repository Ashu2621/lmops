"""The training loop: AMP, gradient accumulation, cosine schedule, checkpoint/resume."""

from __future__ import annotations

import json
import math
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from ..config import Config, save_config
from ..data import DataModule
from ..model import LanguageModel
from .checkpoint import (
    build_model_from_checkpoint,
    capture_rng_state,
    load_checkpoint,
    restore_rng_state,
    save_checkpoint,
)
from .optimizer import build_optimizer, lr_at_step, set_lr

_DTYPES = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}


def resolve_device(spec: str = "auto") -> torch.device:
    if spec != "auto":
        return torch.device(spec)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def resolve_dtype(spec: str, device: torch.device) -> torch.dtype:
    """Pick the autocast dtype.

    ``auto`` means bf16 on hardware that supports it natively and float32 everywhere
    else. Emulated bf16 on an older CPU is slower than float32 *and* less accurate,
    so guessing "half precision is always faster" is the wrong default.
    """
    if spec != "auto":
        if spec not in _DTYPES:
            raise ValueError(f"unknown dtype {spec!r}; expected one of {sorted(_DTYPES)}")
        return _DTYPES[spec]
    if device.type == "cuda" and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float32


def set_seed(seed: int, deterministic: bool = False) -> None:
    """Seed every RNG the training loop touches."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.benchmark = False


@dataclass
class TrainState:
    """Mutable bookkeeping carried across a run (and across a resume)."""

    step: int = 0
    best_val_loss: float = math.inf
    tokens_seen: int = 0
    history: list[dict[str, Any]] = field(default_factory=list)


class Trainer:
    """Owns the model, the optimizer and the loop that connects them to data."""

    def __init__(self, config: Config, data: DataModule | None = None) -> None:
        self.config = config
        self.device = resolve_device(config.train.device)
        self.dtype = resolve_dtype(config.train.dtype, self.device)
        set_seed(config.train.seed, config.train.deterministic)

        self.data = (
            data if data is not None else DataModule(config.data.data_dir, config.data.seq_len)
        )

        vocab_size = self.data.vocab_size
        if vocab_size is not None and vocab_size != config.model.vocab_size:
            # The shards are the source of truth: a model with the wrong vocab size
            # will happily train and only fail at decode time.
            print(
                f"[trainer] aligning model.vocab_size {config.model.vocab_size} -> {vocab_size} "
                "to match the prepared shards"
            )
            config.model.vocab_size = vocab_size

        self.model = LanguageModel(config.model).to(self.device)
        self.optimizer = build_optimizer(self.model, config.optim)
        self.scaler = torch.amp.GradScaler(self.device.type, enabled=(self.dtype is torch.float16))
        self.state = TrainState()

        self.out_dir = Path(config.train.out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_path = self.out_dir / "metrics.jsonl"

        self._compiled: nn.Module = self.model
        if config.train.compile_model:
            try:
                self._compiled = torch.compile(self.model)  # type: ignore[assignment]
            except Exception as exc:  # pragma: no cover - backend dependent
                print(f"[trainer] torch.compile unavailable ({exc}); running eager")

        if config.train.resume:
            self.resume(config.train.resume)

    # -- helpers --------------------------------------------------------------

    @property
    def tokens_per_step(self) -> int:
        cfg = self.config
        return cfg.data.batch_size * cfg.data.seq_len * cfg.train.grad_accum_steps

    def _autocast(self) -> torch.amp.autocast:
        enabled = self.dtype is not torch.float32
        return torch.amp.autocast(device_type=self.device.type, dtype=self.dtype, enabled=enabled)

    def _log(self, record: dict[str, Any]) -> None:
        self.state.history.append(record)
        with open(self.metrics_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")

    # -- evaluation -----------------------------------------------------------

    @torch.no_grad()
    def evaluate(self, split: str = "val", max_batches: int | None = None) -> float:
        """Mean cross-entropy over deterministic, non-overlapping windows."""
        dataset = self.data.split(split)
        limit = max_batches if max_batches is not None else self.config.train.eval_batches
        was_training = self.model.training
        self.model.eval()

        total, count = 0.0, 0
        for inputs, targets in dataset.iter_sequential(self.config.data.batch_size, self.device):
            with self._autocast():
                _, loss = self.model(inputs, targets)
            assert loss is not None
            total += loss.item()
            count += 1
            if count >= limit:
                break

        self.model.train(was_training)
        return total / max(1, count)

    # -- one optimizer step ---------------------------------------------------

    def train_step(self, step: int) -> tuple[float, float, float]:
        """Run ``grad_accum_steps`` micro-batches and one optimizer update."""
        cfg = self.config
        lr = lr_at_step(step, cfg.optim, cfg.train.max_steps)
        set_lr(self.optimizer, lr)

        self.optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0
        for _ in range(cfg.train.grad_accum_steps):
            inputs, targets = self.data.train.get_batch(cfg.data.batch_size, self.device)
            with self._autocast():
                _, loss = self._compiled(inputs, targets)
            assert loss is not None
            # Scale so the accumulated gradient equals the mean over the full batch.
            scaled = loss / cfg.train.grad_accum_steps
            self.scaler.scale(scaled).backward()
            total_loss += loss.item()

        grad_norm = 0.0
        if cfg.optim.grad_clip > 0:
            self.scaler.unscale_(self.optimizer)
            grad_norm = float(
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.optim.grad_clip)
            )

        self.scaler.step(self.optimizer)
        self.scaler.update()
        return total_loss / cfg.train.grad_accum_steps, lr, grad_norm

    # -- the loop -------------------------------------------------------------

    def fit(self) -> TrainState:
        cfg = self.config
        save_config(cfg, self.out_dir / "config.yaml")
        self.model.train()

        params = self.model.num_parameters()
        print(
            f"[trainer] {params / 1e6:.2f}M params | device={self.device} dtype={self.dtype} "
            f"| {self.tokens_per_step:,} tokens/step"
        )

        start_step = self.state.step
        window_start = time.perf_counter()
        window_tokens = 0

        for step in range(start_step, cfg.train.max_steps):
            loss, lr, grad_norm = self.train_step(step)
            self.state.step = step + 1
            self.state.tokens_seen += self.tokens_per_step
            window_tokens += self.tokens_per_step

            if self.state.step % cfg.train.log_interval == 0:
                elapsed = time.perf_counter() - window_start
                tok_per_s = window_tokens / elapsed if elapsed > 0 else 0.0
                self._log(
                    {
                        "step": self.state.step,
                        "event": "train",
                        "loss": round(loss, 5),
                        "lr": lr,
                        "grad_norm": round(grad_norm, 4),
                        "tokens_seen": self.state.tokens_seen,
                        "tokens_per_sec": round(tok_per_s, 1),
                    }
                )
                print(
                    f"step {self.state.step:>6} | loss {loss:.4f} | lr {lr:.2e} "
                    f"| gnorm {grad_norm:.2f} | {tok_per_s:,.0f} tok/s"
                )
                window_start, window_tokens = time.perf_counter(), 0

            if (
                self.state.step % cfg.train.eval_interval == 0
                or self.state.step == cfg.train.max_steps
            ):
                val_loss = self.evaluate("val")
                is_best = val_loss < self.state.best_val_loss
                self.state.best_val_loss = min(val_loss, self.state.best_val_loss)
                self._log(
                    {
                        "step": self.state.step,
                        "event": "eval",
                        "val_loss": round(val_loss, 5),
                        "val_perplexity": round(math.exp(min(val_loss, 20.0)), 3),
                        "best_val_loss": round(self.state.best_val_loss, 5),
                    }
                )
                print(
                    f"step {self.state.step:>6} | val_loss {val_loss:.4f} "
                    f"| ppl {math.exp(min(val_loss, 20.0)):.2f}" + ("  <- best" if is_best else "")
                )
                if is_best:
                    self.save("best.pt", val_loss=val_loss)
                window_start, window_tokens = time.perf_counter(), 0

            if (
                cfg.train.checkpoint_interval
                and self.state.step % cfg.train.checkpoint_interval == 0
            ):
                self.save("last.pt")

        self.save("last.pt")
        return self.state

    # -- persistence ----------------------------------------------------------

    def save(self, name: str, val_loss: float | None = None) -> Path:
        metrics = {"best_val_loss": self.state.best_val_loss, "tokens_seen": self.state.tokens_seen}
        if val_loss is not None:
            metrics["val_loss"] = val_loss
        return save_checkpoint(
            self.out_dir / name,
            self.model,
            step=self.state.step,
            config=self.config,
            optimizer=self.optimizer,
            scaler=self.scaler,
            metrics=metrics,
            rng_state=capture_rng_state(),
        )

    def resume(self, path: str | Path) -> None:
        """Restore weights, optimizer, scaler, step counter and RNG state."""
        payload = load_checkpoint(path, map_location=str(self.device))
        self.model.load_state_dict(payload["model_state"])
        if "optimizer_state" in payload:
            self.optimizer.load_state_dict(payload["optimizer_state"])
        if "scaler_state" in payload:
            self.scaler.load_state_dict(payload["scaler_state"])
        if "rng_state" in payload:
            restore_rng_state(payload["rng_state"])
        self.state.step = int(payload.get("step", 0))
        self.state.best_val_loss = float(payload.get("metrics", {}).get("best_val_loss", math.inf))
        self.state.tokens_seen = int(payload.get("metrics", {}).get("tokens_seen", 0))
        print(f"[trainer] resumed from {path} at step {self.state.step}")


__all__ = [
    "TrainState",
    "Trainer",
    "build_model_from_checkpoint",
    "resolve_device",
    "resolve_dtype",
    "set_seed",
]
