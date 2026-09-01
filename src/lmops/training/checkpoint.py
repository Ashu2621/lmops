"""Checkpoint writing and loading.

Checkpoints store the model config as a plain dict rather than a pickled dataclass.
That keeps them loadable under ``torch.load(..., weights_only=True)``, which is both
the PyTorch >= 2.6 default and the only safe way to open a checkpoint you did not
produce yourself.

Writes go to a temporary file and are then atomically renamed, so an interrupted save
cannot leave a half-written ``best.pt`` where the previous good one used to be.
"""

from __future__ import annotations

import os
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
from torch import nn

from ..config import Config, ModelConfig
from ..model import LanguageModel

CHECKPOINT_VERSION = 1


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    step: int,
    config: Config,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: Any | None = None,
    metrics: dict[str, float] | None = None,
    rng_state: dict[str, Any] | None = None,
) -> Path:
    """Atomically write a checkpoint and return its path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload: dict[str, Any] = {
        "version": CHECKPOINT_VERSION,
        "step": step,
        "model_state": model.state_dict(),
        "model_config": asdict(config.model),
        "config": config.to_dict(),
        "metrics": metrics or {},
    }
    if optimizer is not None:
        payload["optimizer_state"] = optimizer.state_dict()
    if scaler is not None and getattr(scaler, "is_enabled", lambda: False)():
        payload["scaler_state"] = scaler.state_dict()
    if rng_state is not None:
        payload["rng_state"] = rng_state

    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)
    return path


def load_checkpoint(path: str | Path, map_location: str = "cpu") -> dict[str, Any]:
    """Load a checkpoint dict, preferring the safe ``weights_only`` path."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"checkpoint not found: {path}")
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except (TypeError, RuntimeError, AttributeError):
        # Older torch has no `weights_only`; older checkpoints may hold pickled
        # objects. Fall back only for files the caller already chose to open.
        return torch.load(path, map_location=map_location)


def build_model_from_checkpoint(
    path: str | Path, device: str | torch.device = "cpu"
) -> tuple[LanguageModel, dict[str, Any]]:
    """Reconstruct the model described by a checkpoint and load its weights."""
    payload = load_checkpoint(path, map_location=str(device))
    model_config = ModelConfig(**payload["model_config"])
    model = LanguageModel(model_config)
    model.load_state_dict(payload["model_state"])
    model.to(device)
    model.eval()
    meta = {
        "step": payload.get("step", 0),
        "metrics": payload.get("metrics", {}),
        "config": payload.get("config", {}),
    }
    return model, meta


def capture_rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {"torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict[str, Any]) -> None:
    if "torch" in state:
        torch.set_rng_state(state["torch"].cpu().to(torch.uint8))
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])
