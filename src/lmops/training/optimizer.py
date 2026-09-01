"""Optimizer construction and the learning-rate schedule."""

from __future__ import annotations

import math

import torch
from torch import nn

from ..config import OptimConfig


def build_param_groups(model: nn.Module, weight_decay: float) -> list[dict]:
    """Split parameters into decayed and non-decayed groups.

    Weight decay is applied to matrices (embeddings, projections) but never to
    one-dimensional parameters -- norm gains and biases. Decaying a norm gain towards
    zero fights the normalisation it is there to perform.
    """
    decay: list[nn.Parameter] = []
    no_decay: list[nn.Parameter] = []
    seen: set[int] = set()
    for param in model.parameters():
        if not param.requires_grad or id(param) in seen:
            continue
        seen.add(id(param))
        (decay if param.dim() >= 2 else no_decay).append(param)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def build_optimizer(model: nn.Module, config: OptimConfig) -> torch.optim.Optimizer:
    """AdamW with decoupled weight decay and, where available, the fused kernel."""
    groups = build_param_groups(model, config.weight_decay)
    kwargs: dict = {
        "lr": config.lr,
        "betas": (config.beta1, config.beta2),
        "eps": config.eps,
    }
    if torch.cuda.is_available():
        kwargs["fused"] = True
    return torch.optim.AdamW(groups, **kwargs)


def lr_at_step(step: int, config: OptimConfig, max_steps: int) -> float:
    """Linear warmup into cosine decay, floored at ``min_lr_ratio * lr``.

    ``step`` is zero-based. The floor matters for long runs: decaying all the way to
    zero makes the final few hundred steps contribute almost nothing.
    """
    min_lr = config.lr * config.min_lr_ratio

    if step < config.warmup_steps:
        return config.lr * (step + 1) / max(1, config.warmup_steps)

    decay_steps = max(1, max_steps - config.warmup_steps)
    progress = (step - config.warmup_steps) / decay_steps
    progress = min(1.0, max(0.0, progress))
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + coeff * (config.lr - min_lr)


def set_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = lr
