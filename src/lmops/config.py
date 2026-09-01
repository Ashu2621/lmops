"""Typed configuration objects for every stage of the pipeline.

Configs are plain dataclasses so they survive ``torch.save`` round-trips without
dragging a validation library into the checkpoint. YAML files are merged onto the
dataclass defaults, and unknown keys are rejected loudly rather than ignored --
a silently dropped ``learning_rate`` key is one of the more expensive typos in ML.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, TypeVar, get_type_hints

import yaml

T = TypeVar("T")


@dataclass
class ModelConfig:
    """Architecture hyper-parameters for the decoder-only transformer."""

    vocab_size: int = 512
    n_layer: int = 6
    n_head: int = 8
    n_kv_head: int | None = None
    """Number of key/value heads. ``None`` means multi-head attention
    (``n_kv_head == n_head``); a smaller value enables grouped-query attention."""
    d_model: int = 384
    d_ff: int | None = None
    """Feed-forward inner width. ``None`` derives ``8/3 * d_model`` rounded up to
    ``ffn_multiple_of``, which is the usual SwiGLU parameter-matching rule."""
    ffn_multiple_of: int = 64
    max_seq_len: int = 256
    dropout: float = 0.0
    rope_theta: float = 10_000.0
    norm_eps: float = 1e-5
    tie_embeddings: bool = True
    init_std: float = 0.02

    def __post_init__(self) -> None:
        if self.d_model % self.n_head != 0:
            raise ValueError(f"d_model={self.d_model} is not divisible by n_head={self.n_head}")
        if self.n_kv_head is None:
            self.n_kv_head = self.n_head
        if self.n_head % self.n_kv_head != 0:
            raise ValueError(
                f"n_head={self.n_head} is not divisible by n_kv_head={self.n_kv_head}; "
                "grouped-query attention needs an integer number of query heads per kv head"
            )
        if self.d_ff is None:
            hidden = int(8 * self.d_model / 3)
            mult = self.ffn_multiple_of
            self.d_ff = mult * ((hidden + mult - 1) // mult)

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_head

    @property
    def n_rep(self) -> int:
        """How many query heads share each key/value head."""
        assert self.n_kv_head is not None
        return self.n_head // self.n_kv_head


@dataclass
class DataConfig:
    """Where tokenised shards live and how batches are drawn from them."""

    data_dir: str = "data/processed"
    tokenizer_path: str = "data/processed/tokenizer.json"
    batch_size: int = 32
    seq_len: int = 256
    num_workers: int = 0


@dataclass
class OptimConfig:
    """AdamW plus the cosine-with-warmup schedule."""

    lr: float = 3e-4
    min_lr_ratio: float = 0.1
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8
    grad_clip: float = 1.0
    warmup_steps: int = 100


@dataclass
class TrainConfig:
    """Everything the training loop needs that is not model or data shaped."""

    max_steps: int = 2_000
    grad_accum_steps: int = 1
    eval_interval: int = 200
    eval_batches: int = 20
    log_interval: int = 20
    checkpoint_interval: int = 500
    seed: int = 1337
    device: str = "auto"
    dtype: str = "auto"
    """One of ``auto``/``float32``/``bfloat16``/``float16``. ``auto`` picks bf16 on
    hardware that supports it, otherwise float32."""
    compile_model: bool = False
    out_dir: str = "runs/default"
    deterministic: bool = False
    resume: str | None = None


@dataclass
class Config:
    """Root config: one YAML file describes an entire experiment."""

    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def _from_mapping(cls: type[T], data: dict[str, Any], path: str = "") -> T:
    """Build a dataclass from a mapping, rejecting unknown keys.

    ``from __future__ import annotations`` turns every field type into a string, so
    the nested-dataclass check has to go through ``get_type_hints`` rather than
    reading ``Field.type`` directly.
    """
    known = {f.name for f in fields(cls)}  # type: ignore[arg-type]
    unknown = set(data) - known
    if unknown:
        where = path or cls.__name__
        raise ValueError(f"unknown config key(s) under {where!r}: {sorted(unknown)}")

    hints = get_type_hints(cls)
    kwargs: dict[str, Any] = {}
    for key, value in data.items():
        hint = hints.get(key)
        if isinstance(hint, type) and is_dataclass(hint) and isinstance(value, dict):
            kwargs[key] = _from_mapping(hint, value, f"{path}.{key}" if path else key)
        else:
            kwargs[key] = value
    return cls(**kwargs)


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def load_config(path: str | Path | None = None, **overrides: Any) -> Config:
    """Load a YAML experiment file, then apply dotted CLI overrides.

    ``load_config("configs/tiny.yaml", **{"train.max_steps": 5})`` is the shape the
    CLI uses, so a smoke test never needs its own YAML file.
    """
    raw: dict[str, Any] = {}
    if path is not None:
        with open(path, encoding="utf-8") as fh:
            loaded = yaml.safe_load(fh) or {}
        if not isinstance(loaded, dict):
            raise ValueError(f"{path} must contain a YAML mapping at the top level")
        raw = loaded

    nested: dict[str, Any] = {}
    for dotted, value in overrides.items():
        if value is None:
            continue
        cursor = nested
        parts = dotted.split(".")
        for part in parts[:-1]:
            cursor = cursor.setdefault(part, {})
        cursor[parts[-1]] = value

    return _from_mapping(Config, _deep_merge(raw, nested))


def save_config(config: Config, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(config.to_dict(), fh, sort_keys=False)
