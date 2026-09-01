"""Building blocks: RMSNorm, rotary position embeddings and a SwiGLU MLP.

These are written out rather than imported so the numerics are inspectable -- the
rotary cache in particular is the piece most often subtly wrong when a KV cache is
bolted on later, because the second forward pass has to resume at the right offset.
"""

from __future__ import annotations

from typing import cast

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class RMSNorm(nn.Module):
    """Root-mean-square layer norm (no mean subtraction, no bias).

    Normalisation is computed in float32 even under autocast: the reciprocal square
    root of a half-precision mean-square is a well known source of training blow-ups.
    """

    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        dtype = x.dtype
        x32 = x.float()
        normed = x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + self.eps)
        return (normed.to(dtype)) * self.weight


class RotaryEmbedding(nn.Module):
    """Precomputed rotary position embeddings (RoPE).

    ``cos``/``sin`` tables are registered as non-persistent buffers: they are pure
    functions of ``(max_seq_len, head_dim, theta)``, so writing them into every
    checkpoint would just bloat the artifact and pin the context length forever.
    """

    def __init__(self, head_dim: int, max_seq_len: int, theta: float = 10_000.0) -> None:
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError(f"rotary head_dim must be even, got {head_dim}")
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.theta = theta
        cos, sin = self._build_tables(max_seq_len)
        self.register_buffer("cos_cached", cos, persistent=False)
        self.register_buffer("sin_cached", sin, persistent=False)

    def _build_tables(self, seq_len: int) -> tuple[Tensor, Tensor]:
        exponent = torch.arange(0, self.head_dim, 2, dtype=torch.float32) / self.head_dim
        inv_freq = 1.0 / (self.theta**exponent)
        positions = torch.arange(seq_len, dtype=torch.float32)
        angles = torch.outer(positions, inv_freq)  # (seq_len, head_dim / 2)
        return angles.cos(), angles.sin()

    def forward(
        self, seq_len: int, offset: int = 0, device: torch.device | None = None
    ) -> tuple[Tensor, Tensor]:
        """Return ``(cos, sin)`` slices covering ``[offset, offset + seq_len)``."""
        end = offset + seq_len
        # Under tracing `seq_len` is a symbolic value, so this comparison would be
        # frozen into the graph as a constant and emit a TracerWarning. The bound is a
        # developer guard-rail, not part of the exported computation.
        if not torch.jit.is_tracing() and end > self.max_seq_len:
            raise ValueError(
                f"requested rotary positions up to {end} but the table covers "
                f"{self.max_seq_len}; raise model.max_seq_len"
            )
        cos = cast(Tensor, self.cos_cached)[offset:end]
        sin = cast(Tensor, self.sin_cached)[offset:end]
        if device is not None:
            cos, sin = cos.to(device), sin.to(device)
        return cos, sin


def _rotate_half(x: Tensor) -> Tensor:
    """Split the last dim in half and rotate: ``(x1, x2) -> (-x2, x1)``."""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    """Apply RoPE to ``x`` of shape ``(batch, heads, seq, head_dim)``.

    ``cos``/``sin`` arrive as ``(seq, head_dim / 2)`` and are duplicated to the full
    head width so the halved-rotation form above lines up with the interleaved pairs.
    """
    cos = torch.cat((cos, cos), dim=-1)[None, None, :, :].to(x.dtype)
    sin = torch.cat((sin, sin), dim=-1)[None, None, :, :].to(x.dtype)
    return x * cos + _rotate_half(x) * sin


class SwiGLU(nn.Module):
    """Gated feed-forward block: ``down(silu(gate(x)) * up(x))``."""

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(d_model, d_ff, bias=False)
        self.up_proj = nn.Linear(d_model, d_ff, bias=False)
        self.down_proj = nn.Linear(d_ff, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        return self.dropout(self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x)))
