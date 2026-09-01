"""Causal self-attention with grouped-query heads and an incremental KV cache.

The cache is the part worth reading closely. Two invariants make cached decoding
agree with a full forward pass, and both are easy to get wrong:

1.  Rotary embeddings for step *t* must be taken at absolute position *t*, not at
    position 0 of the one-token slice being fed in. Hence the ``offset`` threaded
    through from :class:`~lmops.model.layers.RotaryEmbedding`.
2.  ``is_causal=True`` in ``scaled_dot_product_attention`` aligns the mask to the
    *top-left* of the attention matrix. With a cache, queries sit at the *bottom*
    of a longer key axis, so the flag silently masks the wrong entries. Cached
    passes therefore build an explicit offset mask instead.

``tests/test_attention.py`` asserts the two paths produce identical logits, which
is what keeps these invariants honest.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ..config import ModelConfig
from .layers import apply_rotary


@dataclass
class KVCache:
    """Pre-allocated key/value cache shared by every layer of one model.

    All layers advance in lock-step, so a single ``pos`` cursor is enough; the model
    bumps it once after the full layer stack has consumed the current chunk.
    """

    keys: list[Tensor]
    values: list[Tensor]
    pos: int = 0

    @classmethod
    def allocate(
        cls,
        config: ModelConfig,
        batch_size: int,
        max_seq_len: int | None = None,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> KVCache:
        assert config.n_kv_head is not None
        length = max_seq_len or config.max_seq_len
        shape = (batch_size, config.n_kv_head, length, config.head_dim)
        keys = [torch.zeros(shape, device=device, dtype=dtype) for _ in range(config.n_layer)]
        values = [torch.zeros(shape, device=device, dtype=dtype) for _ in range(config.n_layer)]
        return cls(keys=keys, values=values)

    @property
    def capacity(self) -> int:
        return self.keys[0].shape[2]

    def update(self, layer_idx: int, k: Tensor, v: Tensor) -> tuple[Tensor, Tensor]:
        """Write this step's keys/values and return the full history so far."""
        seq_len = k.shape[2]
        end = self.pos + seq_len
        if end > self.capacity:
            raise ValueError(
                f"KV cache overflow: writing to position {end} of a {self.capacity}-slot "
                "cache. Allocate a longer cache or stop generating."
            )
        self.keys[layer_idx][:, :, self.pos : end] = k
        self.values[layer_idx][:, :, self.pos : end] = v
        return self.keys[layer_idx][:, :, :end], self.values[layer_idx][:, :, :end]

    def advance(self, steps: int) -> None:
        self.pos += steps

    def reset(self) -> None:
        self.pos = 0


def repeat_kv(x: Tensor, n_rep: int) -> Tensor:
    """Expand ``(B, n_kv_head, S, D)`` to ``(B, n_kv_head * n_rep, S, D)`` for GQA."""
    if n_rep == 1:
        return x
    batch, n_kv_head, seq_len, head_dim = x.shape
    expanded = x[:, :, None, :, :].expand(batch, n_kv_head, n_rep, seq_len, head_dim)
    return expanded.reshape(batch, n_kv_head * n_rep, seq_len, head_dim)


def build_cache_mask(q_len: int, offset: int, device: torch.device) -> Tensor | None:
    """Boolean keep-mask of shape ``(q_len, offset + q_len)`` for cached attention.

    Returns ``None`` for the single-token case, where every cached key is visible and
    the mask would be all-True anyway.
    """
    if q_len == 1:
        return None
    kv_len = offset + q_len
    q_pos = torch.arange(offset, kv_len, device=device).unsqueeze(1)
    k_pos = torch.arange(kv_len, device=device).unsqueeze(0)
    return k_pos <= q_pos


class CausalSelfAttention(nn.Module):
    """Multi-head (or grouped-query) causal attention."""

    def __init__(self, config: ModelConfig, layer_idx: int) -> None:
        super().__init__()
        assert config.n_kv_head is not None
        self.layer_idx = layer_idx
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_rep = config.n_rep
        self.head_dim = config.head_dim
        self.dropout_p = config.dropout

        self.q_proj = nn.Linear(config.d_model, config.n_head * config.head_dim, bias=False)
        self.k_proj = nn.Linear(config.d_model, config.n_kv_head * config.head_dim, bias=False)
        self.v_proj = nn.Linear(config.d_model, config.n_kv_head * config.head_dim, bias=False)
        self.o_proj = nn.Linear(config.n_head * config.head_dim, config.d_model, bias=False)
        self.resid_dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        x: Tensor,
        cos: Tensor,
        sin: Tensor,
        cache: KVCache | None = None,
        attn_mask: Tensor | None = None,
    ) -> Tensor:
        batch, seq_len, _ = x.shape

        q = self.q_proj(x).view(batch, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch, seq_len, self.n_kv_head, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch, seq_len, self.n_kv_head, self.head_dim).transpose(1, 2)

        q = apply_rotary(q, cos, sin)
        k = apply_rotary(k, cos, sin)

        if cache is not None:
            k, v = cache.update(self.layer_idx, k, v)

        k = repeat_kv(k, self.n_rep)
        v = repeat_kv(v, self.n_rep)

        # See the module docstring: `is_causal` is only correct when the query axis
        # and the key axis start at the same absolute position.
        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=self.dropout_p if self.training else 0.0,
            is_causal=cache is None,
        )
        out = out.transpose(1, 2).contiguous().view(batch, seq_len, self.n_head * self.head_dim)
        return self.resid_dropout(self.o_proj(out))
