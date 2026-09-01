"""Attention correctness: causality, the KV cache, and grouped-query expansion.

These are the tests that justify the cache existing. A cache that is merely *fast* and
subtly wrong is worse than no cache at all, and the failure mode is quiet -- generated
text stays grammatical, it just stops being the text the model would have produced.
"""

from __future__ import annotations

import pytest
import torch

from lmops.model import KVCache, LanguageModel, build_cache_mask, repeat_kv
from lmops.model.layers import RotaryEmbedding, apply_rotary


def test_repeat_kv_expands_each_head_contiguously() -> None:
    x = torch.arange(2 * 2 * 3 * 4, dtype=torch.float32).reshape(2, 2, 3, 4)
    out = repeat_kv(x, 3)
    assert out.shape == (2, 6, 3, 4)
    # Heads 0,1,2 must all be a copy of source head 0 -- not an interleave of 0 and 1.
    for rep in range(3):
        torch.testing.assert_close(out[:, rep], x[:, 0])
    for rep in range(3, 6):
        torch.testing.assert_close(out[:, rep], x[:, 1])


def test_repeat_kv_is_identity_for_mha() -> None:
    x = torch.randn(1, 4, 5, 8)
    assert repeat_kv(x, 1) is x


def test_cache_mask_allows_exactly_the_past() -> None:
    mask = build_cache_mask(q_len=3, offset=5, device=torch.device("cpu"))
    assert mask is not None
    assert mask.shape == (3, 8)
    # Query 0 sits at absolute position 5, so it sees keys 0..5 and nothing later.
    assert mask[0].tolist() == [True] * 6 + [False] * 2
    assert mask[1].tolist() == [True] * 7 + [False]
    assert mask[2].tolist() == [True] * 8


def test_cache_mask_is_none_for_single_token() -> None:
    assert build_cache_mask(q_len=1, offset=9, device=torch.device("cpu")) is None


def test_no_information_leaks_from_the_future(tiny_model: LanguageModel) -> None:
    """Editing token t must not change any logit at a position before t."""
    torch.manual_seed(0)
    ids = torch.randint(0, tiny_model.config.vocab_size, (1, 16))
    with torch.no_grad():
        baseline, _ = tiny_model(ids)

    edited = ids.clone()
    edited[0, 10] = (edited[0, 10] + 1) % tiny_model.config.vocab_size
    with torch.no_grad():
        changed, _ = tiny_model(edited)

    torch.testing.assert_close(baseline[:, :10], changed[:, :10], rtol=1e-5, atol=1e-6)
    assert not torch.allclose(baseline[:, 10:], changed[:, 10:])


def test_kv_cache_matches_full_forward(tiny_model: LanguageModel) -> None:
    """One token at a time through the cache == a single full-sequence forward pass.

    This is the load-bearing test of the whole serving path.
    """
    torch.manual_seed(1)
    seq_len = 12
    ids = torch.randint(0, tiny_model.config.vocab_size, (2, seq_len))

    with torch.no_grad():
        reference, _ = tiny_model(ids)

        cache = KVCache.allocate(tiny_model.config, batch_size=2, max_seq_len=seq_len)
        incremental = []
        for t in range(seq_len):
            logits, _ = tiny_model(ids[:, t : t + 1], cache=cache)
            incremental.append(logits)
        cached = torch.cat(incremental, dim=1)

    torch.testing.assert_close(reference, cached, rtol=1e-4, atol=1e-5)


def test_kv_cache_matches_full_forward_with_chunked_prefill(tiny_model: LanguageModel) -> None:
    """A multi-token prefill followed by single-token decode must also agree.

    This is the case ``is_causal=True`` gets wrong: the second chunk's queries start at
    absolute position 8, not 0, so the built-in causal mask would align to the wrong
    corner of the attention matrix.
    """
    torch.manual_seed(2)
    seq_len, prefill = 14, 8
    ids = torch.randint(0, tiny_model.config.vocab_size, (1, seq_len))

    with torch.no_grad():
        reference, _ = tiny_model(ids)

        cache = KVCache.allocate(tiny_model.config, batch_size=1, max_seq_len=seq_len)
        head, _ = tiny_model(ids[:, :prefill], cache=cache)
        tail, _ = tiny_model(ids[:, prefill:], cache=cache)
        cached = torch.cat([head, tail], dim=1)

    torch.testing.assert_close(reference, cached, rtol=1e-4, atol=1e-5)


def test_cache_overflow_is_an_explicit_error(tiny_model: LanguageModel) -> None:
    cache = KVCache.allocate(tiny_model.config, batch_size=1, max_seq_len=4)
    ids = torch.randint(0, tiny_model.config.vocab_size, (1, 5))
    with pytest.raises(ValueError, match="KV cache overflow"):
        tiny_model(ids, cache=cache)


def test_cache_reset_restores_initial_behaviour(tiny_model: LanguageModel) -> None:
    ids = torch.randint(0, tiny_model.config.vocab_size, (1, 4))
    cache = KVCache.allocate(tiny_model.config, batch_size=1, max_seq_len=8)
    with torch.no_grad():
        first, _ = tiny_model(ids, cache=cache)
        cache.reset()
        second, _ = tiny_model(ids, cache=cache)
    assert cache.pos == 4
    torch.testing.assert_close(first, second)


# -- rotary embeddings --------------------------------------------------------


def test_rope_offset_matches_a_full_length_slice() -> None:
    """``rope(n, offset=k)`` must equal positions ``k..k+n`` of the full table."""
    rope = RotaryEmbedding(head_dim=16, max_seq_len=32)
    full_cos, full_sin = rope(32, offset=0)
    part_cos, part_sin = rope(5, offset=7)
    torch.testing.assert_close(part_cos, full_cos[7:12])
    torch.testing.assert_close(part_sin, full_sin[7:12])


def test_rope_preserves_vector_norm() -> None:
    """RoPE is a rotation, so it must not change the length of a query vector."""
    rope = RotaryEmbedding(head_dim=16, max_seq_len=32)
    cos, sin = rope(6, offset=3)
    x = torch.randn(2, 4, 6, 16)
    rotated = apply_rotary(x, cos, sin)
    torch.testing.assert_close(x.norm(dim=-1), rotated.norm(dim=-1), rtol=1e-5, atol=1e-5)


def test_rope_rejects_positions_beyond_the_table() -> None:
    rope = RotaryEmbedding(head_dim=8, max_seq_len=16)
    with pytest.raises(ValueError, match="max_seq_len"):
        rope(4, offset=14)


def test_rope_encodes_relative_position() -> None:
    """The dot product of two rotated vectors depends only on their distance."""
    rope = RotaryEmbedding(head_dim=16, max_seq_len=64)
    q = torch.randn(1, 1, 1, 16)
    k = torch.randn(1, 1, 1, 16)

    def score(pos_q: int, pos_k: int) -> float:
        cq, sq = rope(1, offset=pos_q)
        ck, sk = rope(1, offset=pos_k)
        return float((apply_rotary(q, cq, sq) * apply_rotary(k, ck, sk)).sum())

    assert score(10, 4) == pytest.approx(score(30, 24), rel=1e-4, abs=1e-5)
