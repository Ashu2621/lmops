"""Byte-level BPE: round-tripping, determinism, and the incremental training path.

The centrepiece is :func:`test_incremental_training_matches_naive_reference`. The
production trainer keeps incremental pair counts for speed; this test reimplements the
obvious quadratic algorithm in a dozen lines and asserts both learn the same merges, so
the optimisation cannot drift away from the definition of BPE without a failure.
"""

from __future__ import annotations

from collections import Counter
from itertools import pairwise

import pytest

from lmops.data.tokenizer import (
    PRETOKEN_PATTERN,
    ByteBPETokenizer,
    _apply_merge,
)


def _naive_train(text: str, n_merges: int) -> list[tuple[int, int]]:
    """Textbook BPE: recount every pair from scratch before each merge."""
    freqs = Counter(PRETOKEN_PATTERN.findall(text))
    word_freq: list[tuple[list[int], int]] = [
        (list(word.encode("utf-8")), freq) for word, freq in freqs.items()
    ]

    merges: list[tuple[int, int]] = []
    next_id = 256
    for _ in range(n_merges):
        counts: Counter[tuple[int, int]] = Counter()
        for symbols, freq in word_freq:
            for pair in pairwise(symbols):
                counts[pair] += freq
        if not counts:
            break
        best = max(counts, key=lambda p: (counts[p], p))
        if counts[best] < 2:
            break
        merges.append(best)
        word_freq = [(_apply_merge(s, best, next_id), f) for s, f in word_freq]
        next_id += 1
    return merges


def test_incremental_training_matches_naive_reference(corpus_text: str) -> None:
    text = corpus_text[:20_000]
    fast = ByteBPETokenizer.train(text, vocab_size=256 + 60, special_tokens=())
    slow = _naive_train(text, n_merges=60)
    assert fast.merges == slow


def test_roundtrip_is_lossless(corpus_text: str, tokenizer: ByteBPETokenizer) -> None:
    sample = corpus_text[:5_000]
    assert tokenizer.decode(tokenizer.encode(sample)) == sample


@pytest.mark.parametrize(
    "text",
    [
        "",
        " ",
        "\n\n\n",
        "hello world",
        "Ünïcödé ünd ümläüts",
        "emoji: 🚀🔥 mixed with ascii",
        "日本語のテキスト",
        "tabs\tand\r\nwindows newlines",
        "1234567890 !@#$%^&*()",
    ],
)
def test_roundtrip_survives_text_the_corpus_never_contained(
    tokenizer: ByteBPETokenizer, text: str
) -> None:
    """Byte-level means there is no OOV case, even for scripts BPE never saw."""
    assert tokenizer.decode(tokenizer.encode(text)) == text


def test_every_token_id_is_within_the_vocabulary(
    tokenizer: ByteBPETokenizer, corpus_text: str
) -> None:
    ids = tokenizer.encode(corpus_text[:20_000])
    assert ids
    assert all(0 <= i < tokenizer.vocab_size for i in ids)


def test_training_is_deterministic(corpus_text: str) -> None:
    text = corpus_text[:10_000]
    a = ByteBPETokenizer.train(text, vocab_size=300)
    b = ByteBPETokenizer.train(text, vocab_size=300)
    assert a.merges == b.merges


def test_merges_actually_compress(tokenizer: ByteBPETokenizer, corpus_text: str) -> None:
    sample = corpus_text[:20_000]
    n_tokens = len(tokenizer.encode(sample))
    n_bytes = len(sample.encode("utf-8"))
    assert n_tokens < n_bytes, "a trained BPE must beat raw bytes"
    assert n_bytes / n_tokens > 1.5


def test_save_load_roundtrip(tokenizer: ByteBPETokenizer, tmp_path, corpus_text: str) -> None:
    path = tmp_path / "tok.json"
    tokenizer.save(path)
    reloaded = ByteBPETokenizer.load(path)
    assert reloaded.merges == tokenizer.merges
    assert reloaded.special_tokens == tokenizer.special_tokens
    assert reloaded.vocab_size == tokenizer.vocab_size
    sample = corpus_text[:2_000]
    assert reloaded.encode(sample) == tokenizer.encode(sample)


def test_special_tokens_encode_atomically() -> None:
    tok = ByteBPETokenizer.train("hello world hello there", vocab_size=280)
    eot = tok.token_id("<|endoftext|>")
    ids = tok.encode("hello<|endoftext|>world")
    assert eot in ids
    assert ids.count(eot) == 1
    assert tok.decode(ids) == "hello<|endoftext|>world"


def test_special_tokens_can_be_disabled() -> None:
    tok = ByteBPETokenizer.train("hello world", vocab_size=280)
    ids = tok.encode("<|endoftext|>", allowed_special=False)
    assert tok.token_id("<|endoftext|>") not in ids
    assert tok.decode(ids) == "<|endoftext|>"


def test_token_bytes_concatenate_to_the_decoded_string(
    tokenizer: ByteBPETokenizer, corpus_text: str
) -> None:
    """The invariant the streaming decoder relies on."""
    ids = tokenizer.encode(corpus_text[:1_000])
    joined = b"".join(tokenizer.token_bytes(i) for i in ids)
    assert joined.decode("utf-8") == corpus_text[:1_000]


def test_vocab_size_below_the_byte_floor_is_rejected() -> None:
    with pytest.raises(ValueError, match="smaller than the 256 byte tokens"):
        ByteBPETokenizer.train("abc", vocab_size=100)


def test_decode_rejects_out_of_range_ids(tokenizer: ByteBPETokenizer) -> None:
    with pytest.raises(ValueError, match="outside vocab_size"):
        tokenizer.decode([tokenizer.vocab_size + 5])


def test_apply_merge_does_not_overlap() -> None:
    """``aaa`` with the merge ``(a, a)`` yields one merged token plus a leftover."""
    assert _apply_merge([97, 97, 97], (97, 97), 256) == [256, 97]
    assert _apply_merge([97, 97, 97, 97], (97, 97), 256) == [256, 256]
