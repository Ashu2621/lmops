"""A byte-level BPE tokenizer trained from scratch.

Byte-level means the 256 single-byte tokens are always in the vocabulary, so the
tokenizer has no out-of-vocabulary case and ``decode(encode(s)) == s`` holds for
every string -- including emoji and text in scripts the training corpus never saw.

Training keeps incremental pair counts rather than rescanning the corpus once per
merge. On the bundled corpus that is the difference between a couple of seconds and
a couple of minutes, and the ``pair_index`` bookkeeping is small enough to verify by
eye against the naive version in ``tests/test_tokenizer.py``.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Iterable
from itertools import pairwise
from pathlib import Path

# GPT-2-style pre-tokenisation, written with constructs the stdlib `re` supports.
# `[^\W\d_]` is the portable spelling of "unicode letter". Splitting here stops BPE
# from ever learning a merge that straddles a space or a punctuation boundary.
PRETOKEN_PATTERN = re.compile(
    r"'(?:[sdmt]|ll|ve|re)| ?[^\W\d_]+| ?\d+| ?[^\s\w]+|\s+(?!\S)|\s+",
    re.UNICODE,
)

DEFAULT_SPECIAL_TOKENS = ("<|endoftext|>",)


class ByteBPETokenizer:
    """Byte-level byte-pair-encoding tokenizer with save/load round-tripping."""

    def __init__(
        self,
        merges: list[tuple[int, int]] | None = None,
        special_tokens: Iterable[str] = (),
    ) -> None:
        self.merges: list[tuple[int, int]] = list(merges or [])
        self.special_tokens: list[str] = list(special_tokens)
        self._rebuild()

    # -- construction ---------------------------------------------------------

    def _rebuild(self) -> None:
        """Derive the lookup tables that encode/decode need from ``merges``."""
        self.ranks: dict[tuple[int, int], int] = {pair: i for i, pair in enumerate(self.merges)}
        self.vocab: list[bytes] = [bytes([i]) for i in range(256)]
        for a, b in self.merges:
            self.vocab.append(self.vocab[a] + self.vocab[b])
        self.special_to_id: dict[str, int] = {
            tok: len(self.vocab) + i for i, tok in enumerate(self.special_tokens)
        }
        self.id_to_special: dict[int, str] = {v: k for k, v in self.special_to_id.items()}
        self._special_pattern = (
            re.compile("(" + "|".join(re.escape(t) for t in self.special_tokens) + ")")
            if self.special_tokens
            else None
        )
        self._encode_cache: dict[str, list[int]] = {}

    @property
    def vocab_size(self) -> int:
        return len(self.vocab) + len(self.special_tokens)

    def token_id(self, special: str) -> int:
        return self.special_to_id[special]

    # -- training -------------------------------------------------------------

    @classmethod
    def train(
        cls,
        text: str,
        vocab_size: int,
        special_tokens: Iterable[str] = DEFAULT_SPECIAL_TOKENS,
        progress: bool = False,
    ) -> ByteBPETokenizer:
        """Learn merges until the vocabulary reaches ``vocab_size``."""
        specials = list(special_tokens)
        n_merges = vocab_size - 256 - len(specials)
        if n_merges < 0:
            raise ValueError(
                f"vocab_size={vocab_size} is smaller than the 256 byte tokens plus "
                f"{len(specials)} special tokens"
            )

        word_freq_map = Counter(PRETOKEN_PATTERN.findall(text))
        words: list[list[int]] = []
        freqs: list[int] = []
        for word, freq in word_freq_map.items():
            words.append(list(word.encode("utf-8")))
            freqs.append(freq)

        pair_counts: Counter[tuple[int, int]] = Counter()
        pair_index: dict[tuple[int, int], set[int]] = {}
        for widx, symbols in enumerate(words):
            for pair in pairwise(symbols):
                pair_counts[pair] += freqs[widx]
                pair_index.setdefault(pair, set()).add(widx)

        merges: list[tuple[int, int]] = []
        next_id = 256
        for step in range(n_merges):
            if not pair_counts:
                break
            # Tie-break on the pair itself so training is reproducible across runs
            # and across Python versions with different dict ordering.
            best = max(pair_counts, key=lambda p: (pair_counts[p], p))
            if pair_counts[best] < 2:
                break

            merges.append(best)
            affected = list(pair_index.get(best, ()))
            for widx in affected:
                symbols = words[widx]
                freq = freqs[widx]
                for pair in pairwise(symbols):
                    pair_counts[pair] -= freq
                    if pair_counts[pair] <= 0:
                        del pair_counts[pair]
                    holders = pair_index.get(pair)
                    if holders is not None:
                        holders.discard(widx)

                words[widx] = _apply_merge(symbols, best, next_id)

                for pair in pairwise(words[widx]):
                    pair_counts[pair] += freq
                    pair_index.setdefault(pair, set()).add(widx)

            pair_counts.pop(best, None)
            pair_index.pop(best, None)
            next_id += 1

            if progress and (step + 1) % 100 == 0:
                print(f"  bpe merge {step + 1}/{n_merges}", flush=True)

        return cls(merges=merges, special_tokens=specials)

    # -- encode / decode ------------------------------------------------------

    def _encode_chunk(self, chunk: str) -> list[int]:
        cached = self._encode_cache.get(chunk)
        if cached is not None:
            return cached

        ids = list(chunk.encode("utf-8"))
        while len(ids) >= 2:
            # Always apply the lowest-rank (earliest learned) merge available.
            best_rank = None
            best_pos = -1
            for i in range(len(ids) - 1):
                rank = self.ranks.get((ids[i], ids[i + 1]))
                if rank is not None and (best_rank is None or rank < best_rank):
                    best_rank, best_pos = rank, i
            if best_rank is None:
                break
            pair = (ids[best_pos], ids[best_pos + 1])
            ids = _apply_merge(ids, pair, 256 + best_rank)

        self._encode_cache[chunk] = ids
        return ids

    def encode(self, text: str, allowed_special: bool = True) -> list[int]:
        """Encode ``text`` to token ids."""
        if not text:
            return []

        segments: list[str]
        if allowed_special and self._special_pattern is not None:
            segments = self._special_pattern.split(text)
        else:
            segments = [text]

        out: list[int] = []
        for segment in segments:
            if not segment:
                continue
            if allowed_special and segment in self.special_to_id:
                out.append(self.special_to_id[segment])
                continue
            for chunk in PRETOKEN_PATTERN.findall(segment):
                out.extend(self._encode_chunk(chunk))
        return out

    def token_bytes(self, token: int) -> bytes:
        """Raw bytes for a single token id.

        Streaming decoders need this: a token can end mid-way through a multi-byte
        character, so the caller has to buffer bytes rather than decode per token.
        """
        special = self.id_to_special.get(token)
        if special is not None:
            return special.encode("utf-8")
        if 0 <= token < len(self.vocab):
            return self.vocab[token]
        raise ValueError(f"token id {token} is outside vocab_size={self.vocab_size}")

    def decode(self, ids: Iterable[int]) -> str:
        """Decode token ids back to text.

        Byte sequences are joined before decoding so that a multi-byte character split
        across two tokens still round-trips; ``errors="replace"`` only ever fires on a
        truncated generation.
        """
        pieces: list[bytes] = []
        for token in ids:
            special = self.id_to_special.get(token)
            if special is not None:
                pieces.append(special.encode("utf-8"))
            elif 0 <= token < len(self.vocab):
                pieces.append(self.vocab[token])
            else:
                raise ValueError(f"token id {token} is outside vocab_size={self.vocab_size}")
        return b"".join(pieces).decode("utf-8", errors="replace")

    # -- persistence ----------------------------------------------------------

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": 1,
            "type": "byte_bpe",
            "vocab_size": self.vocab_size,
            "special_tokens": self.special_tokens,
            "merges": [[a, b] for a, b in self.merges],
        }
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> ByteBPETokenizer:
        with open(path, encoding="utf-8") as fh:
            payload = json.load(fh)
        if payload.get("type") != "byte_bpe":
            raise ValueError(f"{path} is not a byte-level BPE tokenizer file")
        merges = [(int(a), int(b)) for a, b in payload["merges"]]
        return cls(merges=merges, special_tokens=payload.get("special_tokens", []))


def _apply_merge(symbols: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
    """Replace every non-overlapping occurrence of ``pair`` in ``symbols``."""
    out: list[int] = []
    i = 0
    n = len(symbols)
    while i < n:
        if i < n - 1 and symbols[i] == pair[0] and symbols[i + 1] == pair[1]:
            out.append(new_id)
            i += 2
        else:
            out.append(symbols[i])
            i += 1
    return out
