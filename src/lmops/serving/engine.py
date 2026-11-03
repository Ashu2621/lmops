"""The inference engine behind the HTTP API.

Three things here are worth more than the line count suggests:

*   **Incremental UTF-8 decoding.** A byte-level BPE token can end halfway through a
    multi-byte character. Decoding each token independently produces mojibake in the
    stream, so bytes are buffered until they form complete characters.
*   **Continuous batching.** By default every in-flight request is decoded in the
    same forward pass (``scheduler.py``), so concurrent requests share the cost of
    reading the weights instead of queueing behind each other.
*   **Admission control.** Overload is a fast ``503``, not an unbounded queue where
    every request times out and the client retries into the same wall: a bounded
    waiting queue with a timeout when batching, a bounded semaphore when not.
*   **Thread offload.** Torch releases the GIL inside its kernels, so running
    generation in a worker thread keeps the event loop free to serve ``/healthz``
    while the model is busy.
"""

from __future__ import annotations

import contextlib
import os
import threading
import time
from collections.abc import Generator, Iterator
from dataclasses import dataclass
from pathlib import Path

import torch

from ..data import ByteBPETokenizer
from ..model import LanguageModel
from ..training.checkpoint import build_model_from_checkpoint
from ..training.trainer import resolve_device
from .scheduler import BatchScheduler, EngineBusy
from .schemas import GenerateRequest, GenerateResponse, Usage


class IncrementalDecoder:
    """Turns a stream of token ids into a stream of valid UTF-8 text."""

    def __init__(self, tokenizer: ByteBPETokenizer) -> None:
        self.tokenizer = tokenizer
        self._buffer = b""

    def push(self, token: int) -> str:
        """Append one token and return whatever text is now complete."""
        self._buffer += self.tokenizer.token_bytes(token)
        try:
            text = self._buffer.decode("utf-8")
        except UnicodeDecodeError as err:
            if err.start == 0:
                return ""
            text = self._buffer[: err.start].decode("utf-8")
            self._buffer = self._buffer[err.start :]
            return text
        self._buffer = b""
        return text

    def flush(self) -> str:
        """Emit any trailing bytes, replacing an incomplete final character."""
        if not self._buffer:
            return ""
        text = self._buffer.decode("utf-8", errors="replace")
        self._buffer = b""
        return text


@dataclass
class EngineInfo:
    """Static facts about the loaded model, surfaced on ``/readyz``."""

    checkpoint: str
    device: str
    parameters: int
    max_seq_len: int
    vocab_size: int
    step: int


class GenerationEngine:
    """Loads a checkpoint once and serves generation requests against it."""

    def __init__(
        self,
        checkpoint: str | Path,
        tokenizer_path: str | Path,
        device: str = "auto",
        max_concurrency: int = 4,
        queue_timeout_s: float = 10.0,
        torch_threads: int | None = None,
        batching: bool = True,
        max_queue: int = 32,
    ) -> None:
        if torch_threads:
            torch.set_num_threads(torch_threads)

        self.device = resolve_device(device)
        self.model, meta = build_model_from_checkpoint(checkpoint, device=self.device)
        self.model.eval()
        self.tokenizer = ByteBPETokenizer.load(tokenizer_path)
        self.queue_timeout_s = queue_timeout_s
        self._slots = threading.BoundedSemaphore(max_concurrency)
        self.max_concurrency = max_concurrency

        if self.tokenizer.vocab_size != self.model.config.vocab_size:
            raise ValueError(
                f"tokenizer vocab_size={self.tokenizer.vocab_size} does not match the "
                f"checkpoint's vocab_size={self.model.config.vocab_size}; these artifacts "
                "come from different runs"
            )

        # With batching, max_concurrency is the batch size: that many requests decode
        # together and the rest wait (up to max_queue of them) for a free slot.
        self.scheduler = (
            BatchScheduler(self.model, max_concurrency, max_queue, queue_timeout_s)
            if batching
            else None
        )

        self.info = EngineInfo(
            checkpoint=str(checkpoint),
            device=str(self.device),
            parameters=self.model.num_parameters(),
            max_seq_len=self.model.config.max_seq_len,
            vocab_size=self.model.config.vocab_size,
            step=int(meta.get("step", 0)),
        )

    # -- construction ---------------------------------------------------------

    @classmethod
    def from_env(cls) -> GenerationEngine:
        """Build an engine from ``LMOPS_*`` environment variables (the container path)."""
        return cls(
            checkpoint=os.environ.get("LMOPS_CHECKPOINT", "runs/default/best.pt"),
            tokenizer_path=os.environ.get("LMOPS_TOKENIZER", "data/processed/tokenizer.json"),
            device=os.environ.get("LMOPS_DEVICE", "auto"),
            max_concurrency=int(os.environ.get("LMOPS_MAX_CONCURRENCY", "4")),
            queue_timeout_s=float(os.environ.get("LMOPS_QUEUE_TIMEOUT_S", "10")),
            torch_threads=int(os.environ["LMOPS_TORCH_THREADS"])
            if os.environ.get("LMOPS_TORCH_THREADS")
            else None,
            batching=os.environ.get("LMOPS_BATCHING", "1") not in ("0", "false", "off"),
            max_queue=int(os.environ.get("LMOPS_MAX_QUEUE", "32")),
        )

    def close(self) -> None:
        if self.scheduler is not None:
            self.scheduler.close()

    # -- internals ------------------------------------------------------------

    def _encode_prompt(self, prompt: str, max_new_tokens: int) -> list[int]:
        ids = self.tokenizer.encode(prompt)
        if not ids:
            raise ValueError("prompt encoded to zero tokens")
        budget = self.model.config.max_seq_len - max_new_tokens
        if budget <= 0:
            raise ValueError(
                f"max_new_tokens={max_new_tokens} leaves no room in a "
                f"{self.model.config.max_seq_len}-token context"
            )
        # Left-truncate: the tokens nearest the continuation matter most.
        return ids[-budget:]

    def _generator(self, seed: int | None) -> torch.Generator | None:
        if seed is None:
            return None
        return torch.Generator(device=str(self.device)).manual_seed(seed)

    def _acquire(self) -> None:
        if not self._slots.acquire(timeout=self.queue_timeout_s):
            raise EngineBusy(
                f"all {self.max_concurrency} generation slots busy for {self.queue_timeout_s:.0f}s"
            )

    @contextlib.contextmanager
    def _admitted(self) -> Iterator[None]:
        """The semaphore is the admission control only without batching; the
        scheduler does its own (queue bound and queue timeout)."""
        if self.scheduler is not None:
            yield
            return
        self._acquire()
        try:
            yield
        finally:
            self._slots.release()

    def _token_ids(self, request: GenerateRequest, ids: list[int]) -> Iterator[int]:
        generator = self._generator(request.seed)
        if self.scheduler is not None:
            job = self.scheduler.submit(
                {
                    "ids": ids,
                    "max_new_tokens": request.max_new_tokens,
                    "temperature": request.temperature,
                    "top_k": request.top_k or None,
                    "top_p": request.top_p,
                    "repetition_penalty": request.repetition_penalty,
                    "generator": generator,
                }
            )
            yield from job.tokens()
            return
        idx = torch.tensor([ids], dtype=torch.long, device=self.device)
        for token in self.model.stream(
            idx,
            request.max_new_tokens,
            temperature=request.temperature,
            top_k=request.top_k or None,
            top_p=request.top_p,
            repetition_penalty=request.repetition_penalty,
            generator=generator,
        ):
            yield int(token[0, 0].item())

    # -- public API -----------------------------------------------------------

    def generate(self, request: GenerateRequest) -> GenerateResponse:
        """Run a blocking, non-streaming generation."""
        with self._admitted():
            start = time.perf_counter()
            ids = self._encode_prompt(request.prompt, request.max_new_tokens)

            decoder = IncrementalDecoder(self.tokenizer)
            pieces: list[str] = []
            ttft = 0.0
            produced = 0

            for token in self._token_ids(request, ids):
                if produced == 0:
                    ttft = (time.perf_counter() - start) * 1000.0
                produced += 1
                pieces.append(decoder.push(token))
            pieces.append(decoder.flush())

            completion = "".join(pieces)
            latency = (time.perf_counter() - start) * 1000.0
            return GenerateResponse(
                text=request.prompt + completion,
                completion=completion,
                usage=Usage(
                    prompt_tokens=len(ids),
                    completion_tokens=produced,
                    total_tokens=len(ids) + produced,
                ),
                latency_ms=round(latency, 2),
                ttft_ms=round(ttft, 2),
                model=Path(self.info.checkpoint).name,
                seed=request.seed,
            )

    def stream(self, request: GenerateRequest) -> Generator[dict, None, None]:
        """Yield ``{"delta": ...}`` events, then a final ``{"done": true, ...}`` event."""
        with self._admitted():
            start = time.perf_counter()
            ids = self._encode_prompt(request.prompt, request.max_new_tokens)

            decoder = IncrementalDecoder(self.tokenizer)
            produced = 0
            ttft = 0.0

            for token in self._token_ids(request, ids):
                if produced == 0:
                    ttft = (time.perf_counter() - start) * 1000.0
                produced += 1
                delta = decoder.push(token)
                if delta:
                    yield {"delta": delta}

            tail = decoder.flush()
            if tail:
                yield {"delta": tail}

            yield {
                "done": True,
                "usage": {
                    "prompt_tokens": len(ids),
                    "completion_tokens": produced,
                    "total_tokens": len(ids) + produced,
                },
                "latency_ms": round((time.perf_counter() - start) * 1000.0, 2),
                "ttft_ms": round(ttft, 2),
            }


__all__ = ["EngineBusy", "EngineInfo", "GenerationEngine", "IncrementalDecoder", "LanguageModel"]
