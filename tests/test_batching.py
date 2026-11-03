"""Continuous batching: the per-slot cache, the scheduler, and the engine on top.

The property that matters is the same one the KV-cache tests guard, one level up:
batching is a scheduling decision and must never change what a request returns.
"""

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
import torch

from lmops.model import KVCache, LanguageModel, SlotKVCache, SlotStep
from lmops.serving.engine import EngineBusy, GenerationEngine
from lmops.serving.observability import BATCH_SIZE
from lmops.serving.schemas import GenerateRequest

# -- the slot cache -----------------------------------------------------------


def _prefill_into(
    model: LanguageModel, cache: SlotKVCache, slot: int, ids: torch.Tensor
) -> torch.Tensor:
    single = KVCache.allocate(model.config, 1, max_seq_len=ids.shape[1])
    logits, _ = model(ids, cache=single)
    cache.load(slot, single)
    return logits


def test_slot_decode_matches_full_forward_for_rows_at_different_lengths(
    tiny_model: LanguageModel,
) -> None:
    """Three sequences with different prompt lengths, the third joining two steps
    late, decoded together. Every row's logits must equal a full forward pass over
    that row alone - the batched step may not leak one row's keys into another or
    rotate a row at the wrong position."""
    torch.manual_seed(1)
    vocab = tiny_model.config.vocab_size
    prompts = [torch.randint(0, vocab, (1, n)) for n in (5, 11, 3)]
    follow = [torch.randint(0, vocab, (1, 6)) for _ in prompts]
    joins = [0, 0, 2]

    cache = SlotKVCache.allocate(tiny_model.config, n_slots=4)
    slots = [3, 0, 1]  # deliberately not 0..n-1
    fed = [0, 0, 0]
    got: list[list[torch.Tensor]] = [[], [], []]

    with torch.inference_mode():
        for step in range(8):
            for i in range(3):
                if joins[i] == step:
                    _prefill_into(tiny_model, cache, slots[i], prompts[i])
            rows = [i for i in range(3) if joins[i] <= step and fed[i] < follow[i].shape[1]]
            if not rows:
                break
            tokens = torch.cat([follow[i][:, fed[i] : fed[i] + 1] for i in rows], dim=0)
            logits = tiny_model.decode_step(
                tokens, SlotStep(cache, torch.tensor([slots[i] for i in rows]))
            )
            for row, i in enumerate(rows):
                got[i].append(logits[row : row + 1])
                fed[i] += 1

        for i in range(3):
            full, _ = tiny_model(torch.cat([prompts[i], follow[i]], dim=1))
            n = prompts[i].shape[1]
            expected = full[:, n : n + follow[i].shape[1]]
            torch.testing.assert_close(torch.cat(got[i], dim=1), expected, rtol=1e-4, atol=1e-5)


def test_stale_keys_from_a_previous_occupant_are_invisible(tiny_model: LanguageModel) -> None:
    """A freed slot is reused without zeroing. A long sequence leaves keys behind
    that a short newcomer must not attend to."""
    vocab = tiny_model.config.vocab_size
    long_ids = torch.randint(0, vocab, (1, 40))
    short_ids = torch.randint(0, vocab, (1, 4))
    nxt = torch.randint(0, vocab, (1, 1))

    with torch.inference_mode():
        dirty = SlotKVCache.allocate(tiny_model.config, n_slots=1)
        _prefill_into(tiny_model, dirty, 0, long_ids)
        _prefill_into(tiny_model, dirty, 0, short_ids)
        reused = tiny_model.decode_step(nxt, SlotStep(dirty, torch.tensor([0])))

        clean = SlotKVCache.allocate(tiny_model.config, n_slots=1)
        _prefill_into(tiny_model, clean, 0, short_ids)
        fresh = tiny_model.decode_step(nxt, SlotStep(clean, torch.tensor([0])))
    torch.testing.assert_close(reused, fresh)


def test_slot_overflow_is_an_error_not_a_wraparound(tiny_model: LanguageModel) -> None:
    cache = SlotKVCache.allocate(tiny_model.config, n_slots=1, max_seq_len=4)
    cache.lengths[0] = 4
    with pytest.raises(ValueError, match="overflow"):
        SlotStep(cache, torch.tensor([0]))


# -- the engine ---------------------------------------------------------------


def _engine(ckpt: tuple[Path, Path], **kw) -> GenerationEngine:
    checkpoint, tokenizer = ckpt
    return GenerationEngine(checkpoint, tokenizer, device="cpu", **kw)


REQUESTS = [
    GenerateRequest(prompt="The river", max_new_tokens=16, temperature=0.0),
    GenerateRequest(prompt="Old stone bridge", max_new_tokens=9, temperature=1.0, seed=11),
    GenerateRequest(prompt="Birds", max_new_tokens=20, temperature=0.8, top_k=5, seed=3),
    GenerateRequest(prompt="The quick brown fox jumps", max_new_tokens=5, temperature=0.0),
    GenerateRequest(
        prompt="Morning light",
        max_new_tokens=12,
        temperature=1.2,
        top_p=0.9,
        repetition_penalty=1.3,
        seed=7,
    ),
    GenerateRequest(prompt="Forest", max_new_tokens=14, temperature=1.0, seed=99),
]


def test_batched_output_equals_one_at_a_time_output(served_checkpoint: tuple[Path, Path]) -> None:
    """Six requests with different prompts, lengths and sampling settings, fired at
    once so they share decode steps, return exactly what each returns when served
    alone without batching. Seeded sampling included."""
    alone = _engine(served_checkpoint, batching=False)
    expected = [alone.generate(r).completion for r in REQUESTS]

    batched = _engine(served_checkpoint, batching=True, max_concurrency=4)
    before = BATCH_SIZE._sum.get()
    try:
        with ThreadPoolExecutor(len(REQUESTS)) as pool:
            got = list(pool.map(lambda r: batched.generate(r).completion, REQUESTS))
    finally:
        batched.close()
    assert got == expected
    assert BATCH_SIZE._sum.get() - before > sum(r.max_new_tokens for r in REQUESTS) / 4, (
        "requests never shared a decode step"
    )


def test_streaming_through_the_scheduler_matches_the_plain_response(
    served_checkpoint: tuple[Path, Path],
) -> None:
    engine = _engine(served_checkpoint)
    try:
        request = GenerateRequest(prompt="The", max_new_tokens=10, seed=5)
        streamed = "".join(e.get("delta", "") for e in engine.stream(request))
        assert streamed == engine.generate(request).completion
    finally:
        engine.close()


def _slow_decode(engine: GenerationEngine, seconds: float = 0.005) -> None:
    """Make each decode step take a known minimum time. The admission tests need
    a request to still be running when the next one arrives; on a fast machine an
    80-token generation of the tiny test model can finish before that."""
    real = engine.model.decode_step

    def slow(*args, **kwargs):
        time.sleep(seconds)
        return real(*args, **kwargs)

    engine.model.decode_step = slow  # type: ignore[method-assign]


def test_a_full_queue_is_rejected_immediately(served_checkpoint: tuple[Path, Path]) -> None:
    """One slot busy, one request queued: the next is a fast EngineBusy (503), not a
    third place in an ever-growing line."""
    engine = _engine(served_checkpoint, max_concurrency=1, max_queue=1, queue_timeout_s=30)
    _slow_decode(engine)
    long = GenerateRequest(prompt="x", max_new_tokens=80, temperature=0.0)
    try:
        with ThreadPoolExecutor(2) as pool:
            running = pool.submit(engine.generate, long)
            _wait_for(lambda: engine.scheduler.active == 1)
            queued = pool.submit(engine.generate, long)
            _wait_for(lambda: len(engine.scheduler._waiting) == 1)
            started = time.perf_counter()
            with pytest.raises(EngineBusy):
                engine.generate(GenerateRequest(prompt="y", max_new_tokens=2))
            assert time.perf_counter() - started < 1.0
            assert running.result().usage.completion_tokens == 80
            assert queued.result().usage.completion_tokens == 80
    finally:
        engine.close()


def test_a_request_that_waits_past_its_timeout_is_rejected(
    served_checkpoint: tuple[Path, Path],
) -> None:
    """The timeout holds while the batch stays full. It used to be checked only when
    a slot freed up, so a waiting client could wait for the whole of someone else's
    generation; a fast CI runner exposed it as a flaky pass."""
    engine = _engine(served_checkpoint, max_concurrency=1, max_queue=4, queue_timeout_s=0.05)
    _slow_decode(engine)  # ~0.4 s for 80 tokens, far past the 0.05 s timeout
    try:
        with ThreadPoolExecutor(1) as pool:
            running = pool.submit(
                engine.generate, GenerateRequest(prompt="x", max_new_tokens=80, temperature=0.0)
            )
            _wait_for(lambda: engine.scheduler.active == 1)
            started = time.perf_counter()
            with pytest.raises(EngineBusy, match="waited"):
                engine.generate(GenerateRequest(prompt="y", max_new_tokens=2))
            assert time.perf_counter() - started < 0.3, "rejected only after the slot freed"
            assert not running.done()
            running.result()
    finally:
        engine.close()


def test_a_client_that_goes_away_frees_its_slot(served_checkpoint: tuple[Path, Path]) -> None:
    """Closing a stream mid-generation (what the HTTP layer does on disconnect) must
    retire the sequence at the next step, not keep generating to max_new_tokens."""
    engine = _engine(served_checkpoint, max_concurrency=2)
    _slow_decode(engine)
    try:
        events = engine.stream(GenerateRequest(prompt="The", max_new_tokens=80, temperature=0.0))
        next(events)
        assert engine.scheduler.active == 1
        events.close()
        _wait_for(lambda: engine.scheduler.active == 0)
    finally:
        engine.close()


def _wait_for(condition, timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while not condition():
        if time.monotonic() > deadline:
            raise AssertionError("condition not reached")
        time.sleep(0.002)


def test_the_scheduler_thread_stops_on_close(served_checkpoint: tuple[Path, Path]) -> None:
    engine = _engine(served_checkpoint)
    thread: threading.Thread = engine.scheduler._thread
    engine.close()
    assert not thread.is_alive()
