"""Continuous batching: one decode step serves every request in flight.

Without it each request owns the model for its whole generation, so N concurrent
requests take N times as long as one: per-token cost is dominated by reading the
weights, and that read is repeated for every request separately. Here a single
background thread runs the loop

    admit:   prefill waiting requests into free cache slots (each emits its 1st token)
    decode:  one forward pass over all active slots, one new token each
    retire:  free the slots of requests that finished or whose client went away

so a request joins the running batch at the next step instead of waiting for the
current ones to finish, and the weights are read once per step for all of them.
Each request keeps its own sampling settings and its own seeded generator, so a
seeded request returns the same tokens whether it ran alone or in a batch of eight.

Prefill is done one request at a time through the ordinary single-sequence path and
then copied into its slot, which reuses the code the KV-cache parity tests already
cover. Chunked/batched prefill is the obvious next step and is not needed at this
model size.
"""

from __future__ import annotations

import collections
import queue
import threading
import time
from collections.abc import Iterator
from dataclasses import dataclass, field

import torch

from ..model import KVCache, LanguageModel, SlotKVCache, SlotStep
from ..model.transformer import _sample
from .observability import BATCH_SIZE, QUEUE_DEPTH


class EngineBusy(RuntimeError):
    """Raised when admission control rejects a request: no free slot or queue space
    within the queue timeout."""


_DONE = object()


@dataclass
class Job:
    ids: list[int]
    max_new_tokens: int
    temperature: float
    top_k: int | None
    top_p: float | None
    repetition_penalty: float
    generator: torch.Generator | None
    deadline: float
    out: queue.Queue = field(default_factory=queue.Queue)
    cancelled: threading.Event = field(default_factory=threading.Event)
    slot: int = -1
    produced: int = 0
    history: torch.Tensor | None = None
    last: torch.Tensor | None = None

    def tokens(self) -> Iterator[int]:
        """Token ids as the scheduler produces them. Raises what the scheduler raised."""
        try:
            while True:
                item = self.out.get()
                if item is _DONE:
                    return
                if isinstance(item, BaseException):
                    raise item
                yield item
        finally:
            # A consumer that stops early (client disconnected) frees the slot at the
            # next step instead of generating into the void until max_new_tokens.
            self.cancelled.set()


class BatchScheduler:
    def __init__(
        self,
        model: LanguageModel,
        max_batch: int = 8,
        max_queue: int = 32,
        queue_timeout_s: float = 10.0,
    ) -> None:
        self.model = model
        self.max_batch = max_batch
        self.max_queue = max_queue
        self.queue_timeout_s = queue_timeout_s
        self.device = next(model.parameters()).device
        dtype = next(model.parameters()).dtype
        self.cache = SlotKVCache.allocate(model.config, max_batch, device=self.device, dtype=dtype)
        self._waiting: collections.deque[Job] = collections.deque()
        self._active: dict[int, Job] = {}
        # Lowest slot first, and decode in slot order: the active set is then usually
        # one contiguous run, which SlotStep reads as a view instead of a gather.
        self._free = list(range(max_batch))
        self._cv = threading.Condition()
        self._stop = False
        self._thread = threading.Thread(target=self._run, name="lmops-scheduler", daemon=True)
        self._thread.start()

    # -- public ----------------------------------------------------------------

    def submit(self, job_args: dict) -> Job:
        job = Job(deadline=time.monotonic() + self.queue_timeout_s, **job_args)
        with self._cv:
            if len(self._waiting) >= self.max_queue:
                raise EngineBusy(
                    f"server busy: {len(self._active)} requests generating and "
                    f"{len(self._waiting)} waiting; try again shortly"
                )
            self._waiting.append(job)
            QUEUE_DEPTH.set(len(self._waiting))
            self._cv.notify()
        return job

    def close(self) -> None:
        with self._cv:
            self._stop = True
            self._cv.notify()
        self._thread.join(timeout=5)

    @property
    def active(self) -> int:
        return len(self._active)

    # -- loop ------------------------------------------------------------------

    def _run(self) -> None:
        with torch.inference_mode():
            while True:
                with self._cv:
                    while not self._stop and not self._waiting and not self._active:
                        self._cv.wait()
                    if self._stop:
                        break
                    admitted = self._take_admissible()
                for job in admitted:
                    self._prefill(job)
                self._retire_cancelled()
                if self._active:
                    self._decode()
        for job in [*self._active.values(), *self._waiting]:
            job.out.put(EngineBusy("server is shutting down"))

    def _take_admissible(self) -> list[Job]:
        """Pop waiting jobs into free slots. Called with the lock held."""
        now = time.monotonic()
        admitted = []
        while self._waiting and self._free:
            job = self._waiting.popleft()
            if job.cancelled.is_set():
                continue
            if now > job.deadline:
                job.out.put(
                    EngineBusy(
                        f"server busy: waited {self.queue_timeout_s:.0f}s for one of "
                        f"{self.max_batch} batch slots"
                    )
                )
                continue
            self._free.sort()
            job.slot = self._free.pop(0)
            self._active[job.slot] = job
            admitted.append(job)
        QUEUE_DEPTH.set(len(self._waiting))
        return admitted

    def _prefill(self, job: Job) -> None:
        try:
            idx = torch.tensor([job.ids], dtype=torch.long, device=self.device)
            single = KVCache.allocate(
                self.model.config,
                1,
                max_seq_len=len(job.ids),
                device=self.device,
                dtype=self.cache.keys[0].dtype,
            )
            logits, _ = self.model(idx, cache=single, last_only=True)
            self.cache.load(job.slot, single)
            job.history = idx
            self._emit(job, logits[:, -1, :])
        except Exception as exc:  # pragma: no cover - defensive
            self._finish(job, exc)

    def _decode(self) -> None:
        jobs = sorted(self._active.values(), key=lambda j: j.slot)
        slots = torch.tensor([j.slot for j in jobs], dtype=torch.long, device=self.device)
        tokens = torch.cat([j.last for j in jobs if j.last is not None], dim=0)
        BATCH_SIZE.observe(len(jobs))
        try:
            logits = self.model.decode_step(tokens, SlotStep(self.cache, slots))
        except Exception as exc:  # pragma: no cover - defensive
            for job in jobs:
                self._finish(job, exc)
            return
        for row, job in enumerate(jobs):
            self._emit(job, logits[row : row + 1, -1, :])

    def _emit(self, job: Job, logits: torch.Tensor) -> None:
        """Sample this job's next token, hand it to the consumer, retire if done."""
        assert job.history is not None, "a job is prefilled before it is sampled"
        token = _sample(
            logits,
            temperature=job.temperature,
            top_k=job.top_k,
            top_p=job.top_p,
            repetition_penalty=job.repetition_penalty,
            history=job.history,
            generator=job.generator,
        )
        job.history = torch.cat([job.history, token], dim=1)
        job.last = token
        job.produced += 1
        job.out.put(int(token[0, 0]))
        # Retire straight after the last requested token: running it through the
        # model again would compute logits nobody reads.
        if job.produced >= job.max_new_tokens:
            self._finish(job)

    def _retire_cancelled(self) -> None:
        for job in [j for j in self._active.values() if j.cancelled.is_set()]:
            self._finish(job)

    def _finish(self, job: Job, error: BaseException | None = None) -> None:
        job.out.put(error if error is not None else _DONE)
        with self._cv:
            if self._active.pop(job.slot, None) is not None:
                self._free.append(job.slot)
