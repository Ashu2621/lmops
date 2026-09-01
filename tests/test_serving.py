"""HTTP API: contracts, validation, streaming, health probes and admission control."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from lmops.serving.app import create_app, set_engine
from lmops.serving.engine import EngineBusy, GenerationEngine, IncrementalDecoder
from lmops.serving.schemas import GenerateRequest


@pytest.fixture
def engine(served_checkpoint: tuple[Path, Path]) -> GenerationEngine:
    checkpoint, tokenizer = served_checkpoint
    return GenerationEngine(checkpoint, tokenizer, device="cpu", max_concurrency=2)


@pytest.fixture
def client(engine: GenerationEngine):
    set_engine(engine)
    app = create_app()
    with TestClient(app) as test_client:
        yield test_client
    set_engine(None)


# -- engine -------------------------------------------------------------------


def test_engine_reports_model_facts(engine: GenerationEngine) -> None:
    assert engine.info.parameters > 0
    assert engine.info.vocab_size == engine.tokenizer.vocab_size
    assert engine.info.device == "cpu"


def test_engine_rejects_a_mismatched_tokenizer(
    served_checkpoint: tuple[Path, Path], tmp_path: Path
) -> None:
    """Loading a checkpoint against the wrong tokenizer is a silent-corruption bug:
    every generated token would be decoded through the wrong table."""
    from lmops.data import ByteBPETokenizer

    checkpoint, _ = served_checkpoint
    other = ByteBPETokenizer.train("a b c d e f g", vocab_size=300)
    other_path = tmp_path / "other.json"
    other.save(other_path)

    with pytest.raises(ValueError, match="different runs"):
        GenerationEngine(checkpoint, other_path, device="cpu")


def test_engine_left_truncates_an_overlong_prompt(engine: GenerationEngine) -> None:
    limit = engine.model.config.max_seq_len
    ids = engine._encode_prompt("word " * 5_000, max_new_tokens=8)
    assert len(ids) == limit - 8


def test_engine_rejects_a_budget_that_leaves_no_room(engine: GenerationEngine) -> None:
    with pytest.raises(ValueError, match="leaves no room"):
        engine._encode_prompt("hello", max_new_tokens=engine.model.config.max_seq_len)


def test_seeded_requests_are_reproducible(engine: GenerationEngine) -> None:
    request = GenerateRequest(prompt="The", max_new_tokens=12, temperature=1.0, seed=42)
    assert engine.generate(request).completion == engine.generate(request).completion


def test_admission_control_rejects_when_saturated(engine: GenerationEngine) -> None:
    engine.queue_timeout_s = 0.05
    engine._slots.acquire()
    engine._slots.acquire()
    try:
        with pytest.raises(EngineBusy, match="slots busy"):
            engine.generate(GenerateRequest(prompt="hi", max_new_tokens=2))
    finally:
        engine._slots.release()
        engine._slots.release()


def test_slots_are_released_after_a_failure(engine: GenerationEngine) -> None:
    """A raising request must not leak its semaphore slot."""
    bad = GenerateRequest(prompt="x", max_new_tokens=engine.model.config.max_seq_len)
    with pytest.raises(ValueError):
        engine.generate(bad)
    engine.generate(GenerateRequest(prompt="x", max_new_tokens=2))


# -- incremental UTF-8 decoding ----------------------------------------------


def test_incremental_decoder_reassembles_split_characters(engine: GenerationEngine) -> None:
    """A multi-byte character split across two tokens must not become mojibake."""
    tokenizer = engine.tokenizer
    text = "héllo — wörld 🚀"
    ids = tokenizer.encode(text)
    decoder = IncrementalDecoder(tokenizer)
    assembled = "".join(decoder.push(i) for i in ids) + decoder.flush()
    assert assembled == text


def test_incremental_decoder_holds_back_incomplete_bytes(engine: GenerationEngine) -> None:
    tokenizer = engine.tokenizer
    ids = tokenizer.encode("🚀")
    decoder = IncrementalDecoder(tokenizer)
    emitted = [decoder.push(i) for i in ids]
    # Nothing may be emitted until the final byte of the character arrives.
    assert "".join(emitted) == "🚀"
    assert emitted[-1] == "🚀" or "".join(emitted[:-1]) == ""
    assert decoder.flush() == ""


# -- HTTP ---------------------------------------------------------------------


def test_healthz_does_not_depend_on_the_model() -> None:
    """Liveness must answer even with no engine loaded, or an overloaded server gets
    killed by its own orchestrator."""
    set_engine(None)
    app = create_app()
    client = TestClient(app)
    response = client.get("/healthz")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_readyz_is_503_without_a_model() -> None:
    set_engine(None)
    client = TestClient(create_app())
    response = client.get("/readyz")
    assert response.status_code == 503
    assert response.json()["model_loaded"] is False


def test_readyz_describes_the_loaded_model(client: TestClient) -> None:
    body = client.get("/readyz").json()
    assert body["status"] == "ready"
    assert body["model_loaded"] is True
    assert body["parameters"] > 0
    assert body["step"] == 1


def test_generate_returns_a_completion_and_usage(client: TestClient) -> None:
    response = client.post(
        "/v1/generate", json={"prompt": "The river", "max_new_tokens": 12, "seed": 1}
    )
    assert response.status_code == 200
    body = response.json()
    assert body["usage"]["completion_tokens"] == 12
    assert body["usage"]["total_tokens"] == (
        body["usage"]["prompt_tokens"] + body["usage"]["completion_tokens"]
    )
    assert body["text"].startswith("The river")
    assert body["latency_ms"] > 0


def test_greedy_requests_are_deterministic_over_http(client: TestClient) -> None:
    payload = {"prompt": "Once", "max_new_tokens": 10, "temperature": 0.0}
    first = client.post("/v1/generate", json=payload).json()["completion"]
    second = client.post("/v1/generate", json=payload).json()["completion"]
    assert first == second


@pytest.mark.parametrize(
    "payload",
    [
        {"prompt": "", "max_new_tokens": 4},
        {"prompt": "hi", "max_new_tokens": 0},
        {"prompt": "hi", "max_new_tokens": 100_000},
        {"prompt": "hi", "temperature": -1.0},
        {"prompt": "hi", "top_p": 0.0},
        {"prompt": "hi", "top_p": 1.5},
        {"max_new_tokens": 4},
    ],
)
def test_invalid_requests_are_rejected_with_422(client: TestClient, payload: dict) -> None:
    assert client.post("/v1/generate", json=payload).status_code == 422


def test_a_prompt_that_cannot_fit_returns_422(client: TestClient) -> None:
    """Schema-valid but impossible: the model's context cannot hold the request."""
    response = client.post("/v1/generate", json={"prompt": "hi", "max_new_tokens": 1024})
    assert response.status_code == 422
    assert "no room" in response.json()["detail"]


def test_streaming_emits_deltas_then_a_done_event(client: TestClient) -> None:
    with client.stream(
        "POST", "/v1/generate/stream", json={"prompt": "The", "max_new_tokens": 8, "seed": 3}
    ) as response:
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        events = []
        for line in response.iter_lines():
            if line.startswith("data: ") and line != "data: [DONE]":
                events.append(json.loads(line[6:]))

    assert events, "no events received"
    assert events[-1]["done"] is True
    assert events[-1]["usage"]["completion_tokens"] == 8
    assert any("delta" in e for e in events[:-1])


def test_streaming_text_matches_the_non_streaming_response(client: TestClient) -> None:
    payload = {"prompt": "The", "max_new_tokens": 10, "temperature": 0.0}
    whole = client.post("/v1/generate", json=payload).json()["completion"]

    with client.stream("POST", "/v1/generate/stream", json=payload) as response:
        streamed = "".join(
            json.loads(line[6:]).get("delta", "")
            for line in response.iter_lines()
            if line.startswith("data: ") and line != "data: [DONE]"
        )
    assert streamed == whole


def test_stream_flag_on_the_plain_endpoint_streams(client: TestClient) -> None:
    response = client.post(
        "/v1/generate", json={"prompt": "The", "max_new_tokens": 4, "stream": True}
    )
    assert response.headers["content-type"].startswith("text/event-stream")


def test_metrics_expose_prometheus_counters(client: TestClient) -> None:
    client.post("/v1/generate", json={"prompt": "The", "max_new_tokens": 4})
    body = client.get("/metrics").text
    assert "lmops_requests_total" in body
    assert "lmops_request_latency_seconds" in body
    assert "lmops_ttft_seconds" in body
    assert "lmops_tokens_generated_total" in body


def test_openapi_schema_is_served(client: TestClient) -> None:
    schema = client.get("/openapi.json").json()
    assert "/v1/generate" in schema["paths"]
    assert "/v1/generate/stream" in schema["paths"]


def test_streaming_returns_503_when_saturated(client: TestClient, engine: GenerationEngine) -> None:
    """A rejected stream must be a 503, not a 200 whose body happens to contain an error.

    ``engine.stream`` is a generator function, so admission control does not run until
    the first event is pulled. The route primes the generator before returning the
    response precisely so this status code is still changeable.
    """
    engine.queue_timeout_s = 0.05
    engine._slots.acquire()
    engine._slots.acquire()
    try:
        response = client.post("/v1/generate/stream", json={"prompt": "hi", "max_new_tokens": 4})
        assert response.status_code == 503
        assert "busy" in response.json()["detail"]
    finally:
        engine._slots.release()
        engine._slots.release()


def test_streaming_returns_422_for_an_impossible_budget(client: TestClient) -> None:
    response = client.post("/v1/generate/stream", json={"prompt": "hi", "max_new_tokens": 1024})
    assert response.status_code == 422
    assert "no room" in response.json()["detail"]


def test_streaming_releases_its_slot(client: TestClient, engine: GenerationEngine) -> None:
    """After a completed stream every concurrency slot must be back in the pool."""
    for _ in range(3):
        with client.stream(
            "POST", "/v1/generate/stream", json={"prompt": "The", "max_new_tokens": 4}
        ) as response:
            assert response.status_code == 200
            response.read()

    acquired = [engine._slots.acquire(blocking=False) for _ in range(engine.max_concurrency)]
    try:
        assert all(acquired), "a streaming request leaked a concurrency slot"
    finally:
        for ok in acquired:
            if ok:
                engine._slots.release()
