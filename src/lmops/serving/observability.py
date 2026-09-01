"""Prometheus instrumentation for the serving layer.

Bucket boundaries are chosen for a small CPU-served model rather than copied from a
web-framework default: time-to-first-token and per-request latency for generation live
in the tens-of-milliseconds-to-seconds range, and buckets that all saturate tell you
nothing about the tail.
"""

from __future__ import annotations

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram, Info

REGISTRY = CollectorRegistry(auto_describe=True)

LATENCY_BUCKETS = (0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0)
TTFT_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0)

REQUESTS = Counter(
    "lmops_requests_total",
    "Generation requests by endpoint and outcome.",
    ["endpoint", "status"],
    registry=REGISTRY,
)

REQUEST_LATENCY = Histogram(
    "lmops_request_latency_seconds",
    "Wall-clock latency of a complete generation request.",
    ["endpoint"],
    buckets=LATENCY_BUCKETS,
    registry=REGISTRY,
)

TIME_TO_FIRST_TOKEN = Histogram(
    "lmops_ttft_seconds",
    "Time from request arrival to the first generated token.",
    ["endpoint"],
    buckets=TTFT_BUCKETS,
    registry=REGISTRY,
)

TOKENS_GENERATED = Counter(
    "lmops_tokens_generated_total",
    "Completion tokens produced.",
    registry=REGISTRY,
)

TOKENS_PROMPT = Counter(
    "lmops_tokens_prompt_total",
    "Prompt tokens consumed.",
    registry=REGISTRY,
)

IN_FLIGHT = Gauge(
    "lmops_requests_in_flight",
    "Generation requests currently executing.",
    registry=REGISTRY,
)

REJECTED = Counter(
    "lmops_requests_rejected_total",
    "Requests rejected by admission control.",
    registry=REGISTRY,
)

MODEL_INFO = Info(
    "lmops_model",
    "Static description of the loaded checkpoint.",
    registry=REGISTRY,
)


def record_success(
    endpoint: str,
    latency_ms: float,
    ttft_ms: float,
    prompt_tokens: int,
    completion_tokens: int,
) -> None:
    REQUESTS.labels(endpoint=endpoint, status="ok").inc()
    REQUEST_LATENCY.labels(endpoint=endpoint).observe(latency_ms / 1000.0)
    TIME_TO_FIRST_TOKEN.labels(endpoint=endpoint).observe(ttft_ms / 1000.0)
    TOKENS_PROMPT.inc(prompt_tokens)
    TOKENS_GENERATED.inc(completion_tokens)


def record_failure(endpoint: str, status: str = "error") -> None:
    REQUESTS.labels(endpoint=endpoint, status=status).inc()
    if status == "busy":
        REJECTED.inc()


def set_model_info(**fields: str) -> None:
    MODEL_INFO.info({k: str(v) for k, v in fields.items()})
