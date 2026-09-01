"""Request and response models for the HTTP API.

Validation lives here rather than in the handlers so that a bad request is rejected
with a 422 and a field-level message before it ever reaches the GPU/CPU queue. The
bounds are deliberately explicit: an unbounded ``max_new_tokens`` is a denial-of-
service vector on any generation endpoint.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class GenerateRequest(BaseModel):
    """A single text-completion request."""

    prompt: str = Field(..., min_length=1, max_length=8_000, description="Text to continue.")
    max_new_tokens: int = Field(64, ge=1, le=1_024, description="Tokens to generate.")
    temperature: float = Field(0.8, ge=0.0, le=5.0, description="0 selects greedy decoding.")
    top_k: int | None = Field(40, ge=0, description="Keep only the k most likely tokens.")
    top_p: float | None = Field(0.95, gt=0.0, le=1.0, description="Nucleus sampling threshold.")
    repetition_penalty: float = Field(1.0, ge=0.5, le=2.0)
    seed: int | None = Field(None, description="Set for reproducible sampling.")
    stream: bool = Field(False, description="Emit server-sent events instead of one JSON body.")


class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class GenerateResponse(BaseModel):
    text: str
    completion: str
    usage: Usage
    latency_ms: float
    ttft_ms: float
    model: str
    seed: int | None = None


class HealthResponse(BaseModel):
    status: str
    version: str


class ReadyResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str
    parameters: int
    max_seq_len: int
    vocab_size: int
    checkpoint: str | None = None
    step: int | None = None


class ErrorResponse(BaseModel):
    detail: str
