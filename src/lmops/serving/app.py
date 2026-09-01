"""FastAPI application: generation, streaming, health probes and metrics.

``/healthz`` and ``/readyz`` are deliberately different. Liveness answers "is the
process wedged?" and must never touch the model, or a saturated server gets killed by
its own orchestrator mid-request. Readiness answers "should traffic be routed here?"
and does depend on the checkpoint being loaded.
"""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from starlette.concurrency import iterate_in_threadpool, run_in_threadpool

from .. import __version__
from .engine import EngineBusy, GenerationEngine
from .observability import (
    IN_FLIGHT,
    REGISTRY,
    record_failure,
    record_success,
    set_model_info,
)
from .schemas import GenerateRequest, GenerateResponse, HealthResponse, ReadyResponse

logger = logging.getLogger("lmops.serving")

_engine: GenerationEngine | None = None


def get_engine() -> GenerationEngine:
    """FastAPI dependency: the loaded engine, or 503 while it is still starting."""
    if _engine is None:
        raise HTTPException(status_code=503, detail="model is not loaded yet")
    return _engine


def set_engine(engine: GenerationEngine | None) -> None:
    """Install an engine directly. Used by the CLI and by tests."""
    global _engine
    _engine = engine
    if engine is not None:
        set_model_info(
            checkpoint=engine.info.checkpoint,
            device=engine.info.device,
            parameters=str(engine.info.parameters),
            max_seq_len=str(engine.info.max_seq_len),
            vocab_size=str(engine.info.vocab_size),
            step=str(engine.info.step),
            version=__version__,
        )


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Load the checkpoint once at start-up rather than on the first request."""
    if _engine is None:
        try:
            set_engine(GenerationEngine.from_env())
            logger.info("loaded checkpoint %s", get_engine().info.checkpoint)
        except Exception as exc:  # pragma: no cover - depends on deployment env
            # Start anyway so /healthz answers and the orchestrator reports a clear
            # "not ready" instead of a crash-loop with no diagnostics.
            logger.error("failed to load model at start-up: %s", exc)
    yield
    set_engine(None)


def create_app() -> FastAPI:
    app = FastAPI(
        title="lmops inference API",
        version=__version__,
        summary="Text generation from a decoder-only transformer trained with lmops.",
        lifespan=lifespan,
    )

    @app.get("/healthz", response_model=HealthResponse, tags=["ops"])
    async def healthz() -> HealthResponse:
        """Liveness: the process is up. Never touches the model."""
        return HealthResponse(status="ok", version=__version__)

    @app.get("/readyz", response_model=ReadyResponse, tags=["ops"])
    async def readyz() -> Response:
        """Readiness: a checkpoint is loaded and the engine can serve traffic."""
        if _engine is None:
            return JSONResponse(
                status_code=503,
                content={
                    "status": "not_ready",
                    "model_loaded": False,
                    "device": "",
                    "parameters": 0,
                    "max_seq_len": 0,
                    "vocab_size": 0,
                },
            )
        info = _engine.info
        return JSONResponse(
            content=ReadyResponse(
                status="ready",
                model_loaded=True,
                device=info.device,
                parameters=info.parameters,
                max_seq_len=info.max_seq_len,
                vocab_size=info.vocab_size,
                checkpoint=info.checkpoint,
                step=info.step,
            ).model_dump()
        )

    @app.get("/metrics", tags=["ops"])
    async def metrics() -> Response:
        return Response(content=generate_latest(REGISTRY), media_type=CONTENT_TYPE_LATEST)

    @app.post(
        "/v1/generate",
        response_model=GenerateResponse,
        tags=["inference"],
        responses={503: {"description": "Model not loaded or all slots busy"}},
    )
    async def generate(
        request: GenerateRequest, engine: GenerationEngine = Depends(get_engine)
    ) -> Response:
        """Generate a completion and return it in one response body."""
        if request.stream:
            return await _stream_response(request, engine)

        IN_FLIGHT.inc()
        try:
            result = await run_in_threadpool(engine.generate, request)
        except EngineBusy as exc:
            record_failure("generate", "busy")
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except ValueError as exc:
            record_failure("generate", "bad_request")
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except Exception as exc:
            record_failure("generate", "error")
            logger.exception("generation failed")
            raise HTTPException(status_code=500, detail="generation failed") from exc
        finally:
            IN_FLIGHT.dec()

        record_success(
            "generate",
            result.latency_ms,
            result.ttft_ms,
            result.usage.prompt_tokens,
            result.usage.completion_tokens,
        )
        return JSONResponse(content=result.model_dump())

    @app.post("/v1/generate/stream", tags=["inference"])
    async def generate_stream(
        request: GenerateRequest, engine: GenerationEngine = Depends(get_engine)
    ) -> Response:
        """Stream the completion as server-sent events."""
        return await _stream_response(request, engine)

    @app.exception_handler(EngineBusy)
    async def busy_handler(_: Request, exc: EngineBusy) -> JSONResponse:
        record_failure("generate", "busy")
        return JSONResponse(status_code=503, content={"detail": str(exc)})

    return app


async def _stream_response(request: GenerateRequest, engine: GenerationEngine) -> Response:
    """Wrap the engine's blocking generator in an SSE response.

    The generator is **primed** before the response is returned. ``engine.stream`` is a
    generator function, so merely calling it executes none of its body -- admission
    control and prompt validation would not run until the first event was pulled, by
    which point the status line is already sent and a rejected request has no way to
    become a 503. Pulling the first event here costs one prefill of latency and buys
    correct status codes: 503 when saturated, 422 when the prompt cannot fit.

    ``iterate_in_threadpool`` then keeps the remaining blocking torch work off the event
    loop; without it a single long generation stalls every other connection, health
    probes included.
    """
    events = engine.stream(request)
    try:
        first = await run_in_threadpool(lambda: next(events, None))
    except EngineBusy as exc:
        record_failure("generate_stream", "busy")
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        record_failure("generate_stream", "bad_request")
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        record_failure("generate_stream", "error")
        logger.exception("streaming generation failed to start")
        raise HTTPException(status_code=500, detail="generation failed") from exc

    def observe(event: dict) -> None:
        if event.get("done"):
            usage = event["usage"]
            record_success(
                "generate_stream",
                event["latency_ms"],
                event["ttft_ms"],
                usage["prompt_tokens"],
                usage["completion_tokens"],
            )

    def encode(event: dict) -> bytes:
        return f"data: {json.dumps(event)}\n\n".encode()

    async def event_source() -> AsyncIterator[bytes]:
        IN_FLIGHT.inc()
        try:
            if first is not None:
                observe(first)
                yield encode(first)
                async for event in iterate_in_threadpool(events):
                    observe(event)
                    yield encode(event)
            yield b"data: [DONE]\n\n"
        except Exception as exc:  # pragma: no cover - defensive
            # The status line is already sent, so a mid-stream failure can only be
            # reported inside the stream.
            record_failure("generate_stream", "error")
            logger.exception("streaming generation failed")
            yield encode({"error": str(exc)})
        finally:
            # Closing the generator runs its `finally`, which releases the concurrency
            # slot. Without this, a client that disconnects mid-stream leaks a slot
            # until the generator happens to be garbage collected.
            events.close()
            IN_FLIGHT.dec()

    return StreamingResponse(
        event_source(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


app = create_app()
