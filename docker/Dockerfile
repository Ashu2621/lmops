# Multi-stage build for the lmops inference API.
#
# Two decisions worth calling out:
#
#   1. Torch is installed from the CPU wheel index. The default PyPI wheel drags in
#      several gigabytes of CUDA libraries that a CPU serving container will never
#      load -- that is the difference between a ~700MB image and a ~6GB one.
#   2. The runtime stage copies only the virtualenv and the package. Build tools,
#      tests and the raw corpus never reach the shipped image.

# ---------------------------------------------------------------- builder ----
FROM python:3.12-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /build

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Torch first, from the CPU index, so the dependency resolver never considers the
# CUDA build when the project's own dependencies are installed below.
RUN pip install --index-url https://download.pytorch.org/whl/cpu "torch>=2.2"

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install ".[serve]"

# ---------------------------------------------------------------- runtime ----
FROM python:3.12-slim AS runtime

LABEL org.opencontainers.image.title="lmops" \
      org.opencontainers.image.description="Inference API for a decoder-only LM trained with lmops" \
      org.opencontainers.image.licenses="MIT"

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    LMOPS_CHECKPOINT=/models/best.pt \
    LMOPS_TOKENIZER=/models/tokenizer.json \
    LMOPS_DEVICE=cpu \
    LMOPS_MAX_CONCURRENCY=4 \
    OMP_NUM_THREADS=4

# curl is here for the HEALTHCHECK below and nothing else.
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# Run as an unprivileged user: a model server has no reason to be root.
RUN useradd --create-home --uid 10001 lmops
COPY --from=builder /opt/venv /opt/venv

WORKDIR /app
USER lmops

EXPOSE 8000

# Liveness only -- readiness (is the checkpoint loaded?) is /readyz, which the
# orchestrator should poll separately.
HEALTHCHECK --interval=30s --timeout=3s --start-period=20s --retries=3 \
    CMD curl -fsS http://localhost:8000/healthz || exit 1

CMD ["uvicorn", "lmops.serving.app:app", "--host", "0.0.0.0", "--port", "8000"]
