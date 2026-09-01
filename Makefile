.DEFAULT_GOAL := help
PY ?= python
IMAGE ?= lmops:latest
CKPT ?= runs/small/best.pt

.PHONY: help install install-dev prepare train train-gqa eval sample export bench serve \
        lint format typecheck test test-cov check docker docker-run clean

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
	  awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

install: ## Install the package
	$(PY) -m pip install -e .

install-dev: ## Install with dev, serving and export extras
	$(PY) -m pip install -e ".[dev,serve,export]"

# -- pipeline ---------------------------------------------------------------

prepare: ## Train the tokenizer and write token shards
	$(PY) -m lmops.cli prepare --input data/raw/tinyshakespeare.txt --out data/processed --vocab-size 1024

train: ## Train the reference model (configs/small.yaml)
	$(PY) -m lmops.cli train --config configs/small.yaml

train-gqa: ## Train the grouped-query-attention ablation
	$(PY) -m lmops.cli train --config configs/small-gqa.yaml

eval: ## Score the reference checkpoint on the validation shard
	$(PY) -m lmops.cli eval --checkpoint $(CKPT) --data data/processed --report runs/eval.json

sample: ## Generate a sample continuation
	$(PY) -m lmops.cli sample --checkpoint $(CKPT) --prompt "KING RICHARD" --tokens 200

export: ## Export TorchScript + ONNX and verify parity against eager
	$(PY) -m lmops.cli export --checkpoint $(CKPT) --out artifacts

bench: ## Measure TTFT and decode throughput, with and without the KV cache
	$(PY) -m lmops.cli bench --checkpoint $(CKPT) --report runs/bench.json

serve: ## Serve the model on http://127.0.0.1:8000
	$(PY) -m lmops.cli serve --checkpoint $(CKPT) --tokenizer data/processed/tokenizer.json

# -- quality ----------------------------------------------------------------

lint: ## Lint and check formatting
	$(PY) -m ruff check .
	$(PY) -m ruff format --check .

format: ## Auto-fix lint and format
	$(PY) -m ruff check . --fix
	$(PY) -m ruff format .

typecheck: ## Run mypy
	$(PY) -m mypy

test: ## Run the test suite
	$(PY) -m pytest

test-cov: ## Run tests with a coverage report
	$(PY) -m pytest --cov=lmops --cov-report=term-missing --cov-report=xml

check: lint typecheck test ## Everything CI runs

# -- container --------------------------------------------------------------

docker: ## Build the serving image
	docker build -f docker/Dockerfile -t $(IMAGE) .

docker-run: ## Run the serving stack (API + Prometheus)
	docker compose -f docker/docker-compose.yml up --build

clean: ## Remove caches and build artifacts
	rm -rf .pytest_cache .ruff_cache .mypy_cache build dist *.egg-info coverage.xml htmlcov
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
