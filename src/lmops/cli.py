"""``lmops`` command line: prepare, train, eval, sample, export, bench, serve.

Every stage is reachable both as a library call and as a subcommand. The CLI stays a
thin argument-parsing shell -- no logic lives here that a test would want to call.
"""

from __future__ import annotations

import json
from pathlib import Path

import typer

app = typer.Typer(
    name="lmops",
    help="Train, evaluate, export, serve and observe a small language model.",
    no_args_is_help=True,
    add_completion=False,
)


@app.command()
def prepare(
    input_path: Path = typer.Option(
        Path("data/raw/tinyshakespeare.txt"), "--input", "-i", help="Raw UTF-8 corpus."
    ),
    data_dir: Path = typer.Option(Path("data/processed"), "--out", "-o"),
    vocab_size: int = typer.Option(1024, help="Total vocabulary, including 256 byte tokens."),
    val_fraction: float = typer.Option(0.1, min=0.0, max=0.5),
    tokenizer_chars: int = typer.Option(500_000, help="Prefix length used to learn merges."),
) -> None:
    """Train a byte-level BPE tokenizer and write train/val token shards."""
    from .data import prepare_corpus

    result = prepare_corpus(
        input_path,
        data_dir,
        vocab_size=vocab_size,
        val_fraction=val_fraction,
        tokenizer_train_chars=tokenizer_chars,
        progress=True,
    )
    typer.echo(json.dumps(result.to_dict(), indent=2))


@app.command()
def train(
    config: Path = typer.Option(Path("configs/small.yaml"), "--config", "-c"),
    max_steps: int = typer.Option(None, help="Override train.max_steps."),
    out_dir: str = typer.Option(None, help="Override train.out_dir."),
    device: str = typer.Option(None, help="Override train.device (auto/cpu/cuda)."),
    resume: str = typer.Option(None, help="Checkpoint to resume from."),
    seed: int = typer.Option(None, help="Override train.seed."),
) -> None:
    """Train a model from a YAML config."""
    from .config import load_config
    from .training import Trainer

    cfg = load_config(
        config,
        **{
            "train.max_steps": max_steps,
            "train.out_dir": out_dir,
            "train.device": device,
            "train.resume": resume,
            "train.seed": seed,
        },
    )
    state = Trainer(cfg).fit()
    typer.echo(
        json.dumps(
            {
                "steps": state.step,
                "best_val_loss": round(state.best_val_loss, 5),
                "tokens_seen": state.tokens_seen,
                "out_dir": cfg.train.out_dir,
            },
            indent=2,
        )
    )


@app.command(name="eval")
def evaluate(
    checkpoint: Path = typer.Option(Path("runs/default/best.pt"), "--checkpoint", "-c"),
    data_dir: Path = typer.Option(Path("data/processed"), "--data"),
    split: str = typer.Option("val"),
    batch_size: int = typer.Option(8),
    max_batches: int = typer.Option(None, help="Cap the number of eval batches."),
    device: str = typer.Option("auto"),
    report: Path = typer.Option(None, help="Also write the report as JSON here."),
) -> None:
    """Score a checkpoint: loss, perplexity, bits/byte and top-1 accuracy."""
    from .evaluation import evaluate_checkpoint, write_report

    result = evaluate_checkpoint(
        checkpoint,
        data_dir,
        split=split,
        batch_size=batch_size,
        max_batches=max_batches,
        device=device,
    )
    typer.echo(result.render())
    if report:
        write_report(result, report)
        typer.echo(f"\nwrote {report}")


@app.command()
def sample(
    checkpoint: Path = typer.Option(Path("runs/default/best.pt"), "--checkpoint", "-c"),
    tokenizer: Path = typer.Option(Path("data/processed/tokenizer.json"), "--tokenizer", "-t"),
    prompt: str = typer.Option("\n", "--prompt", "-p"),
    max_new_tokens: int = typer.Option(200, "--tokens", "-n"),
    temperature: float = typer.Option(0.8),
    top_k: int = typer.Option(40),
    top_p: float = typer.Option(0.95),
    seed: int = typer.Option(None),
    device: str = typer.Option("auto"),
) -> None:
    """Generate a continuation from a checkpoint."""
    from .data import ByteBPETokenizer
    from .evaluation import sample_text
    from .training import build_model_from_checkpoint, resolve_device

    dev = resolve_device(device)
    model, _ = build_model_from_checkpoint(checkpoint, device=dev)
    tok = ByteBPETokenizer.load(tokenizer)
    typer.echo(
        sample_text(
            model,
            tok,
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k or None,
            top_p=top_p,
            seed=seed,
            device=dev,
        )
    )


@app.command(name="export")
def export_cmd(
    checkpoint: Path = typer.Option(Path("runs/default/best.pt"), "--checkpoint", "-c"),
    out_dir: Path = typer.Option(Path("artifacts"), "--out", "-o"),
    fmt: str = typer.Option("all", "--format", "-f", help="torchscript | onnx | all"),
    seq_len: int = typer.Option(32, help="Example sequence length used for tracing."),
    opset: int = typer.Option(17),
) -> None:
    """Export the scoring graph and verify it against the eager model."""
    from .export import (
        export_onnx,
        export_torchscript,
        verify_onnx,
        verify_torchscript,
    )
    from .training import build_model_from_checkpoint

    model, _ = build_model_from_checkpoint(checkpoint, device="cpu")
    out_dir.mkdir(parents=True, exist_ok=True)
    failures = 0

    if fmt in ("torchscript", "all"):
        path = export_torchscript(model, out_dir / "model.pt", seq_len=seq_len)
        result = verify_torchscript(model, path)
        typer.echo(f"{path}\n  {result.render()}")
        failures += not result.passed

    if fmt in ("onnx", "all"):
        path = export_onnx(model, out_dir / "model.onnx", seq_len=seq_len, opset=opset)
        result = verify_onnx(model, path)
        typer.echo(f"{path}\n  {result.render()}")
        failures += not result.passed

    if failures:
        raise typer.Exit(code=1)


@app.command()
def bench(
    checkpoint: Path = typer.Option(Path("runs/default/best.pt"), "--checkpoint", "-c"),
    prompt_tokens: int = typer.Option(32),
    generated_tokens: int = typer.Option(64),
    batch_size: int = typer.Option(1),
    runs: int = typer.Option(3),
    compare: bool = typer.Option(True, help="Also measure decoding without the KV cache."),
    device: str = typer.Option("cpu"),
    report: Path = typer.Option(None, help="Write results as JSON here."),
) -> None:
    """Measure time-to-first-token and decode throughput."""
    from .evaluation import benchmark_generation, compare_cache
    from .training import build_model_from_checkpoint

    model, _ = build_model_from_checkpoint(checkpoint, device=device)
    results = (
        compare_cache(
            model,
            prompt_tokens=prompt_tokens,
            generated_tokens=generated_tokens,
            batch_size=batch_size,
            runs=runs,
            device=device,
        )
        if compare
        else [
            benchmark_generation(
                model,
                prompt_tokens=prompt_tokens,
                generated_tokens=generated_tokens,
                batch_size=batch_size,
                runs=runs,
                device=device,
            )
        ]
    )
    for result in results:
        typer.echo(result.render())
    if len(results) == 2 and results[1].total_ms_p50 > 0:
        typer.echo(f"\nKV-cache speedup: {results[1].total_ms_p50 / results[0].total_ms_p50:.2f}x")
    if report:
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text(json.dumps([r.to_dict() for r in results], indent=2), encoding="utf-8")
        typer.echo(f"wrote {report}")


@app.command()
def serve(
    checkpoint: Path = typer.Option(Path("runs/default/best.pt"), "--checkpoint", "-c"),
    tokenizer: Path = typer.Option(Path("data/processed/tokenizer.json"), "--tokenizer", "-t"),
    host: str = typer.Option("127.0.0.1"),
    port: int = typer.Option(8000),
    device: str = typer.Option("auto"),
    max_concurrency: int = typer.Option(4),
) -> None:
    """Serve the model over HTTP."""
    import uvicorn

    from .serving.app import app as fastapi_app
    from .serving.app import set_engine
    from .serving.engine import GenerationEngine

    set_engine(
        GenerationEngine(
            checkpoint=checkpoint,
            tokenizer_path=tokenizer,
            device=device,
            max_concurrency=max_concurrency,
        )
    )
    uvicorn.run(fastapi_app, host=host, port=port, log_level="info")


@app.command()
def info(
    checkpoint: Path = typer.Option(Path("runs/default/best.pt"), "--checkpoint", "-c"),
) -> None:
    """Print a checkpoint's architecture, size and recorded metrics."""
    from .training import build_model_from_checkpoint

    model, meta = build_model_from_checkpoint(checkpoint, device="cpu")
    cfg = model.config
    typer.echo(
        json.dumps(
            {
                "checkpoint": str(checkpoint),
                "step": meta.get("step"),
                "metrics": meta.get("metrics"),
                "parameters": model.num_parameters(),
                "parameters_non_embedding": model.num_parameters(include_embeddings=False),
                "architecture": {
                    "n_layer": cfg.n_layer,
                    "n_head": cfg.n_head,
                    "n_kv_head": cfg.n_kv_head,
                    "d_model": cfg.d_model,
                    "d_ff": cfg.d_ff,
                    "max_seq_len": cfg.max_seq_len,
                    "vocab_size": cfg.vocab_size,
                    "tie_embeddings": cfg.tie_embeddings,
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    app()
