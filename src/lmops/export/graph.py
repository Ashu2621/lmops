"""Export the model to TorchScript and ONNX, and verify the exports numerically.

Scope note, stated up front because it is a real limitation rather than an oversight:
only the **full-sequence scoring graph** (``token ids -> logits``) is exported. Cached
incremental decode would require the KV cache tensors to become graph inputs and
outputs, which turns a 40-line exporter into a stateful-session contract that every
downstream runtime implements differently. The exported graph is the right artifact
for scoring, ranking and embedding services; the Python server keeps the cache for
interactive generation.

Every exporter here is paired with a parity check, because an export that loads is not
the same thing as an export that is correct -- a silently wrong opset conversion still
produces a perfectly valid ``.onnx`` file.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor, nn

from ..model import LanguageModel


class LogitsOnly(nn.Module):
    """Adapter exposing ``ids -> logits``.

    Tracers and ONNX both dislike optional arguments and ``(tensor, None)`` returns,
    so the export boundary is deliberately a single tensor in, single tensor out.
    """

    def __init__(self, model: LanguageModel) -> None:
        super().__init__()
        self.model = model

    def forward(self, idx: Tensor) -> Tensor:
        logits, _ = self.model(idx)
        return logits


@dataclass
class ParityResult:
    """Outcome of comparing an exported graph against the eager model."""

    backend: str
    max_abs_diff: float
    tolerance: float
    passed: bool

    def render(self) -> str:
        verdict = "PASS" if self.passed else "FAIL"
        return (
            f"[{verdict}] {self.backend}: max |eager - exported| = "
            f"{self.max_abs_diff:.3e} (tolerance {self.tolerance:.1e})"
        )


def _example_input(model: LanguageModel, batch_size: int, seq_len: int, seed: int = 0) -> Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randint(0, model.config.vocab_size, (batch_size, seq_len), generator=generator)


# -- TorchScript -------------------------------------------------------------


def export_torchscript(
    model: LanguageModel,
    path: str | Path,
    batch_size: int = 1,
    seq_len: int = 32,
) -> Path:
    """Trace the scoring graph and save a TorchScript archive."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    model.eval()
    wrapper = LogitsOnly(model).eval()
    example = _example_input(model, batch_size, seq_len)
    with torch.no_grad():
        traced = torch.jit.trace(wrapper, example, strict=False)
    traced = torch.jit.freeze(traced)
    traced.save(str(path))
    return path


def verify_torchscript(
    model: LanguageModel,
    path: str | Path,
    batch_size: int = 2,
    seq_len: int = 24,
    tolerance: float = 1e-4,
) -> ParityResult:
    """Compare traced output against eager output on a *different* shape than tracing.

    Using a new batch and sequence length is the point: a trace that baked in the
    example shape fails here rather than in production.
    """
    model.eval()
    loaded = torch.jit.load(str(path))
    example = _example_input(model, batch_size, seq_len, seed=1)
    with torch.no_grad():
        eager, _ = model(example)
        exported = loaded(example)
    diff = float((eager - exported).abs().max().item())
    return ParityResult("torchscript", diff, tolerance, diff <= tolerance)


# -- ONNX --------------------------------------------------------------------


def export_onnx(
    model: LanguageModel,
    path: str | Path,
    batch_size: int = 1,
    seq_len: int = 32,
    opset: int = 17,
) -> Path:
    """Export the scoring graph to ONNX with dynamic batch and sequence axes."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    model.eval()
    wrapper = LogitsOnly(model).eval()
    example = _example_input(model, batch_size, seq_len)

    torch.onnx.export(
        wrapper,
        (example,),
        str(path),
        input_names=["input_ids"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "sequence"},
            "logits": {0: "batch", 1: "sequence"},
        },
        opset_version=opset,
        do_constant_folding=True,
    )
    return path


def verify_onnx(
    model: LanguageModel,
    path: str | Path,
    batch_size: int = 2,
    seq_len: int = 24,
    tolerance: float = 1e-3,
) -> ParityResult:
    """Run the ONNX graph under onnxruntime and compare against eager output.

    The tolerance is looser than TorchScript's on purpose: onnxruntime is free to fuse
    and reassociate float operations, so bit-identical output is not a reasonable bar.
    """
    try:
        import onnxruntime as ort
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "onnxruntime is required to verify an ONNX export; install lmops[export]"
        ) from exc

    model.eval()
    example = _example_input(model, batch_size, seq_len, seed=2)
    with torch.no_grad():
        eager, _ = model(example)

    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    (exported,) = session.run(["logits"], {"input_ids": example.numpy()})
    diff = float((eager - torch.from_numpy(exported)).abs().max().item())
    return ParityResult("onnx", diff, tolerance, diff <= tolerance)


# -- Quantisation ------------------------------------------------------------


def quantize_dynamic_int8(model: LanguageModel) -> nn.Module:
    """Dynamically quantise every ``nn.Linear`` to int8 for CPU inference.

    Dynamic quantisation is the right first move for a transformer on CPU: weights are
    stored as int8 while activations are quantised per-batch at runtime, so no
    calibration dataset is needed. Always re-measure perplexity afterwards -- the
    speedup is real, and so is the quality cost.
    """
    model.eval()
    return torch.ao.quantization.quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)


def state_dict_size_mb(module: nn.Module) -> float:
    """Approximate on-disk size of a module's parameters and buffers, in MiB."""
    total = sum(p.numel() * p.element_size() for p in module.parameters())
    total += sum(b.numel() * b.element_size() for b in module.buffers())
    return total / (1024 * 1024)
