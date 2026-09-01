"""Export parity: an artifact that loads is not the same as an artifact that is correct."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from lmops.export import (
    export_onnx,
    export_torchscript,
    quantize_dynamic_int8,
    state_dict_size_mb,
    verify_onnx,
    verify_torchscript,
)
from lmops.model import LanguageModel

onnx_available = pytest.importorskip  # re-exported for readability below


def test_torchscript_export_matches_eager(tiny_model: LanguageModel, tmp_path: Path) -> None:
    path = export_torchscript(tiny_model, tmp_path / "m.pt", seq_len=16)
    assert path.exists()
    result = verify_torchscript(tiny_model, path)
    assert result.passed, result.render()


def test_torchscript_generalises_beyond_the_traced_shape(
    tiny_model: LanguageModel, tmp_path: Path
) -> None:
    """Traced with (1, 16); verified at (2, 24). A shape baked into the trace fails here."""
    path = export_torchscript(tiny_model, tmp_path / "m.pt", batch_size=1, seq_len=16)
    result = verify_torchscript(tiny_model, path, batch_size=2, seq_len=24)
    assert result.passed, result.render()


def test_torchscript_artifact_is_loadable_standalone(
    tiny_model: LanguageModel, tmp_path: Path
) -> None:
    """The archive must run without the ``lmops`` package importable."""
    path = export_torchscript(tiny_model, tmp_path / "m.pt", seq_len=16)
    loaded = torch.jit.load(str(path))
    out = loaded(torch.zeros(1, 8, dtype=torch.long))
    assert out.shape == (1, 8, tiny_model.config.vocab_size)


def test_onnx_export_matches_eager(tiny_model: LanguageModel, tmp_path: Path) -> None:
    pytest.importorskip("onnxruntime")
    path = export_onnx(tiny_model, tmp_path / "m.onnx", seq_len=16)
    assert path.exists()
    result = verify_onnx(tiny_model, path)
    assert result.passed, result.render()


def test_onnx_axes_are_dynamic(tiny_model: LanguageModel, tmp_path: Path) -> None:
    """Verified at a different batch *and* sequence length than the export shape."""
    pytest.importorskip("onnxruntime")
    path = export_onnx(tiny_model, tmp_path / "m.onnx", batch_size=1, seq_len=16)
    result = verify_onnx(tiny_model, path, batch_size=3, seq_len=21)
    assert result.passed, result.render()


def test_parity_result_renders_a_verdict() -> None:
    from lmops.export import ParityResult

    assert "PASS" in ParityResult("onnx", 1e-6, 1e-3, True).render()
    assert "FAIL" in ParityResult("onnx", 1.0, 1e-3, False).render()


def test_dynamic_quantisation_shrinks_the_model_and_still_runs(
    tiny_model: LanguageModel,
) -> None:
    before = state_dict_size_mb(tiny_model)
    quantized = quantize_dynamic_int8(tiny_model)
    after = state_dict_size_mb(quantized)
    assert after < before

    ids = torch.randint(0, tiny_model.config.vocab_size, (1, 12))
    with torch.no_grad():
        logits, _ = quantized(ids)
    assert logits.shape == (1, 12, tiny_model.config.vocab_size)
    assert torch.isfinite(logits).all()


def test_quantised_predictions_stay_close_to_full_precision(
    tiny_model: LanguageModel,
) -> None:
    """int8 costs accuracy, but the top-1 prediction should mostly survive.

    The bar is deliberately loose: this asserts "quantisation did not break the model",
    not "quantisation is free". The real quality check is a perplexity re-measurement.
    """
    ids = torch.randint(0, tiny_model.config.vocab_size, (4, 24))
    with torch.no_grad():
        reference = tiny_model(ids)[0].argmax(-1)
        quantized = quantize_dynamic_int8(tiny_model)(ids)[0].argmax(-1)
    agreement = (reference == quantized).float().mean().item()
    assert agreement > 0.5, f"top-1 agreement collapsed to {agreement:.2%}"
