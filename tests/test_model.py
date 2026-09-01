"""Model-level behaviour: shapes, initialisation, weight tying, learning and sampling."""

from __future__ import annotations

import math

import pytest
import torch

from lmops.config import ModelConfig
from lmops.model import LanguageModel


def test_forward_shapes(tiny_model: LanguageModel) -> None:
    ids = torch.randint(0, tiny_model.config.vocab_size, (3, 11))
    logits, loss = tiny_model(ids)
    assert logits.shape == (3, 11, tiny_model.config.vocab_size)
    assert loss is None


def test_loss_is_returned_when_targets_are_given(tiny_model: LanguageModel) -> None:
    ids = torch.randint(0, tiny_model.config.vocab_size, (2, 8))
    _, loss = tiny_model(ids, targets=ids)
    assert loss is not None
    assert loss.ndim == 0 and loss.item() > 0


def test_untrained_loss_is_near_uniform(tiny_model_config: ModelConfig) -> None:
    """At step 0 the model should be ~uniform over the vocabulary, i.e. loss ~ ln(V).

    A materially lower loss means the initialisation is leaking signal; a much higher
    one means the output scale is wrong. Targets are drawn *independently* of the
    inputs -- see the next test for why reusing the inputs would not measure this.
    """
    torch.manual_seed(0)
    model = LanguageModel(tiny_model_config)
    ids = torch.randint(0, tiny_model_config.vocab_size, (8, 32))
    targets = torch.randint(0, tiny_model_config.vocab_size, (8, 32))
    _, loss = model(ids, targets=targets)
    assert loss is not None
    assert loss.item() == pytest.approx(math.log(tiny_model_config.vocab_size), rel=0.05)


def test_tied_embeddings_bias_an_untrained_model_toward_the_input_token() -> None:
    """Weight tying makes ``logits = x @ E^T``, and at initialisation the residual
    stream ``x`` is dominated by the input token's own embedding. The untrained model
    therefore scores "predict the token you were just given" far better than chance,
    while an untied model does not.

    This is expected, not a defect -- but it is exactly why an initialisation check
    must use targets that are independent of the inputs.
    """
    ids = torch.randint(0, 97, (8, 32))
    losses = {}
    for tie in (True, False):
        torch.manual_seed(0)
        model = LanguageModel(
            ModelConfig(
                vocab_size=97,
                n_layer=2,
                n_head=4,
                n_kv_head=2,
                d_model=64,
                max_seq_len=64,
                dropout=0.0,
                tie_embeddings=tie,
            )
        )
        _, loss = model(ids, targets=ids)
        assert loss is not None
        losses[tie] = loss.item()

    assert losses[True] < losses[False] - 0.5
    assert losses[False] == pytest.approx(math.log(97), rel=0.05)


def test_last_only_matches_the_final_position(tiny_model: LanguageModel) -> None:
    ids = torch.randint(0, tiny_model.config.vocab_size, (2, 9))
    with torch.no_grad():
        full, _ = tiny_model(ids)
        last, _ = tiny_model(ids, last_only=True)
    assert last.shape == (2, 1, tiny_model.config.vocab_size)
    torch.testing.assert_close(last[:, 0], full[:, -1])


def test_weight_tying_shares_one_tensor() -> None:
    config = ModelConfig(vocab_size=64, n_layer=1, n_head=2, d_model=32, tie_embeddings=True)
    model = LanguageModel(config)
    assert model.lm_head.weight is model.token_emb.weight

    untied = LanguageModel(
        ModelConfig(vocab_size=64, n_layer=1, n_head=2, d_model=32, tie_embeddings=False)
    )
    assert untied.lm_head.weight is not untied.token_emb.weight
    assert untied.num_parameters() > model.num_parameters()


def test_non_embedding_parameter_count_excludes_the_table(tiny_model: LanguageModel) -> None:
    total = tiny_model.num_parameters()
    body = tiny_model.num_parameters(include_embeddings=False)
    assert body == total - tiny_model.token_emb.weight.numel()


def test_rejects_non_2d_input(tiny_model: LanguageModel) -> None:
    with pytest.raises(ValueError, match=r"\(batch, seq\)"):
        tiny_model(torch.zeros(4, dtype=torch.long))


def test_d_ff_is_derived_and_rounded() -> None:
    config = ModelConfig(d_model=384, ffn_multiple_of=64)
    assert config.d_ff is not None
    assert config.d_ff % 64 == 0
    assert config.d_ff >= int(8 * 384 / 3)


def test_invalid_head_configurations_are_rejected() -> None:
    with pytest.raises(ValueError, match="not divisible by n_head"):
        ModelConfig(d_model=100, n_head=8)
    with pytest.raises(ValueError, match="grouped-query attention"):
        ModelConfig(d_model=64, n_head=8, n_kv_head=3)


# -- learning -----------------------------------------------------------------


def test_model_can_overfit_a_single_batch(tiny_model_config: ModelConfig) -> None:
    """The canonical sanity check: if a model cannot memorise one batch, the loop is
    broken. Loss must fall close to zero, not merely decrease."""
    torch.manual_seed(0)
    model = LanguageModel(tiny_model_config)
    model.train()
    ids = torch.randint(0, tiny_model_config.vocab_size, (2, 16))
    targets = torch.randint(0, tiny_model_config.vocab_size, (2, 16))
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)

    first = None
    for _ in range(220):
        optimizer.zero_grad(set_to_none=True)
        _, loss = model(ids, targets=targets)
        assert loss is not None
        if first is None:
            first = loss.item()
        loss.backward()
        optimizer.step()

    assert first is not None
    final = loss.item()
    assert final < 0.15, f"failed to memorise one batch: {first:.3f} -> {final:.3f}"


def test_every_parameter_receives_a_gradient(tiny_model_config: ModelConfig) -> None:
    """An unused parameter usually means a block was silently skipped."""
    model = LanguageModel(tiny_model_config)
    model.train()
    ids = torch.randint(0, tiny_model_config.vocab_size, (2, 12))
    _, loss = model(ids, targets=ids)
    assert loss is not None
    loss.backward()

    missing = [n for n, p in model.named_parameters() if p.requires_grad and p.grad is None]
    assert not missing, f"no gradient reached: {missing}"


def test_forward_is_deterministic_in_eval_mode(tiny_model: LanguageModel) -> None:
    ids = torch.randint(0, tiny_model.config.vocab_size, (2, 10))
    with torch.no_grad():
        a, _ = tiny_model(ids)
        b, _ = tiny_model(ids)
    torch.testing.assert_close(a, b)


# -- generation ---------------------------------------------------------------


def test_generate_extends_the_prompt(tiny_model: LanguageModel) -> None:
    prompt = torch.randint(0, tiny_model.config.vocab_size, (2, 5))
    out = tiny_model.generate(prompt, max_new_tokens=7, temperature=1.0, top_k=5)
    assert out.shape == (2, 12)
    torch.testing.assert_close(out[:, :5], prompt)
    assert int(out.max()) < tiny_model.config.vocab_size


def test_greedy_decoding_is_reproducible(tiny_model: LanguageModel) -> None:
    prompt = torch.randint(0, tiny_model.config.vocab_size, (1, 4))
    a = tiny_model.generate(prompt, max_new_tokens=8, temperature=0.0)
    b = tiny_model.generate(prompt, max_new_tokens=8, temperature=0.0)
    torch.testing.assert_close(a, b)


def test_seeded_sampling_is_reproducible(tiny_model: LanguageModel) -> None:
    prompt = torch.randint(0, tiny_model.config.vocab_size, (1, 4))
    a = tiny_model.generate(
        prompt, max_new_tokens=8, temperature=1.0, generator=torch.Generator().manual_seed(3)
    )
    b = tiny_model.generate(
        prompt, max_new_tokens=8, temperature=1.0, generator=torch.Generator().manual_seed(3)
    )
    torch.testing.assert_close(a, b)


def test_top_k_restricts_the_support(tiny_model: LanguageModel) -> None:
    """With k=1, sampling must collapse onto greedy decoding at any temperature."""
    prompt = torch.randint(0, tiny_model.config.vocab_size, (1, 4))
    greedy = tiny_model.generate(prompt, max_new_tokens=6, temperature=0.0)
    top1 = tiny_model.generate(prompt, max_new_tokens=6, temperature=1.5, top_k=1)
    torch.testing.assert_close(greedy, top1)


def test_stream_yields_one_token_at_a_time(tiny_model: LanguageModel) -> None:
    prompt = torch.randint(0, tiny_model.config.vocab_size, (1, 3))
    tokens = list(tiny_model.stream(prompt, max_new_tokens=5, temperature=0.0))
    assert len(tokens) == 5
    assert all(t.shape == (1, 1) for t in tokens)
    streamed = torch.cat([prompt, *tokens], dim=1)
    torch.testing.assert_close(streamed, tiny_model.generate(prompt, 5, temperature=0.0))


def test_generation_respects_the_context_limit(tiny_model: LanguageModel) -> None:
    limit = tiny_model.config.max_seq_len
    prompt = torch.randint(0, tiny_model.config.vocab_size, (1, limit - 2))
    with pytest.raises(ValueError, match="exceeds max_seq_len"):
        tiny_model.generate(prompt, max_new_tokens=10)


class _ConstantHead(torch.nn.Module):
    """An lm_head that always votes for one token, whatever the hidden state.

    Replacing the whole module rather than editing its weights matters: with
    ``tie_embeddings=True`` the head *is* the embedding table, so zeroing its weights
    would wipe out the input embeddings instead of pinning the output.
    """

    def __init__(self, vocab_size: int, token: int) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.token = token

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = torch.full((*x.shape[:-1], self.vocab_size), -10.0, device=x.device)
        logits[..., self.token] = 10.0
        return logits


def test_eos_stops_generation_early(tiny_model: LanguageModel) -> None:
    """Force EOS to be the only reachable token and check the loop exits immediately."""
    tiny_model.lm_head = _ConstantHead(tiny_model.config.vocab_size, token=7)
    prompt = torch.randint(0, tiny_model.config.vocab_size, (1, 3))
    tokens = list(tiny_model.stream(prompt, max_new_tokens=20, temperature=0.0, eos_token_id=7))
    assert len(tokens) == 1
    assert int(tokens[0].item()) == 7


def test_generation_runs_to_the_budget_without_eos(tiny_model: LanguageModel) -> None:
    """Same forced token, but no ``eos_token_id``: the loop must not stop early."""
    tiny_model.lm_head = _ConstantHead(tiny_model.config.vocab_size, token=7)
    prompt = torch.randint(0, tiny_model.config.vocab_size, (1, 3))
    tokens = list(tiny_model.stream(prompt, max_new_tokens=6, temperature=0.0))
    assert len(tokens) == 6


def test_stream_restores_training_mode(tiny_model: LanguageModel) -> None:
    tiny_model.train()
    prompt = torch.randint(0, tiny_model.config.vocab_size, (1, 3))
    list(tiny_model.stream(prompt, max_new_tokens=2, temperature=0.0))
    assert tiny_model.training is True
