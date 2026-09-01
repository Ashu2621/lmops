"""The decoder-only transformer itself, plus cached autoregressive sampling."""

from __future__ import annotations

import math
from collections.abc import Iterator

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ..config import ModelConfig
from .attention import CausalSelfAttention, KVCache, build_cache_mask
from .layers import RMSNorm, RotaryEmbedding, SwiGLU


class TransformerBlock(nn.Module):
    """Pre-norm block: ``x + attn(norm(x))`` then ``x + mlp(norm(x))``."""

    def __init__(self, config: ModelConfig, layer_idx: int) -> None:
        super().__init__()
        assert config.d_ff is not None
        self.attn_norm = RMSNorm(config.d_model, config.norm_eps)
        self.attn = CausalSelfAttention(config, layer_idx)
        self.mlp_norm = RMSNorm(config.d_model, config.norm_eps)
        self.mlp = SwiGLU(config.d_model, config.d_ff, config.dropout)

    def forward(
        self,
        x: Tensor,
        cos: Tensor,
        sin: Tensor,
        cache: KVCache | None = None,
        attn_mask: Tensor | None = None,
    ) -> Tensor:
        x = x + self.attn(self.attn_norm(x), cos, sin, cache=cache, attn_mask=attn_mask)
        return x + self.mlp(self.mlp_norm(x))


class LanguageModel(nn.Module):
    """A small decoder-only LM: RMSNorm + RoPE + grouped-query attention + SwiGLU."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config

        self.token_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.emb_dropout = nn.Dropout(config.dropout)
        self.rope = RotaryEmbedding(config.head_dim, config.max_seq_len, config.rope_theta)
        self.blocks = nn.ModuleList([TransformerBlock(config, i) for i in range(config.n_layer)])
        self.final_norm = RMSNorm(config.d_model, config.norm_eps)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        if config.tie_embeddings:
            # Tied weights: the head *is* the embedding table, which removes
            # vocab_size * d_model parameters and usually helps small models.
            self.lm_head.weight = self.token_emb.weight

        self.apply(self._init_weights)
        # GPT-2 style scaled init on residual output projections keeps the variance
        # of the residual stream from growing linearly with depth.
        residual_std = config.init_std / math.sqrt(2 * config.n_layer)
        for name, param in self.named_parameters():
            if name.endswith(("o_proj.weight", "down_proj.weight")):
                nn.init.normal_(param, mean=0.0, std=residual_std)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.init_std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.init_std)

    # -- introspection --------------------------------------------------------

    def num_parameters(self, trainable_only: bool = True, include_embeddings: bool = True) -> int:
        params = (p for p in self.parameters() if p.requires_grad or not trainable_only)
        total = sum(p.numel() for p in params)
        if not include_embeddings:
            total -= self.token_emb.weight.numel()
            if not self.config.tie_embeddings:
                total -= self.lm_head.weight.numel()
        return total

    # -- forward --------------------------------------------------------------

    def forward(
        self,
        idx: Tensor,
        targets: Tensor | None = None,
        cache: KVCache | None = None,
        last_only: bool = False,
    ) -> tuple[Tensor, Tensor | None]:
        """Run the stack.

        Args:
            idx: ``(batch, seq)`` int64 token ids.
            targets: optional ``(batch, seq)`` next-token labels for the loss.
            cache: optional :class:`KVCache` for incremental decoding. When given,
                ``idx`` holds only the *new* tokens and positions resume at
                ``cache.pos``.
            last_only: project just the final position to vocab space. Saves a
                ``seq x vocab`` matmul during generation.

        Returns:
            ``(logits, loss)``; ``loss`` is ``None`` when ``targets`` is ``None``.
        """
        if idx.dim() != 2:
            raise ValueError(f"expected (batch, seq) token ids, got shape {tuple(idx.shape)}")
        _, seq_len = idx.shape
        offset = cache.pos if cache is not None else 0

        cos, sin = self.rope(seq_len, offset=offset, device=idx.device)
        attn_mask = build_cache_mask(seq_len, offset, idx.device) if cache is not None else None

        x = self.emb_dropout(self.token_emb(idx))
        for block in self.blocks:
            x = block(x, cos, sin, cache=cache, attn_mask=attn_mask)
        x = self.final_norm(x)

        if cache is not None:
            cache.advance(seq_len)

        if last_only and targets is None:
            x = x[:, -1:, :]

        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)).float(),
                targets.reshape(-1),
                ignore_index=-100,
            )
        return logits, loss

    # -- generation -----------------------------------------------------------

    @torch.inference_mode()
    def generate(
        self,
        idx: Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
        top_p: float | None = None,
        repetition_penalty: float = 1.0,
        eos_token_id: int | None = None,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        """Sample ``max_new_tokens`` continuations of ``idx`` and return the full sequence."""
        out = idx
        for token in self.stream(
            idx,
            max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            eos_token_id=eos_token_id,
            generator=generator,
        ):
            out = torch.cat([out, token], dim=1)
        return out

    @torch.inference_mode()
    def stream(
        self,
        idx: Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
        top_p: float | None = None,
        repetition_penalty: float = 1.0,
        eos_token_id: int | None = None,
        generator: torch.Generator | None = None,
    ) -> Iterator[Tensor]:
        """Yield one ``(batch, 1)`` token tensor per decoding step.

        The prompt is consumed in a single prefill pass; every subsequent step feeds
        exactly one token through the cache, so cost per token is independent of the
        prompt length.
        """
        was_training = self.training
        self.eval()
        try:
            device = idx.device
            batch, prompt_len = idx.shape
            budget = prompt_len + max_new_tokens
            if budget > self.config.max_seq_len:
                raise ValueError(
                    f"prompt ({prompt_len}) + max_new_tokens ({max_new_tokens}) = {budget} "
                    f"exceeds max_seq_len={self.config.max_seq_len}"
                )

            param_dtype = next(self.parameters()).dtype
            cache = KVCache.allocate(
                self.config, batch, max_seq_len=budget, device=device, dtype=param_dtype
            )

            logits, _ = self(idx, cache=cache, last_only=True)
            generated = idx
            finished = torch.zeros(batch, dtype=torch.bool, device=device)

            for _ in range(max_new_tokens):
                next_token = _sample(
                    logits[:, -1, :],
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    repetition_penalty=repetition_penalty,
                    history=generated,
                    generator=generator,
                )
                if eos_token_id is not None:
                    # Once a row has emitted EOS, pin it there so batched decoding
                    # does not resume a finished sequence with garbage.
                    next_token = torch.where(
                        finished.unsqueeze(1),
                        torch.full_like(next_token, eos_token_id),
                        next_token,
                    )
                    finished = finished | (next_token.squeeze(1) == eos_token_id)

                generated = torch.cat([generated, next_token], dim=1)
                yield next_token

                if eos_token_id is not None and bool(finished.all()):
                    break
                logits, _ = self(next_token, cache=cache, last_only=True)
        finally:
            self.train(was_training)


def _sample(
    logits: Tensor,
    temperature: float,
    top_k: int | None,
    top_p: float | None,
    repetition_penalty: float,
    history: Tensor,
    generator: torch.Generator | None,
) -> Tensor:
    """Apply the sampling stack to ``(batch, vocab)`` logits and draw one token."""
    logits = logits.float()

    if repetition_penalty != 1.0:
        # Divide positive logits, multiply negative ones -- the CTRL formulation,
        # which keeps the penalty monotone regardless of logit sign.
        gathered = logits.gather(1, history)
        gathered = torch.where(
            gathered > 0, gathered / repetition_penalty, gathered * repetition_penalty
        )
        logits.scatter_(1, history, gathered)

    if temperature <= 0.0:
        return logits.argmax(dim=-1, keepdim=True)

    logits = logits / temperature

    if top_k is not None and top_k > 0:
        k = min(top_k, logits.size(-1))
        threshold = logits.topk(k, dim=-1).values[:, -1, None]
        logits = logits.masked_fill(logits < threshold, float("-inf"))

    if top_p is not None and 0.0 < top_p < 1.0:
        sorted_logits, sorted_idx = logits.sort(dim=-1, descending=True)
        cumulative = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
        # Keep the first token that crosses the threshold, drop everything after it.
        drop = cumulative - sorted_logits.softmax(dim=-1) >= top_p
        drop[:, 0] = False
        sorted_logits = sorted_logits.masked_fill(drop, float("-inf"))
        logits = torch.full_like(logits, float("-inf")).scatter_(1, sorted_idx, sorted_logits)

    probs = logits.softmax(dim=-1)
    return torch.multinomial(probs, num_samples=1, generator=generator)
