"""Cross-attention transformer blocks: RMSNorm + cross-attention encoder.

A cross-attention encoder differs from a standard transformer encoder in two
important ways:

1. **Cross-attention only between query and context** — query tokens attend
   to context tokens (the first ``context_length`` of the sequence) but never
   to each other. This is what makes the block an in-context predictor:
   adding more query points doesn't change the per-query prediction. Used by
   the TCPFN v2/v3 reference architecture; also useful for any PFN-style
   model where queries should be independent given the context.
2. **RMSNorm everywhere** — replaces LayerNorm. Faster, no learnable bias.
   Also applied to Q and K per-head, which stabilises attention scaling.

The block API stays compatible with priorstudio's default chain: the encoder
takes ``(B, N, d_model)`` and returns ``(B, N, d_model)``. ``context_length``
is set at construction time (read from model.yaml), defaulting to ``None``
which means "treat the whole sequence as context" (i.e. standard
self-attention).
"""

from __future__ import annotations

import math
from typing import Any

from ..registry import register_block


def _build_layer_class():
    """Build the RMSNorm + TransformerEncoderLayer classes lazily.

    Done inside a function so that importing this module doesn't require torch.
    Returns ``(RMSNorm, CrossAttentionEncoderLayer)`` — both fresh classes per
    call, but that's fine since we only call this once during first use.
    """
    import torch
    import torch.nn as nn
    import torch.nn.functional as F  # noqa: N812 — F is the canonical alias for torch.nn.functional

    class RMSNorm(nn.Module):
        def __init__(self, dim: int, eps: float = 1e-6):
            super().__init__()
            self.eps = eps
            self.weight = nn.Parameter(torch.ones(dim))

        def _norm(self, x):
            return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

        def forward(self, x):
            output = self._norm(x.float()).type_as(x)
            return output * self.weight

    class CrossAttentionEncoderLayer(nn.Module):
        """One cross-attention transformer layer.

        Input: ``(seq_len, batch, embed_dim)`` (seq-first; transposed for the
        attention computation, then transposed back).
        Output: same shape.
        """

        def __init__(self, embed_dim: int, num_heads: int, ff_dim: int):
            super().__init__()
            self.embed_dim = embed_dim
            self.head_dim = embed_dim // num_heads
            self.num_heads = num_heads
            self.ff_dim = ff_dim
            self.kv_proj = nn.Linear(embed_dim, 2 * embed_dim, bias=False)
            self.q_proj = nn.Linear(embed_dim, 3 * embed_dim, bias=False)
            self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)
            self.attn_norm = RMSNorm(embed_dim)
            self.ff_norm = RMSNorm(embed_dim)
            self.q_norm = RMSNorm(self.head_dim)
            self.k_norm = RMSNorm(self.head_dim)

        def forward(self, x, context_length, kv_cache=None, graph_bias=None):
            x = x.transpose(0, 1)
            B, L, _ = x.size()
            h = self.attn_norm(x)
            q, ff_h, ff_gate = self.q_proj(h).chunk(3, dim=-1)

            if kv_cache is not None:
                k, v = kv_cache
            else:
                k, v = self.kv_proj(h[:, :context_length]).chunk(2, dim=-1)
                k = k.view(B, context_length, self.num_heads, self.head_dim).transpose(1, 2)
                v = v.view(B, context_length, self.num_heads, self.head_dim).transpose(1, 2)
                k = self.k_norm(k).to(v.dtype)

            q = q.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
            q = self.q_norm(q)
            q = q * math.log2(max(context_length, 2)) / 10
            q = self.q_norm(q).to(v.dtype)

            attn_mask = None
            if graph_bias is not None:
                attn_mask = graph_bias.unsqueeze(1) if graph_bias.dim() == 3 else graph_bias

            attn = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask).transpose(1, 2)
            attn = attn.reshape(B, L, self.num_heads * self.head_dim)
            ff_x = ff_h * F.silu(ff_gate)
            residual = self.ff_norm(self.out_proj(attn + ff_x))
            x_out = x + residual
            return x_out.transpose(0, 1)

    return RMSNorm, CrossAttentionEncoderLayer


@register_block("rms_norm")
class RMSNormBlock:
    """Root-mean-square layer norm. Cheaper than LayerNorm, no learnable bias.

    Input/output: ``(..., dim)``.
    """

    def __init__(self, dim: int = 256, eps: float = 1e-6):
        try:
            RMSNorm, _ = _build_layer_class()
        except ImportError as e:
            raise ImportError("rms_norm requires torch.") from e

        self.dim = dim
        self.module = RMSNorm(dim=dim, eps=eps)

    def __call__(self, x: Any) -> Any:
        return self.module(x)


@register_block("cross_attention_layer")
class CrossAttentionLayerBlock:
    """One cross-attention transformer layer (RMSNorm + SiLU-gated FFN).

    Takes ``(B, N, embed_dim)`` in priorstudio convention and transposes
    internally to seq-first for the attention computation, then transposes
    back. ``context_length`` is set at construction time; the first
    ``context_length`` tokens are the context, every token's query attends
    to those context tokens via cross-attention.

    Pass ``context_length=None`` (default) for standard self-attention over
    the whole sequence.
    """

    def __init__(
        self,
        embed_dim: int = 512,
        num_heads: int = 8,
        ff_dim: int | None = None,
        context_length: int | None = None,
    ):
        try:
            _, CrossAttentionEncoderLayer = _build_layer_class()
        except ImportError as e:
            raise ImportError("cross_attention_layer requires torch.") from e

        self.embed_dim = embed_dim
        self.context_length = context_length
        self.module = CrossAttentionEncoderLayer(
            embed_dim=embed_dim,
            num_heads=num_heads,
            ff_dim=ff_dim if ff_dim is not None else embed_dim * 4,
        )

    def __call__(self, x: Any) -> Any:
        # priorstudio convention is (B, N, d); attention path expects (N, B, d).
        x_seq_first = x.transpose(0, 1)
        ctx_len = self.context_length if self.context_length is not None else x_seq_first.shape[0]
        out = self.module(x_seq_first, ctx_len)
        return out.transpose(0, 1)


@register_block("cross_attention_encoder")
class CrossAttentionEncoderBlock:
    """Stack of cross-attention transformer layers.

    The "thinking core" of in-context predictors like TCPFN v2/v3. n_layers
    layers, each using cross-attention from query → context with RMSNorm +
    SiLU-gated FFN.
    """

    def __init__(
        self,
        d_model: int = 512,
        n_heads: int = 8,
        n_layers: int = 12,
        ff_mult: int = 4,
        context_length: int | None = None,
        dropout: float = 0.0,  # accepted for parity; this layer doesn't use dropout
    ):
        try:
            import torch.nn as nn

            _, CrossAttentionEncoderLayer = _build_layer_class()
        except ImportError as e:
            raise ImportError("cross_attention_encoder requires torch.") from e

        self.d_model = d_model
        self.context_length = context_length
        self._dropout = dropout
        layers = [
            CrossAttentionEncoderLayer(
                embed_dim=d_model,
                num_heads=n_heads,
                ff_dim=d_model * ff_mult,
            )
            for _ in range(n_layers)
        ]
        self.module = nn.ModuleList(layers)

    def __call__(self, x: Any) -> Any:
        x_seq_first = x.transpose(0, 1)
        ctx_len = self.context_length if self.context_length is not None else x_seq_first.shape[0]
        for layer in self.module:
            x_seq_first = layer(x_seq_first, ctx_len)
        return x_seq_first.transpose(0, 1)
