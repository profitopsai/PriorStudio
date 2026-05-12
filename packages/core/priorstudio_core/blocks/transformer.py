"""Transformer encoder block.

Implementation requires PyTorch. Importing this module without torch installed
will raise on first instantiation, not on import — so projects that use a
non-PyTorch framework can still load this package.
"""

from __future__ import annotations

from typing import Any

from ..registry import register_block


@register_block("transformer_encoder")
class TransformerEncoder:
    """Standard pre-norm transformer encoder."""

    def __init__(
        self,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 6,
        dropout: float = 0.1,
        ff_mult: int = 4,
    ):
        try:
            import torch.nn as nn
        except ImportError as e:
            raise ImportError(
                "transformer_encoder block requires torch. "
                "Install with: pip install priorstudio-core[torch]"
            ) from e

        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * ff_mult,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.module = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.d_model = d_model

    def __call__(self, x: Any) -> Any:
        return self.module(x)


@register_block("causal_attention_pool")
class CausalAttentionPool:
    """Pools a sequence to a single token via attention with a learned query."""

    def __init__(self, d_model: int = 256):
        try:
            import torch
            import torch.nn as nn
        except ImportError as e:
            raise ImportError("causal_attention_pool requires torch.") from e

        self.query = nn.Parameter(torch.zeros(1, 1, d_model))
        self.attn = nn.MultiheadAttention(d_model, num_heads=1, batch_first=True)

    def __call__(self, x: Any) -> Any:
        b = x.shape[0]
        q = self.query.expand(b, -1, -1)
        out, _ = self.attn(q, x, x)
        return out.squeeze(1)
