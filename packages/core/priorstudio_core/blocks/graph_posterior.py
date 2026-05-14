"""Joint Graph Posterior blocks for in-context causal-discovery models.

Ported from `tcpfn.models.graph_posterior`. Three blocks compose into the
discovery pipeline:

1. ``temporal_variable_encoder`` — per-variable temporal attention. Takes
   ``(B, T, V)`` multivariate time series, returns ``(B, V, d_model)`` —
   one representation per variable.
2. ``cross_variable_attention`` — attention across variables, takes
   ``(B, V, d_model)`` and returns ``(B, V, d_model)`` enriched with
   inter-variable info.
3. ``graph_posterior_decoder`` — pairwise edge prediction from variable
   representations. Returns a dict with ``adjacency``, ``edge_type_logits``,
   ``edge_identifiability``.

Each is independently registered so YAML specs can mix and match.
"""

from __future__ import annotations

from typing import Any

from ..registry import register_block


def _build_temporal_variable_encoder():
    import torch
    import torch.nn as nn

    class TemporalVariableEncoder(nn.Module):
        def __init__(
            self,
            max_time_steps: int = 200,
            d_model: int = 256,
            n_heads: int = 4,
            n_layers: int = 3,
        ):
            super().__init__()
            self.d_model = d_model
            self.input_proj = nn.Linear(1, d_model)
            self.pos_enc = nn.Parameter(torch.randn(max_time_steps, d_model) * 0.02)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_heads,
                dim_feedforward=d_model * 2,
                dropout=0.1,
                batch_first=True,
                norm_first=True,
            )
            self.temporal_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
            self.aggregate = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.GELU(),
                nn.LayerNorm(d_model),
            )

        def forward(self, x):
            B, T, V = x.shape
            x_flat = x.transpose(1, 2).reshape(B * V, T, 1)
            h = self.input_proj(x_flat)
            h = h + self.pos_enc[:T].unsqueeze(0)
            h = self.temporal_encoder(h)
            h = h.mean(dim=1)
            h = self.aggregate(h)
            return h.view(B, V, self.d_model)

    return TemporalVariableEncoder


def _build_cross_variable_attention():
    import torch.nn as nn

    class CrossVariableAttention(nn.Module):
        def __init__(self, d_model: int = 256, n_heads: int = 4, n_layers: int = 3):
            super().__init__()
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_heads,
                dim_feedforward=d_model * 2,
                dropout=0.1,
                batch_first=True,
                norm_first=True,
            )
            self.cross_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        def forward(self, var_reps):
            return self.cross_encoder(var_reps)

    return CrossVariableAttention


def _build_graph_posterior_decoder():
    import torch
    import torch.nn as nn

    class GraphPosteriorDecoder(nn.Module):
        def __init__(self, d_model: int = 256, max_lag: int = 5):
            super().__init__()
            self.max_lag = max_lag
            self.edge_mlp = nn.Sequential(
                nn.Linear(2 * d_model, d_model),
                nn.GELU(),
                nn.Linear(d_model, d_model // 2),
                nn.GELU(),
                nn.Linear(d_model // 2, max_lag + 1),
            )
            self.edge_type_mlp = nn.Sequential(
                nn.Linear(2 * d_model, d_model // 2),
                nn.GELU(),
                nn.Linear(d_model // 2, 4),
            )
            self.edge_ident_mlp = nn.Sequential(
                nn.Linear(2 * d_model, d_model // 2),
                nn.GELU(),
                nn.Linear(d_model // 2, 1),
            )

        def forward(self, var_reps):
            B, V, d = var_reps.shape
            rep_i = var_reps.unsqueeze(2).expand(B, V, V, d)
            rep_j = var_reps.unsqueeze(1).expand(B, V, V, d)
            pairs = torch.cat([rep_i, rep_j], dim=-1)
            adj = torch.sigmoid(self.edge_mlp(pairs))
            edge_types = self.edge_type_mlp(pairs)
            ident = torch.sigmoid(self.edge_ident_mlp(pairs).squeeze(-1))
            return {
                "adjacency": adj,
                "edge_type_logits": edge_types,
                "edge_identifiability": ident,
            }

    return GraphPosteriorDecoder


@register_block("temporal_variable_encoder")
class TemporalVariableEncoderBlock:
    """Encode each variable's temporal trajectory into a per-variable representation.

    Input: ``(B, T, V)`` multivariate time series.
    Output: ``(B, V, d_model)`` — one rich representation per variable.
    """

    def __init__(
        self,
        max_time_steps: int = 200,
        d_model: int = 256,
        n_heads: int = 4,
        n_layers: int = 3,
    ):
        try:
            TemporalVariableEncoder = _build_temporal_variable_encoder()
        except ImportError as e:
            raise ImportError("temporal_variable_encoder requires torch.") from e

        self.d_model = d_model
        self.module = TemporalVariableEncoder(
            max_time_steps=max_time_steps,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
        )

    def __call__(self, x: Any) -> Any:
        return self.module(x)


@register_block("cross_variable_attention")
class CrossVariableAttentionBlock:
    """Attention across variables to capture inter-variable relationships.

    Input/output: ``(B, V, d_model)``.
    """

    def __init__(self, d_model: int = 256, n_heads: int = 4, n_layers: int = 3):
        try:
            CrossVariableAttention = _build_cross_variable_attention()
        except ImportError as e:
            raise ImportError("cross_variable_attention requires torch.") from e

        self.d_model = d_model
        self.module = CrossVariableAttention(d_model=d_model, n_heads=n_heads, n_layers=n_layers)

    def __call__(self, x: Any) -> Any:
        return self.module(x)


@register_block("graph_posterior_decoder")
class GraphPosteriorDecoderBlock:
    """Decode pairwise edge predictions from variable representations.

    Input: ``(B, V, d_model)``.
    Output: dict with ``adjacency`` ``(B, V, V, max_lag+1)``,
    ``edge_type_logits`` ``(B, V, V, 4)``, ``edge_identifiability``
    ``(B, V, V)``.

    Returns a dict — downstream code (eval / step_fn) consumes the fields
    individually. The default training step doesn't know about dict
    outputs, so this block requires a custom step_fn.
    """

    def __init__(self, d_model: int = 256, max_lag: int = 5):
        try:
            GraphPosteriorDecoder = _build_graph_posterior_decoder()
        except ImportError as e:
            raise ImportError("graph_posterior_decoder requires torch.") from e

        self.d_model = d_model
        self.max_lag = max_lag
        self.module = GraphPosteriorDecoder(d_model=d_model, max_lag=max_lag)

    def __call__(self, x: Any) -> Any:
        return self.module(x)
