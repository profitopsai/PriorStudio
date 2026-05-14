"""Generic multi-layer perceptron block.

Used as the head-projection in TCPFN models (``head_mlp`` in v2.1/v2.2/v3
YAML specs) and as a general-purpose nonlinear projection. Sits between a
pooled representation and a task head.
"""

from __future__ import annotations

from typing import Any

from ..registry import register_block

_ACTIVATIONS = ("gelu", "relu", "silu", "tanh")


@register_block("mlp")
class MLPBlock:
    """Feed-forward stack: ``n_layers`` of (Linear → activation → dropout).

    Args:
        d_in: input dim (None → ``LazyLinear`` so it adapts on first call).
        d_hidden: hidden dim. Defaults to 4× ``d_in`` if provided, else 512.
        d_out: output dim. ``None`` → same as ``d_hidden`` (chainable).
        n_layers: number of (Linear → activation) pairs *before* the final
            projection. ``n_layers=2`` ≡ "input → hidden → hidden → out".
        activation: ``gelu`` | ``relu`` | ``silu`` | ``tanh``.
        dropout: applied after each activation. ``0.0`` disables.
    """

    def __init__(
        self,
        d_in: int | None = None,
        d_hidden: int | None = None,
        d_out: int | None = None,
        n_layers: int = 2,
        activation: str = "gelu",
        dropout: float = 0.0,
    ):
        try:
            import torch.nn as nn
        except ImportError as e:
            raise ImportError("mlp requires torch.") from e

        if activation not in _ACTIVATIONS:
            raise ValueError(f"Unknown mlp activation {activation!r}; known: {_ACTIVATIONS}")
        act_cls = {"gelu": nn.GELU, "relu": nn.ReLU, "silu": nn.SiLU, "tanh": nn.Tanh}[activation]

        hidden = d_hidden if d_hidden is not None else (4 * d_in if d_in else 512)
        out_dim = d_out if d_out is not None else hidden

        layers: list[Any] = []
        first_in = d_in
        if first_in is None:
            layers.append(nn.LazyLinear(hidden))
        else:
            layers.append(nn.Linear(first_in, hidden))
        layers.append(act_cls())
        if dropout > 0.0:
            layers.append(nn.Dropout(dropout))
        for _ in range(max(0, n_layers - 1)):
            layers.append(nn.Linear(hidden, hidden))
            layers.append(act_cls())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(hidden, out_dim))

        self.d_in = d_in
        self.d_hidden = hidden
        self.d_out = out_dim
        self.module = nn.Sequential(*layers)

    def __call__(self, x: Any) -> Any:
        return self.module(x)
