"""Temporal positional encodings for TCPFN-style models.

Ported from `tcpfn.models.temporal_encoding`. Two strategies:

- ``short_range_time_encoding`` (v2.1 / v3 default): learnable per-step
  embedding for delta_t clamped to ``[-max_horizon, +max_horizon]``, a phase
  embedding (pre/post-treatment), and an optional elapsed-time projection.
  Best when all lags fit inside ``max_horizon`` (v2.1 trains at
  ``max_horizon=100``).

- ``long_range_time_encoding`` (v2.2): adds a log-spaced sinusoidal
  encoding on top of the learnable one, so the model can extrapolate to
  delta_t values **outside** ``[-max_horizon, +max_horizon]``. This is what
  unlocks v2.2's 12-hour industrial-lag support.

These blocks have a non-standard ``__call__(delta_t, phase, elapsed_t=None)``
signature — they don't fit priorstudio's default ``__call__(x)`` chain. A
TCPFN-style model uses a custom ``step_fn`` that calls the time encoding
separately and adds the result to the tabular embedding before the
transformer. Registering them here makes the model.yaml resolvable and
makes them visible in the block library; the runtime integration is part
of the TCPFN model wrapper that lives downstream.
"""

from __future__ import annotations

import math
from typing import Any

from ..registry import register_block


def _build_short_range():
    import torch
    import torch.nn as nn

    class ShortRangeEncoding(nn.Module):
        def __init__(self, d_model: int, max_horizon: int = 100):
            super().__init__()
            self.d_model = d_model
            self.max_horizon = max_horizon
            self.relative_embed = nn.Embedding(2 * max_horizon + 1, d_model)
            self.phase_embed = nn.Embedding(2, d_model)
            self.elapsed_proj = nn.Linear(1, d_model, bias=False)
            nn.init.normal_(self.relative_embed.weight, std=0.02)
            nn.init.normal_(self.phase_embed.weight, std=0.02)
            nn.init.normal_(self.elapsed_proj.weight, std=0.02)

        def forward(self, delta_t, phase, elapsed_t=None):
            idx = (delta_t + self.max_horizon).clamp(0, 2 * self.max_horizon).long()
            enc = self.relative_embed(idx) + self.phase_embed(phase.long())
            if elapsed_t is not None:
                enc = enc + self.elapsed_proj(elapsed_t.unsqueeze(-1).float())
            return enc

    return ShortRangeEncoding


def _build_long_range():
    import torch
    import torch.nn as nn

    class LongRangeEncoding(nn.Module):
        def __init__(self, d_model: int, max_horizon: int = 100):
            super().__init__()
            self.d_model = d_model
            self.max_horizon = max_horizon
            self.relative_embed = nn.Embedding(2 * max_horizon + 1, d_model)
            self.continuous_proj = nn.Linear(d_model, d_model, bias=False)
            self.phase_embed = nn.Embedding(2, d_model)
            self.elapsed_proj = nn.Linear(1, d_model, bias=False)

            freqs = torch.exp(
                torch.linspace(math.log(1.0), math.log(1.0 / 10000), d_model // 2)
            )
            self.register_buffer("freqs", freqs)

            nn.init.normal_(self.relative_embed.weight, std=0.02)
            nn.init.normal_(self.continuous_proj.weight, std=0.02)
            nn.init.normal_(self.phase_embed.weight, std=0.02)
            nn.init.normal_(self.elapsed_proj.weight, std=0.02)

        def _continuous_encoding(self, delta_t):
            t = delta_t.unsqueeze(-1).float()
            angles = t * self.freqs
            enc = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
            return self.continuous_proj(enc)

        def forward(self, delta_t, phase, elapsed_t=None):
            enc = self._continuous_encoding(delta_t)
            in_range = (delta_t >= -self.max_horizon) & (delta_t <= self.max_horizon)
            if in_range.any():
                idx = (delta_t + self.max_horizon).clamp(0, 2 * self.max_horizon).long()
                learnable = self.relative_embed(idx)
                enc = enc + learnable * in_range.unsqueeze(-1).float()
            enc = enc + self.phase_embed(phase.long())
            if elapsed_t is not None:
                enc = enc + self.elapsed_proj(elapsed_t.unsqueeze(-1).float())
            return enc

    return LongRangeEncoding


@register_block("short_range_time_encoding")
class ShortRangeTimeEncodingBlock:
    """Learnable per-step delta_t embedding + pre/post phase + elapsed projection.

    Inputs are batch-side tensors (``delta_t``, ``phase``, optional
    ``elapsed_t``) — not a single ``x``. Used inside a TCPFN model wrapper,
    not as a link in the default block chain.
    """

    def __init__(self, d_model: int = 512, max_horizon: int = 100):
        try:
            ShortRangeEncoding = _build_short_range()
        except ImportError as e:
            raise ImportError("short_range_time_encoding requires torch.") from e

        self.d_model = d_model
        self.max_horizon = max_horizon
        self.module = ShortRangeEncoding(d_model=d_model, max_horizon=max_horizon)

    def __call__(self, delta_t, phase, elapsed_t=None):
        return self.module(delta_t, phase, elapsed_t)


@register_block("long_range_time_encoding")
class LongRangeTimeEncodingBlock:
    """Log-spaced sinusoidal + learnable delta_t encoding. Handles arbitrary lags.

    Inside ``[-max_horizon, max_horizon]`` the learnable embedding adds
    detail on top of the continuous encoding. Outside that range only the
    continuous encoding fires — that's the v2.2 "12+ hour industrial lags"
    capability.
    """

    def __init__(self, d_model: int = 512, max_horizon: int = 100):
        try:
            LongRangeEncoding = _build_long_range()
        except ImportError as e:
            raise ImportError("long_range_time_encoding requires torch.") from e

        self.d_model = d_model
        self.max_horizon = max_horizon
        self.module = LongRangeEncoding(d_model=d_model, max_horizon=max_horizon)

    def __call__(self, delta_t, phase, elapsed_t=None):
        return self.module(delta_t, phase, elapsed_t)
