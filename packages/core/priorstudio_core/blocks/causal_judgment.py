"""Causal-judgment head for TCPFN v2/v3 models.

Ported from `tcpfn.models.causal_judgment_head.CausalJudgmentHead`. Unlike
a single-scalar CATE head, this emits a structured causal judgment per
(treatment, outcome) pair:

- ``total_effect`` / ``direct_effect`` (regression)
- ``null_probability`` — P(true effect = 0 | data); a confounding detector
- ``confounding_score`` — estimated unmeasured-confounding strength
- ``identifiability`` — P(effect is identifiable from observed data)
- ``mediation_fraction`` — fraction of total effect that flows through
  observed mediators
- ``uncertainty`` (lower / upper offsets)
- ``regime_logits`` — softmax over [direct, confounded, mediated, feedback]

Returns a dict — downstream code (eval / step_fn) consumes the fields
individually. The default training step doesn't handle dict outputs, so
this block needs a custom step_fn (the v2.1 model uses one).
"""

from __future__ import annotations

from typing import Any

from ..registry import register_block

REGIMES = ("direct", "confounded", "mediated", "feedback")


def _build_causal_judgment_head():
    import torch
    import torch.nn as nn

    class CausalJudgmentHead(nn.Module):
        def __init__(self, d_model: int = 512, hidden_dim: int = 256):
            super().__init__()
            self.shared = nn.Sequential(
                nn.Linear(d_model, hidden_dim),
                nn.GELU(),
                nn.LayerNorm(hidden_dim),
            )
            self.total_effect_head = nn.Linear(hidden_dim, 1)
            self.direct_effect_head = nn.Linear(hidden_dim, 1)
            self.null_head = nn.Linear(hidden_dim, 1)
            self.confounding_head = nn.Linear(hidden_dim, 1)
            self.identifiability_head = nn.Linear(hidden_dim, 1)
            self.mediation_head = nn.Linear(hidden_dim, 1)
            self.uncertainty_head = nn.Linear(hidden_dim, 2)
            self.regime_head = nn.Linear(hidden_dim, len(REGIMES))

        def forward(self, hidden):
            # hidden: (B, query_len, d_model) → pool over query tokens
            if hidden.dim() == 3:
                h = hidden.mean(dim=1)
            else:
                h = hidden
            shared = self.shared(h)
            total = self.total_effect_head(shared).squeeze(-1)
            direct = self.direct_effect_head(shared).squeeze(-1)
            null_prob = torch.sigmoid(self.null_head(shared).squeeze(-1))
            confounding = torch.sigmoid(self.confounding_head(shared).squeeze(-1))
            identifiability = torch.sigmoid(self.identifiability_head(shared).squeeze(-1))
            mediation = torch.sigmoid(self.mediation_head(shared).squeeze(-1))
            uncertainty = self.uncertainty_head(shared)
            regime_logits = self.regime_head(shared)
            return {
                "total_effect": total,
                "direct_effect": direct,
                "null_probability": null_prob,
                "confounding_score": confounding,
                "identifiability": identifiability,
                "mediation_fraction": mediation,
                "uncertainty": uncertainty,
                "regime_logits": regime_logits,
            }

    return CausalJudgmentHead


@register_block("causal_judgment_head")
class CausalJudgmentHeadBlock:
    """8-task structured causal judgment head.

    Input: ``(B, query_len, d_model)`` (transformer output for query tokens)
    OR ``(B, d_model)`` (already pooled).

    Output: dict with the 8 judgment fields. Requires a custom step_fn for
    training — the multi-task loss is what makes the v2/v3 numbers work.
    """

    def __init__(self, d_model: int = 512, hidden_dim: int = 256):
        try:
            CausalJudgmentHead = _build_causal_judgment_head()
        except ImportError as e:
            raise ImportError("causal_judgment_head requires torch.") from e

        self.d_model = d_model
        self.hidden_dim = hidden_dim
        self.regimes = list(REGIMES)
        self.module = CausalJudgmentHead(d_model=d_model, hidden_dim=hidden_dim)

    def __call__(self, x: Any) -> Any:
        return self.module(x)
