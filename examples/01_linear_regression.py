"""01 — Train a small PFN to do in-context Bayesian linear regression.

This is the simplest end-to-end PriorStudio example. We:

  1. Define a Bayesian linear regression prior — random (slope, intercept)
     plus i.i.d. Gaussian noise.
  2. Build a small transformer model (d_model=64, 3 layers).
  3. Train it for 2000 steps with a custom in-context step function that
     packs each task as [(x_ctx, y_ctx, 1), …, (x_query, 0, 0), …] tokens.
  4. On held-out tasks, beat the mean baseline by ~100× and come within
     ~1.2× of the closed-form OLS (Bayesian-optimal) solution. The model
     has never seen these tasks — it's doing Bayesian inference from a
     single forward pass over the context.

Expected runtime: ~40–60 seconds on a modern laptop CPU. Output: a saved
checkpoint at ``checkpoint/model.pt`` you can load for inference, plus a
final MSE comparison printed at the end.

Requires: ``pip install priorstudio-core[torch]``
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
from priorstudio_core import Model, ModelSpec, RunSpec
from priorstudio_core.model import BlockConfig, OutputHead
from priorstudio_core.prior import Prior, PriorSpec
from priorstudio_core.registry import register_prior
from priorstudio_core.run import ModelRef, PriorRef
from priorstudio_core.training import train_pfn

# Cosmetic: PyTorch warns about nested-tensor optimisations not applying
# when norm_first=True. Doesn't affect correctness.
warnings.filterwarnings("ignore", message=".*enable_nested_tensor.*")

# Fraction of each task's points used as context; the rest are queries.
CTX_FRAC = 0.75

# ── 1. The prior ──────────────────────────────────────────────────────────


@register_prior("bayesian_linear_demo")
class BayesianLinearPrior(Prior):
    """y = a*x + b + ε, with (a, b) ~ N(0, weight_std) and ε ~ N(0, noise_scale).

    A correct PFN trained on this prior, given a context of (x, y) pairs
    from a single (a, b), predicts on new x's in a way that approaches
    the closed-form Bayesian posterior mean — without ever inverting a
    covariance matrix at inference time.
    """

    def sample(
        self,
        *,
        seed: int,
        num_points: int = 64,
        weight_std: float = 1.0,
        noise_scale: float = 0.1,
        x_range: float = 2.0,
        **_: Any,
    ) -> dict[str, Any]:
        rng = np.random.default_rng(seed)
        a = float(rng.normal(0.0, weight_std))
        b = float(rng.normal(0.0, weight_std))
        x = rng.uniform(-x_range, x_range, size=num_points).astype(np.float32)
        y = (a * x + b + rng.normal(0.0, noise_scale, size=num_points)).astype(np.float32)
        return {
            "X": x.reshape(-1, 1),
            "y": y,
            "a_true": a,
            "b_true": b,
        }


# ── 2. The model ──────────────────────────────────────────────────────────


def build_model() -> Model:
    """Small tabular PFN — converges on CPU in under a minute."""
    spec = ModelSpec(
        id="linear_regression_demo_model",
        name="Linear-regression demo PFN",
        version="0.1.0",
        description="3-layer tabular PFN for in-context linear regression.",
        blocks=[
            BlockConfig(type="tabular_embedder", config={"d_model": 64}),
            BlockConfig(
                type="transformer_encoder",
                config={
                    "d_model": 64,
                    "n_heads": 4,
                    "n_layers": 3,
                    "dropout": 0.0,
                },
            ),
            BlockConfig(type="scalar_head", config={"d_model": 64, "d_out": 1}),
        ],
        output_heads=[OutputHead(name="pred_y", task="forecast")],
    )
    return Model(spec)


# ── 3. In-context step function ───────────────────────────────────────────


def _pack_in_context_sequence(
    x_all: np.ndarray, y_all: np.ndarray, n_ctx: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Build a single training sequence + the y-targets for the query points.

    The transformer sees tokens of shape (x, y_or_0, is_context_flag):
      - Context tokens carry the real y and a flag of 1.
      - Query tokens carry y=0 and flag=0; the model must predict y for these.

    Both context and query positions live in the same sequence so the
    transformer's self-attention can route information from context to
    query — which is the whole point of in-context inference.
    """
    n_total = x_all.shape[0]
    perm = rng.permutation(n_total)
    x_perm = x_all[perm]
    y_perm = y_all[perm]
    x_ctx, x_q = x_perm[:n_ctx], x_perm[n_ctx:]
    y_ctx, y_q = y_perm[:n_ctx], y_perm[n_ctx:]

    ctx_tok = np.concatenate(
        [x_ctx, y_ctx[:, None], np.ones((n_ctx, 1), dtype=np.float32)], axis=1
    )
    q_tok = np.concatenate(
        [
            x_q,
            np.zeros((x_q.shape[0], 1), dtype=np.float32),
            np.zeros((x_q.shape[0], 1), dtype=np.float32),
        ],
        axis=1,
    )
    seq = np.concatenate([ctx_tok, q_tok], axis=0).astype(np.float32)
    return seq, y_q


def in_context_step(model: Any, batch: list[dict], hp: dict) -> Any:
    """Custom train_pfn step that packs each task as a (context, query) sequence.

    The default step in priorstudio_core only feeds X to the model — fine for
    structure-discovery priors that emit a target adjacency, but useless for
    in-context regression where the model needs to *see* the (x, y) context
    pairs alongside the query x's. So we override it here.
    """
    import torch
    import torch.nn.functional as F

    n_total = batch[0]["X"].shape[0]
    n_ctx = int(n_total * CTX_FRAC)
    rng = np.random.default_rng()

    inputs: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    for b in batch:
        seq, y_q = _pack_in_context_sequence(b["X"], b["y"], n_ctx, rng)
        inputs.append(seq)
        targets.append(y_q)

    inp = torch.from_numpy(np.stack(inputs))  # (B, N, 3)
    tgt = torch.from_numpy(np.stack(targets))  # (B, n_query)

    out = inp
    for _, mod in model.modules:
        out = mod(out)

    # Predictions at query positions only — the model also outputs at the
    # context positions, but the loss only looks at queries.
    pred = out[:, n_ctx:, 0]  # (B, n_query)
    return F.mse_loss(pred, tgt)


# ── 4. Training ───────────────────────────────────────────────────────────


def train(steps: int = 2000) -> tuple[Model, dict[str, Any]]:
    prior_spec = PriorSpec(
        id="bayesian_linear_demo",
        name="Bayesian linear regression (demo)",
        version="0.1.0",
        kind="tabular",
        parameters={
            # Fixed-size sequences keep the batched step simple; 64 points
            # is small enough to train fast and large enough to be a real
            # Bayesian-inference target.
            "num_points": {"type": "int", "range": [64, 64]},
            "weight_std": {"type": "float", "range": [0.5, 1.5]},
            "noise_scale": {"type": "float", "range": [0.05, 0.2]},
        },
        outputs={
            "variables": [
                {"name": "X", "type": "tensor", "shape": "(N, 1)"},
                {"name": "y", "type": "tensor", "shape": "(N,)"},
            ]
        },
    )
    prior = BayesianLinearPrior()
    prior.spec = prior_spec

    run = RunSpec(
        id="linear_regression_demo_run",
        prior=PriorRef(id="bayesian_linear_demo", version="0.1.0"),
        model=ModelRef(id="linear_regression_demo_model", version="0.1.0"),
        evals=[],
        hyperparams={
            "optimizer": "adamw",
            "lr": 5e-4,
            "batch_size": 32,
            "steps": steps,
            "seed": 42,
        },
    )

    model = build_model()
    return model, train_pfn(model=model, prior=prior, run=run, step_fn=in_context_step)


# ── 5. Verification ───────────────────────────────────────────────────────


def evaluate(model: Model, num_tasks: int = 50, num_points: int = 80) -> dict[str, float]:
    """On fresh tasks the model has never seen, compare in-context PFN
    predictions against (a) the mean baseline and (b) closed-form OLS."""
    import torch

    prior = BayesianLinearPrior()

    pfn_sse = 0.0
    mean_sse = 0.0
    ols_sse = 0.0
    total = 0

    for k in range(num_tasks):
        task = prior.sample(
            seed=10_000 + k, num_points=num_points, weight_std=1.0, noise_scale=0.1
        )
        x_all = task["X"]
        y_all = task["y"]
        n_ctx = int(num_points * CTX_FRAC)
        x_ctx, x_q = x_all[:n_ctx], x_all[n_ctx:]
        y_ctx, y_q = y_all[:n_ctx], y_all[n_ctx:]

        # Build the same packed sequence the model was trained on.
        ctx_tok = np.concatenate(
            [x_ctx, y_ctx[:, None], np.ones((n_ctx, 1), dtype=np.float32)], axis=1
        )
        n_q = num_points - n_ctx
        q_tok = np.concatenate(
            [
                x_q,
                np.zeros((n_q, 1), dtype=np.float32),
                np.zeros((n_q, 1), dtype=np.float32),
            ],
            axis=1,
        )
        seq = np.concatenate([ctx_tok, q_tok], axis=0).astype(np.float32)

        with torch.no_grad():
            out = torch.from_numpy(seq).unsqueeze(0)
            for _, mod in model.modules:
                out = mod(out)
            preds = out[0, n_ctx:, 0].cpu().numpy()

        # Baselines: mean of the visible y's, and OLS on (x_ctx, y_ctx).
        mean_pred = np.full_like(y_q, y_ctx.mean())
        a_ols, b_ols = np.polyfit(x_ctx[:, 0], y_ctx, 1)
        ols_pred = a_ols * x_q[:, 0] + b_ols

        pfn_sse += float(np.sum((preds - y_q) ** 2))
        mean_sse += float(np.sum((mean_pred - y_q) ** 2))
        ols_sse += float(np.sum((ols_pred - y_q) ** 2))
        total += n_q

    return {
        "pfn_mse": pfn_sse / total,
        "mean_mse": mean_sse / total,
        "ols_mse": ols_sse / total,
        "tasks": num_tasks,
    }


# ── 6. Glue ───────────────────────────────────────────────────────────────


def main() -> None:
    print("Training a 3-layer PFN on Bayesian linear regression (2000 steps)…")
    print("Each step samples a fresh task; the model learns to do regression")
    print("from the context (x, y) pairs in a single forward pass.")
    print()

    model, result = train(steps=2000)
    print()
    print(
        f"Training: status={result.get('status')!r}  "
        f"final_loss={result.get('final_loss'):.4f}  "
        f"wall_time={result.get('wall_time_s'):.1f}s"
    )
    print()

    if result.get("status") != "ok":
        print(f"Training did not complete cleanly: {result.get('reason')}")
        return

    print("Evaluating on 50 fresh held-out tasks (in-context inference)…")
    metrics = evaluate(model, num_tasks=50, num_points=80)
    print()
    print(f"  PFN MSE        : {metrics['pfn_mse']:.4f}")
    print(
        f"  Mean baseline  : {metrics['mean_mse']:.4f}    "
        "(predict the context mean — what a useless model does)"
    )
    print(
        f"  OLS baseline   : {metrics['ols_mse']:.4f}    "
        "(closed-form linear fit on the context — the Bayesian-optimal target)"
    )
    print()

    pfn = metrics["pfn_mse"]
    if pfn < metrics["mean_mse"]:
        print(
            f"✓ PFN beats the mean baseline by {metrics['mean_mse'] / pfn:.0f}× — "
            "it has learned to read the context."
        )
    else:
        print(
            "⚠ PFN does not beat the mean baseline — try more steps or a deeper model."
        )

    if pfn < metrics["ols_mse"] * 2.0:
        print(
            f"✓ PFN is within {pfn / metrics['ols_mse']:.2f}× of OLS — "
            "in-context Bayesian inference from a single forward pass. "
            "This is the headline claim of the PFN paper."
        )
    else:
        print(
            "⚠ PFN is well above OLS — train longer / wider for tighter convergence."
        )


if __name__ == "__main__":
    main()
