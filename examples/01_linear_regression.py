"""01 — Train a small PFN to do in-context Bayesian linear regression.

This is the simplest end-to-end PriorStudio example. We:

  1. Define a Bayesian linear regression prior — random (slope, intercept)
     plus i.i.d. Gaussian noise. Each sample is *packed* as a (context,
     query) token sequence and carries an ``n_ctx`` boundary so the
     default training step does in-context regression.
  2. Build a small transformer model (d_model=64, 3 layers).
  3. Train it for 2000 steps with the standard ``train_pfn`` loop —
     no custom step function needed.
  4. On held-out tasks, beat the mean baseline by ~100× and come within
     ~1.3× of the closed-form OLS (Bayesian-optimal) solution. The model
     has never seen these tasks — it's doing Bayesian inference from a
     single forward pass over the context.

Same prior, model, and run live as a full FM project in
``studies/linear-regression-bayes/`` and reproduce the same result via
``priorstudio run studies/linear-regression-bayes/runs/v0_1.yaml``. The
hosted studio uses that path; this script is the from-Python equivalent.

Expected runtime: ~40-60 seconds on a modern laptop CPU. Output: a saved
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

    Each call to sample() emits a single packed sequence:

        X[i] = (x_i, y_i, 1.0)   for i in 0..n_ctx-1     (context tokens)
        X[i] = (x_i, 0.0, 0.0)   for i in n_ctx..N-1     (query tokens)
        y    = the query y values
        n_ctx = boundary index

    `train_pfn`'s default step recognises ``n_ctx`` and slices the model's
    output at that boundary so the loss is computed only at query
    positions. The transformer sees the context tokens in the same
    sequence and learns to route from query → context to do regression.
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

        # Random ctx/query partition per task.
        perm = rng.permutation(num_points)
        x_p = x[perm]
        y_p = y[perm]

        n_ctx = int(num_points * CTX_FRAC)
        x_ctx, x_q = x_p[:n_ctx], x_p[n_ctx:]
        y_ctx, y_q = y_p[:n_ctx], y_p[n_ctx:]

        ctx_tok = np.stack([x_ctx, y_ctx, np.ones_like(x_ctx)], axis=1)
        q_tok = np.stack(
            [x_q, np.zeros_like(x_q), np.zeros_like(x_q)], axis=1
        )
        seq = np.concatenate([ctx_tok, q_tok], axis=0).astype(np.float32)

        return {
            "X": seq,
            "y": y_q,
            "n_ctx": n_ctx,
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


# ── 3. Training ───────────────────────────────────────────────────────────


def train(steps: int = 2000) -> tuple[Model, dict[str, Any]]:
    prior_spec = PriorSpec(
        id="bayesian_linear_demo",
        name="Bayesian linear regression (demo)",
        version="0.1.0",
        kind="tabular",
        parameters={
            "num_points": {"type": "int", "range": [64, 64]},
            "weight_std": {"type": "float", "range": [0.5, 1.5]},
            "noise_scale": {"type": "float", "range": [0.05, 0.2]},
        },
        outputs={
            "variables": [
                {"name": "X", "type": "tensor", "shape": "(N, 3)"},
                {"name": "y", "type": "tensor", "shape": "(n_query,)"},
                {"name": "n_ctx", "type": "scalar"},
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
    return model, train_pfn(model=model, prior=prior, run=run)


# ── 4. Verification ───────────────────────────────────────────────────────


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
        seq = task["X"]
        n_ctx = task["n_ctx"]

        x_ctx = seq[:n_ctx, 0:1]
        y_ctx = seq[:n_ctx, 1]
        x_q = seq[n_ctx:, 0:1]
        y_q = task["y"]

        with torch.no_grad():
            out = torch.from_numpy(seq).unsqueeze(0)
            for _, mod in model.modules:
                out = mod(out)
            preds = out[0, n_ctx:, 0].cpu().numpy()

        mean_pred = np.full_like(y_q, y_ctx.mean())
        a_ols, b_ols = np.polyfit(x_ctx[:, 0], y_ctx, 1)
        ols_pred = a_ols * x_q[:, 0] + b_ols

        pfn_sse += float(np.sum((preds - y_q) ** 2))
        mean_sse += float(np.sum((mean_pred - y_q) ** 2))
        ols_sse += float(np.sum((ols_pred - y_q) ** 2))
        total += y_q.shape[0]

    return {
        "pfn_mse": pfn_sse / total,
        "mean_mse": mean_sse / total,
        "ols_mse": ols_sse / total,
        "tasks": num_tasks,
    }


# ── 5. Glue ───────────────────────────────────────────────────────────────


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
        print("⚠ PFN does not beat the mean baseline — try more steps or a deeper model.")

    if pfn < metrics["ols_mse"] * 2.0:
        print(
            f"✓ PFN is within {pfn / metrics['ols_mse']:.2f}× of OLS — "
            "in-context Bayesian inference from a single forward pass. "
            "This is the headline claim of the PFN paper."
        )
    else:
        print("⚠ PFN is well above OLS — train longer / wider for tighter convergence.")


if __name__ == "__main__":
    main()
