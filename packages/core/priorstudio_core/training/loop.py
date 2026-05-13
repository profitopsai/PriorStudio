"""Minimal PFN training loop.

This is *the* canonical PFN loop: each step, sample a task from the prior,
split into context / query, predict the query from the context.

It's intentionally framework-agnostic at the boundary: the model is anything
callable; the prior returns numpy arrays; the loss is supplied by the caller.
The default loss assumes a discovery-style adjacency target and uses BCE.

JSON-line progress
──────────────────
When the env var ``PRIORSTUDIO_JSON_PROGRESS=1`` is set, every step prints one
line of JSON to stdout (line-buffered). The PriorStudio API parses these lines
to drive the live training UI; running in a normal terminal still works (the
output just looks a bit verbose).

Event schema (one JSON object per line):
    { "event": "start",  "steps": int, "batch_size": int, "lr": float, ... }
    { "event": "step",   "step": int, "loss": float, "elapsed_s": float }
    { "event": "done",   "status": "ok"|"skipped"|"failed", "final_loss": float, ... }
    { "event": "error",  "message": str }
"""

from __future__ import annotations

import json
import os
import sys
import time
from collections.abc import Callable
from typing import Any

from ..prior import Prior
from ..run import RunSpec

_EMIT_JSON = os.environ.get("PRIORSTUDIO_JSON_PROGRESS") == "1"


def _emit(event: str, **fields: Any) -> None:
    """Print one JSON-progress line to stdout, line-buffered, when enabled."""
    if not _EMIT_JSON:
        return
    payload = {"event": event, **fields}
    sys.stdout.write(json.dumps(payload) + "\n")
    sys.stdout.flush()


def _default_step(model: Any, batch: list[dict], hp: dict) -> Any:
    """Default step. Auto-detects the prior's task by inspecting batch
    keys and picks an appropriate loss:

    - ``A`` present       → discovery / structure: BCE-with-logits
                            against the ground-truth adjacency.
    - ``labels`` present  → classification: BCE-with-logits against
                            0/1 labels (binary). The model emits one
                            logit per point.
    - ``y`` present       → regression: MSE between model output and
                            ``y`` (broadcast to the model's output
                            shape).
    - none of the above   → no-op skip; the loop reports this as a
                            structural mismatch so the operator can
                            fix the prior or provide a custom step_fn.

    Falls back to a no-op if torch is unavailable.
    """
    try:
        import torch
        import torch.nn.functional as F  # noqa: N812 — F is the canonical alias for torch.nn.functional
    except ImportError:
        return None

    if not batch:
        return None
    sample0 = batch[0]
    has_A = "A" in sample0
    has_labels = "labels" in sample0
    has_y = "y" in sample0
    if not has_A and not has_labels and not has_y:
        # Prior didn't emit a recognised target. The loop turns this
        # into a {"status": "skipped", "reason": "step_fn returned None"}
        # which the run-detail page surfaces — caller can write a
        # project-specific step_fn for novel target shapes.
        return None

    X = torch.stack([torch.from_numpy(b["X"]).float() for b in batch])

    logits = X
    for _, mod in getattr(model, "modules", []):
        logits = mod(logits)

    if has_A:
        A = torch.stack([torch.from_numpy(b["A"]).float() for b in batch])
        if logits.shape[-2:] != A.shape[-2:]:
            return None
        return F.binary_cross_entropy_with_logits(logits, A)

    if has_labels:
        # Classification branch. Targets are 0/1 per point; the model
        # emits one logit per point (scalar_head with d_out=1). Squeeze
        # the trailing feature axis to align with the (B, N) target.
        labels = torch.stack([torch.from_numpy(b["labels"]).float() for b in batch])
        pred = logits
        if pred.dim() == labels.dim() + 1 and pred.shape[-1] == 1:
            pred = pred.squeeze(-1)
        if pred.shape != labels.shape:
            return None
        return F.binary_cross_entropy_with_logits(pred, labels)

    # Regression branch
    y = torch.stack([torch.from_numpy(b["y"]).float() for b in batch])
    # In-context regression: when the prior packs each task as a
    # (context, query) sequence and emits n_ctx, only the query
    # positions are scored. Context positions feed the transformer
    # so it can attend from query → context — the actual PFN trick.
    n_ctx = sample0.get("n_ctx")
    if n_ctx is not None:
        logits = logits[:, int(n_ctx) :, :]
    # Squeeze the trailing-1 feature axis off the model output if the
    # target is shaped (B, N) — common for scalar regressions like
    # linear_regression where the model emits a per-point prediction.
    pred = logits
    if pred.dim() == y.dim() + 1 and pred.shape[-1] == 1:
        pred = pred.squeeze(-1)
    if pred.shape != y.shape:
        return None
    return F.mse_loss(pred, y)


def train_pfn(
    model: Any,
    prior: Prior,
    run: RunSpec,
    *,
    step_fn: Callable[[Any, list[dict], dict], Any] | None = None,
    on_step: Callable[[int, float], None] | None = None,
) -> dict[str, Any]:
    """Run the PFN training loop in-process. Returns a results dict.

    Set ``PRIORSTUDIO_JSON_PROGRESS=1`` in the env to emit one JSON line per
    step to stdout (consumed by the PriorStudio API to drive the live UI).
    """
    try:
        import torch
    except ImportError:
        result = {
            "status": "skipped",
            "reason": "torch not installed; install priorstudio-core[torch] to train",
            "steps": 0,
        }
        _emit("done", **result)
        return result

    hp = run.hyperparams
    steps = int(hp.get("steps", 100))
    batch_size = int(hp.get("batch_size", 8))
    lr = float(hp.get("lr", 1e-4))
    seed = int(hp.get("seed", 42))

    _emit(
        "start",
        steps=steps,
        batch_size=batch_size,
        lr=lr,
        seed=seed,
        prior_id=run.prior.id,
        model_id=run.model.id,
    )

    params = []
    for _, mod in getattr(model, "modules", []):
        if hasattr(mod, "parameters"):
            params.extend(mod.parameters())
        elif hasattr(mod, "module") and hasattr(mod.module, "parameters"):
            params.extend(mod.module.parameters())
    if not params:
        result = {"status": "skipped", "reason": "no trainable parameters found", "steps": 0}
        _emit("done", **result)
        return result

    optim = torch.optim.AdamW(params, lr=lr)
    step_fn = step_fn or _default_step

    losses: list[float] = []
    t0 = time.time()
    prior_params = {**run.prior.overrides}

    for step in range(steps):
        batch = prior.sample_batch(batch_size=batch_size, seed=seed + step, **prior_params)
        loss = step_fn(model, batch, hp)
        if loss is None:
            result = {"status": "skipped", "reason": "step_fn returned None", "steps": step}
            _emit("done", **result)
            return result

        optim.zero_grad()
        loss.backward()
        optim.step()

        loss_val = float(loss.item())
        losses.append(loss_val)
        elapsed = time.time() - t0
        _emit("step", step=step, loss=loss_val, elapsed_s=elapsed)
        if on_step:
            on_step(step, loss_val)

    # Persist the trained model so /runs/:id/predict can load it.
    # Two files, both under <cwd>/checkpoint/:
    #   model.pt      — state_dict from the trainable modules
    #   topology.json — the block list + per-block init config, so
    #                   the inference path can reconstruct the
    #                   model object before loading state_dict
    # This is per-run-cwd; the API runner copies the directory out
    # to stable storage after the CLI exits.
    checkpoint_dir: str | None = None
    try:
        import json
        import os

        ckpt_dir = os.path.join(os.getcwd(), "checkpoint")
        os.makedirs(ckpt_dir, exist_ok=True)
        # Save weights — flat dict keyed by "<block_name>.<param>"
        # so the inference loader knows which weights go where even
        # if blocks change order.
        sd: dict[str, Any] = {}
        for name, mod in getattr(model, "modules", []):
            if hasattr(mod, "state_dict"):
                for k, v in mod.state_dict().items():
                    sd[f"{name}.{k}"] = v
            elif hasattr(mod, "module") and hasattr(mod.module, "state_dict"):
                for k, v in mod.module.state_dict().items():
                    sd[f"{name}.{k}"] = v
        torch.save(sd, os.path.join(ckpt_dir, "model.pt"))
        # Topology: the same list the trainer iterated, in order.
        # Stored as JSON so non-Python tooling can introspect it.
        topology = {
            "model_id": run.model.id,
            "prior_id": run.prior.id,
            "blocks": [
                {"name": name, "type": type(mod).__name__}
                for name, mod in getattr(model, "modules", [])
            ],
        }
        with open(os.path.join(ckpt_dir, "topology.json"), "w") as fh:
            json.dump(topology, fh, indent=2)
        checkpoint_dir = ckpt_dir
    except Exception as e:
        # Don't fail the run if checkpoint save fails — the metrics
        # are still useful. Surface the reason in the result so the
        # runner can decide whether to copy a checkpoint or not.
        checkpoint_dir = None
        _emit("log", line=f"checkpoint save failed: {type(e).__name__}: {e}")

    result = {
        "status": "ok",
        "steps": steps,
        "final_loss": losses[-1] if losses else None,
        "mean_loss_last_10pct": (
            sum(losses[-max(1, steps // 10) :]) / max(1, steps // 10) if losses else None
        ),
        "wall_time_s": time.time() - t0,
        "checkpoint_dir": checkpoint_dir,
    }
    _emit("done", **result)
    return result
