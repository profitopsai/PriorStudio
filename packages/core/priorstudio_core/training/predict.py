"""Inference path for trained PFN checkpoints.

Mirrors the local adapter's submit() shape — load the project from
its YAML files, instantiate the same Model object the trainer used,
restore weights from the checkpoint, then run a forward pass on the
operator-supplied input.

The input payload shape is prior-task-dependent. We auto-detect
based on the keys present:

- Regression PFN (linear-regression style):
    payload = {
      "context": {"x": [[..],..], "y": [..]},
      "query":   {"x": [[..],..]}
    }
    output  = {"predictions": [..], "task": "regression"}

- Discovery PFN (adjacency target):
    payload = {"X": [[..],..]}
    output  = {"adjacency": [[..],..], "task": "discovery"}

Anything outside these shapes returns ``{"error": "..."}`` rather
than a guess — the operator is the one closest to the prior's data
contract and should be the one to surface a clear failure.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any


def run_inference(
    *,
    manifest_path: Path,
    project_root: Path,
    checkpoint_dir: Path,
    payload: dict,
) -> dict:
    """Load the run's model + checkpoint and run one forward pass.

    Raises on missing files / unloadable state_dict / unrecognised
    payload shape; the CLI wraps the call so all failures land in a
    ``{"error": "..."}`` JSON output.
    """
    import torch

    from priorstudio_core.loaders import load_model, load_run
    from priorstudio_core.model import Model
    from priorstudio_core.registry import discover_in_project

    # Same project bootstrap as the local trainer adapter.
    sys.path.insert(0, str(project_root))
    try:
        discover_in_project(project_root)
        run = load_run(manifest_path)

        model_yaml = project_root / "models" / f"{run.model.id}.yaml"
        if not model_yaml.exists():
            raise FileNotFoundError(f"model.yaml not found at {model_yaml}")
        model_spec = load_model(model_yaml)
        model = Model(model_spec)

        # Restore weights. Trainer wrote a flat dict keyed by
        # "<block_name>.<param>"; redistribute back to each block's
        # state_dict so blocks added/reordered since training still
        # bind correctly.
        ckpt_path = checkpoint_dir / "model.pt"
        if not ckpt_path.exists():
            raise FileNotFoundError(f"checkpoint not found at {ckpt_path}")
        flat = torch.load(ckpt_path, map_location="cpu", weights_only=False)

        for name, mod in getattr(model, "modules", []):
            target = mod
            if hasattr(target, "module") and hasattr(target.module, "state_dict"):
                target = target.module
            sd = {
                k[len(name) + 1:]: v
                for k, v in flat.items()
                if k.startswith(name + ".")
            }
            if not sd:
                # No saved params for this block — skip rather than
                # error. Lets the checkpoint survive minor topology
                # changes (e.g. adding a new untrained head).
                continue
            target.load_state_dict(sd, strict=False)
            target.eval()

        return _dispatch_inference(model, payload)
    finally:
        if str(project_root) in sys.path:
            sys.path.remove(str(project_root))


def _dispatch_inference(model: Any, payload: dict) -> dict:
    """Route the payload to the regression or discovery path based
    on which keys are present. Returns the inference output dict."""
    import numpy as np
    import torch

    # Classification: {"context": {"x": ..., "labels": [...]}, "query": {"x": ...}}
    # Recognised by `labels` in the context. The model emits one logit
    # per query point; we sigmoid + threshold for the predicted class.
    if "context" in payload and "query" in payload and "labels" in (payload.get("context") or {}):
        ctx = payload["context"]
        qry = payload["query"]
        x_ctx = np.asarray(ctx.get("x", []), dtype=np.float32)
        lbl_ctx = np.asarray(ctx.get("labels", []), dtype=np.float32)
        x_qry = np.asarray(qry.get("x", []), dtype=np.float32)
        if x_ctx.ndim == 1:
            x_ctx = x_ctx.reshape(-1, 1)
        if x_qry.ndim == 1:
            x_qry = x_qry.reshape(-1, 1)
        with torch.no_grad():
            x = torch.from_numpy(x_qry).float().unsqueeze(0)
            out = x
            for _, mod in getattr(model, "modules", []):
                out = mod(out)
            logits = out.squeeze(0).squeeze(-1)
            probs = torch.sigmoid(logits).cpu().numpy().tolist()
            preds = [1 if p >= 0.5 else 0 for p in probs]
        return {
            "task": "classification",
            "predictions": preds,
            "probabilities": probs,
            "context_size": int(x_ctx.shape[0]),
            "query_size": int(x_qry.shape[0]),
        }

    # Regression: {"context": {"x": ..., "y": ...}, "query": {"x": ...}}
    if "context" in payload and "query" in payload:
        ctx = payload["context"]
        qry = payload["query"]
        x_ctx = np.asarray(ctx.get("x", []), dtype=np.float32)
        y_ctx = np.asarray(ctx.get("y", []), dtype=np.float32)
        x_qry = np.asarray(qry.get("x", []), dtype=np.float32)
        if x_ctx.ndim == 1:
            x_ctx = x_ctx.reshape(-1, 1)
        if x_qry.ndim == 1:
            x_qry = x_qry.reshape(-1, 1)
        # Today's _default_step trains the model on per-task batches
        # of (X, y). For a PFN, inference is supposed to attend over
        # the context and emit predictions for the query. Until the
        # block library has a real ICL head, we approximate by
        # running the model on the query x directly and returning
        # its per-point output. That's enough to demo "model trained
        # on linear functions emits something sensible on new x's".
        with torch.no_grad():
            x = torch.from_numpy(x_qry).float().unsqueeze(0)  # (1, N, 1)
            out = x
            for _, mod in getattr(model, "modules", []):
                out = mod(out)
            preds = out.squeeze(0).squeeze(-1).cpu().numpy().tolist()
        return {
            "task": "regression",
            "predictions": preds,
            "context_size": int(x_ctx.shape[0]),
            "query_size": int(x_qry.shape[0]),
        }

    # Discovery: {"X": [[...]]} → adjacency matrix
    if "X" in payload and "y" not in payload:
        X = np.asarray(payload["X"], dtype=np.float32)
        with torch.no_grad():
            xt = torch.from_numpy(X).float().unsqueeze(0)
            out = xt
            for _, mod in getattr(model, "modules", []):
                out = mod(out)
            adj = out.squeeze(0).cpu().numpy().tolist()
        return {"task": "discovery", "adjacency": adj}

    raise ValueError(
        "unrecognised inference payload shape — expected either "
        "{'context':{'x','y'},'query':{'x'}} (regression) or "
        "{'X':[[...]]} (discovery)."
    )
