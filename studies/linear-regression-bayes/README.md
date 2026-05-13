# Study: in-context Bayesian linear regression

> **Claim**: a tiny transformer (3 layers, d_model=64, ~50k params) trained
> on synthetic random linear functions does Bayesian linear regression on
> data it has never seen — in a single forward pass — landing within
> ~1.15× of the closed-form OLS solution. Trains in ~47 seconds on CPU.

This is the canonical PFN setup from Müller et al. ICLR 2022, reproduced
end-to-end with PriorStudio. Every artifact is in this directory; nothing
is hidden in the binary.

## Result

From a fresh run on a laptop CPU (M-series, Python 3.13, PyTorch 2.x):

| metric                       | value     | notes |
|------------------------------|-----------|-------|
| **PFN MSE**                  | **0.0118**| in-context inference on 50 held-out tasks |
| Mean baseline                | 1.1371    | predict the context mean — a useless model |
| OLS baseline                 | 0.0103    | closed-form Bayesian posterior mean |
| Training loss (final, step 2000) | 0.0121 | converged |
| Training wall time           | 46.8 s    | CPU only |

**The PFN beats the mean baseline by ~96× and lands within 1.15× of the
Bayesian-optimal OLS solution.** It has never seen the (a, b) parameters
that generated the held-out tasks; the entire inference happens in one
forward pass through the transformer, conditioned on the context (x, y)
pairs the prior packs into the same sequence as the query x's.

Full metrics: [`outputs/metrics.json`](outputs/metrics.json).
Training log (sampled): [`outputs/run.log.snippet`](outputs/run.log.snippet).

## Reproduce

Two equivalent paths. The CLI path is what the hosted studio runs under
the hood; the script path is the from-Python equivalent.

### A. From the CLI (same path the hosted studio uses)

```bash
pip install "priorstudio-core[torch]" priorstudio
git clone https://github.com/profitopsai/PriorStudio
cd PriorStudio

priorstudio validate studies/linear-regression-bayes/priors/bayesian_linear/
priorstudio run     studies/linear-regression-bayes/runs/v0_1.yaml
```

This loads `prior.yaml` + `prior.py`, builds the model from `model.yaml`,
runs `train_pfn` with default settings, and writes a checkpoint at
`./checkpoint/model.pt`. Exit status is non-zero if anything fails.

### B. From Python (same prior + model + hparams, plus held-out eval)

```bash
pip install "priorstudio-core[torch]"
python examples/01_linear_regression.py
```

Trains the same model in the same way and adds an in-script evaluation
on 50 held-out tasks against the mean and OLS baselines. This is the
script that produced the table above.

Both paths should produce roughly the same training loss and metrics; any
variance comes from the in-script eval drawing fresh tasks each run.

## What's in here

```
studies/linear-regression-bayes/
├── README.md                              ← you are here
├── priors/
│   └── bayesian_linear/
│       ├── prior.yaml                     ← spec (schema-validated)
│       └── prior.py                       ← @register_prior implementation
├── models/
│   └── in_context_pfn.yaml                ← 3-layer transformer config
├── runs/
│   └── v0_1.yaml                          ← hparams (steps, lr, seed)
└── outputs/
    ├── metrics.json                       ← the result above
    └── run.log.snippet                    ← sampled training log
```

The prior emits each task as a packed `(context, query)` token sequence
with an `n_ctx` boundary. `priorstudio_core`'s default training step
slices the model's logits at `n_ctx` so the loss is computed only at
query positions; context tokens are visible to the transformer's
self-attention so it can route information from them. This is the
mechanism that lets one forward pass produce a Bayesian posterior mean.

## Caveats — what this study does *not* show

- **Scale.** This is a 3-layer / d_model=64 model on a univariate prior.
  Real PFNs (TabPFN, etc.) are 10-100× larger and trained for hours, not
  minutes. The point is that the *training-once, infer-via-context*
  paradigm works end-to-end, not that this particular model is
  competitive with TabPFN.
- **Distribution.** The held-out tasks come from the same prior as
  training. PFN performance on distribution-shifted tasks is a separate
  empirical question.
- **No structured uncertainty.** The model outputs point predictions, not
  the full posterior. Adding a Gaussian head + NLL loss is a natural
  next step but not in scope here.

## Citation

The PFN idea is from:

> Müller, S., Hollmann, N., Arango, S. P., Grabocka, J., & Hutter, F. (2022).
> *Transformers Can Do Bayesian Inference.* ICLR 2022.
> https://arxiv.org/abs/2112.10510

This study is an independent reimplementation; no code is taken from the
authors' repository, and the model architecture / training loop are
PriorStudio's own. The conceptual lineage is the paper.

## License

Apache-2.0, same as the rest of this repo.
