<!-- Thanks for the PR! Fill in what's relevant; trim the sections that aren't. -->

## What this change does

<!-- One-line summary. -->

## Why

<!-- The motivation. Link an issue if there is one (`Fixes #123`). -->

## Type of change

- [ ] **New prior** (`priors/<slug>/`)
- [ ] **New scorer** (`packages/core/priorstudio_core/scorers/`)
- [ ] **New block** (`packages/core/priorstudio_core/blocks/`)
- [ ] **CLI / runtime** change in `packages/`
- [ ] **Docs / examples** only
- [ ] **Schema** change (`schemas/*.schema.json`) — needs CHANGELOG note
- [ ] Other (please explain):

## Smoke tests run locally

- [ ] `ruff check . && ruff format --check .`
- [ ] `mypy packages/`
- [ ] `pytest tests/`
- [ ] (if you touched a prior) `priorstudio validate priors/<slug>/`
- [ ] (if you touched a prior) `priorstudio sample priors/<slug>/prior.yaml --seeds 3 --num-points 5`

## Honesty checks

- [ ] If this PR adds a paper-replication template or scorer, the README
      is explicit about what's matched and what isn't vs the published numbers.
- [ ] No dependency on the hosted studio at `priorstudio.ai` —
      everything here must work locally.
