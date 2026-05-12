# Getting started

## Install

```bash
pip install priorstudio
```

(or, from source, while the package is pre-release:)

```bash
git clone https://github.com/profitopsai/priorstudio
cd priorstudio/packages/cli
pip install -e .
```

## Create a project

```bash
priorstudio init my-fm
cd my-fm
```

You now have:

```
my-fm/
├── ROADMAP.md
├── initiatives/0001-define-base-prior.md
├── priors/example_linear_scm/{prior.yaml, prior.py, prior.md}
├── models/example_transformer.yaml
├── evals/example_sachs.yaml
├── literature/{references.bib, summaries/mueller2022pfn.md}
└── runs/example_run.yaml
```

## Validate

```bash
priorstudio validate
```

Catches schema violations across every artifact.

## Add your first real prior

```bash
mkdir -p priors/my_prior
cp priors/example_linear_scm/prior.yaml priors/my_prior/prior.yaml
cp priors/example_linear_scm/prior.py priors/my_prior/prior.py
cp priors/example_linear_scm/prior.md priors/my_prior/prior.md
# edit each file
priorstudio validate
```

## Open a PR

PriorStudio is git-native. Every change — new prior, new initiative, updated roadmap — goes through normal code review. The schema validation runs in CI.

## Bringing in priors from an existing package

Already have a Python codebase with prior classes you want to expose through the web UI? See [import-priors.md](import-priors.md) — it covers `priorstudio author wrap`, wheel bundling for private deps, and end-to-end push to a project.
