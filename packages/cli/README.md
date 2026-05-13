# priorstudio — CLI

The command-line interface for [PriorStudio](https://github.com/profitopsai/priorstudio),
the toolkit for training [prior-fitted foundation models](https://arxiv.org/abs/2112.10510).

## Install

```bash
pip install priorstudio
```

For training (requires PyTorch):

```bash
pip install "priorstudio[torch]"
```

## Commands

```text
priorstudio init <dir>            # scaffold a new FM project
priorstudio validate <path>       # check artifacts against JSON Schema
priorstudio lint <project>        # cross-reference + style checks
priorstudio sample <prior.yaml>   # draw N tasks from a prior
priorstudio run <run.yaml>        # execute a training run end-to-end
priorstudio predict <run-dir>     # inference against a trained checkpoint
priorstudio export <project>     # tar-gzipped project archive
```

Run `priorstudio --help` for the full list and `<cmd> --help` for each
subcommand's flags.

## What this CLI is for

PriorStudio organises every PFN project around five first-class
artifacts: **priors** (synthetic data generators), **models** (block
compositions), **evals** (benchmarks + metrics), **runs** (training
manifests), and **initiatives** (research workstreams). This CLI
operates on the file layout those artifacts produce — scaffolding new
projects, validating them, running training, and exporting them for
sharing.

The full story (concepts, architecture, examples, marketplace catalog)
lives at the main repo:
**[github.com/profitopsai/priorstudio](https://github.com/profitopsai/priorstudio)**

## License

Apache-2.0.
