"""PriorStudio core abstractions."""

from . import blocks as _blocks  # noqa: F401 — registers built-in blocks on import
from .datasets import DatasetUnavailable, RegistryDatasetLoader
from .eval import Eval, EvalSpec
from .model import Model, ModelSpec
from .prior import Prior, PriorSpec
from .registry import (
    get_block,
    get_eval,
    get_prior,
    list_blocks,
    list_evals,
    list_priors,
    register_block,
    register_eval,
    register_prior,
)
from .run import Run, RunSpec

__version__ = "0.6.0"

__all__ = [
    "DatasetUnavailable",
    "Eval",
    "EvalSpec",
    "Model",
    "ModelSpec",
    "Prior",
    "PriorSpec",
    "RegistryDatasetLoader",
    "Run",
    "RunSpec",
    "get_block",
    "get_eval",
    "get_prior",
    "list_blocks",
    "list_evals",
    "list_priors",
    "register_block",
    "register_eval",
    "register_prior",
]
