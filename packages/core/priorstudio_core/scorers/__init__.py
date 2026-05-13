"""Built-in dataset scorers.

A "scorer" owns the full pipeline from raw dataset bytes to metric value:
load the dataset via RegistryDatasetLoader, run the trained model on the
relevant slice, compute the metrics declared on the eval spec.

The CLI's local adapter looks up scorers by eval slug after training
completes. Templates that want a real-data eval ship an eval YAML whose
slug matches a key in BUILTIN_SCORERS; any unmatched slug is treated as
synthetic-only (no real-data scoring step).

To add a new scorer:
  1. Implement a subclass of DatasetScorer in this package.
  2. Register it in BUILTIN_SCORERS below by eval slug.
"""

from __future__ import annotations

from .base import DatasetScorer, ScorerResult
from .in_context_regression_ols import InContextRegressionVsOLS
from .m4_monthly_forecast import M4MonthlyForecast

BUILTIN_SCORERS: dict[str, DatasetScorer] = {
    "m4_monthly_mse": M4MonthlyForecast(),
    "in_context_regression_ols": InContextRegressionVsOLS(),
}

__all__ = ["BUILTIN_SCORERS", "DatasetScorer", "ScorerResult"]
