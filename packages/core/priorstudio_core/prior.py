"""Prior abstraction.

A Prior is a synthetic data generator — the defining mechanic of a PFN.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, Field


class PriorParameter(BaseModel):
    type: str
    range: list[float] | None = None
    choices: list[Any] | None = None
    default: Any = None
    description: str | None = None


class PriorOutputVariable(BaseModel):
    name: str
    type: str
    shape: str | None = None
    description: str | None = None


class PriorOutputs(BaseModel):
    variables: list[PriorOutputVariable]


class PriorSpec(BaseModel):
    """Typed view of prior.yaml. Validated against schemas/prior.schema.json by the CLI."""

    id: str
    name: str
    version: str
    kind: str
    description: str | None = None
    parameters: dict[str, PriorParameter] = Field(default_factory=dict)
    outputs: PriorOutputs
    citations: list[str] = Field(default_factory=list)
    implementation: str = "prior.py"


class Prior(ABC):
    """Subclass and decorate with @register_prior(\"<id>\") to bind to a PriorSpec."""

    spec: PriorSpec

    @abstractmethod
    def sample(self, *, seed: int, **params: Any) -> dict[str, Any]:
        """Return one sample as a dict keyed by `outputs.variables[*].name`."""

    def sample_batch(self, *, batch_size: int, seed: int, **params: Any) -> list[dict[str, Any]]:
        return [self.sample(seed=seed + i, **params) for i in range(batch_size)]
