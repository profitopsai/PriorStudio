"""Built-in architecture blocks. Imported eagerly to register them on package load."""

from . import (  # noqa: F401  (registers on import)
    causal_judgment,
    cross_attention,
    graph_posterior,
    heads,
    mlp,
    tabular,
    temporal_encoding,
    transformer,
)
