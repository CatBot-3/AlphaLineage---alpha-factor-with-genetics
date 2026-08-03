"""Analysis and provider plumbing for the agent.

This package used to be a feature of its own — a one-shot factor explainer with its own prompt,
its own output schema, and a hand-written dictionary of aspect labels. Phase 11 folded all of
that into a single conversational agent, and what is left here is the support library the agent
draws on:

* :mod:`~alphalineage.explain.anatomy` — *measuring* an expression tree. Compounded effective
  lookback, window profile, size, fields read, unit consistency, structural diagnostics. All
  arithmetic over the tree's shape, so it works on any operator including ones added next year.
  The hand-written aspect taxonomy that used to live beside it is gone: semantic labels now come
  from the model, which is what judgement is for.
* :mod:`~alphalineage.explain.corpus` / :mod:`~alphalineage.explain.retrieval` — an
  embedding-free BM25 index over this workspace's own history, reachable through the agent's
  ``search_workspace`` tool rather than stuffed into every prompt.
* :mod:`~alphalineage.explain.providers` / :mod:`~alphalineage.explain.credentials` — the
  provider catalog, the egress allowlist, and key resolution that never echoes a secret.

Not investment advice. Research output only (invariant 8).
"""

from alphalineage.explain.anatomy import (
    Measurement,
    base_name,
    effective_lookback,
    formula_text,
    measure,
    render_markdown,
)
from alphalineage.explain.corpus import Document, build_corpus
from alphalineage.explain.credentials import LLMConfig, load_config, provider_status
from alphalineage.explain.providers import PROVIDERS, LLMError, LLMRequest, complete
from alphalineage.explain.retrieval import Hit, Query, retrieve

__all__ = [
    "PROVIDERS",
    "Document",
    "Hit",
    "LLMConfig",
    "LLMError",
    "LLMRequest",
    "Measurement",
    "Query",
    "base_name",
    "build_corpus",
    "complete",
    "effective_lookback",
    "formula_text",
    "load_config",
    "measure",
    "provider_status",
    "render_markdown",
    "retrieve",
]
