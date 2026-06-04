"""Phase 1 — the strongly-typed expression-tree engine.

Encode strategies as typed trees (:mod:`tree`), built from a vectorized primitive DSL
(:mod:`primitives`, :mod:`types`), generated randomly (:mod:`generate`), evaluated over a
panel (:mod:`evaluate`, :mod:`panel`), and simplified (:mod:`simplify`).
"""

from alphaforge.core.evaluate import evaluate
from alphaforge.core.generate import RandomTreeGenerator
from alphaforge.core.panel import Panel
from alphaforge.core.primitives import REGISTRY, Kind, Primitive
from alphaforge.core.simplify import simplify
from alphaforge.core.tree import (
    InvalidTree,
    Node,
    from_dict,
    from_json,
    is_valid,
    to_dict,
    to_json,
    validate,
)
from alphaforge.core.types import DType, is_subtype

__all__ = [
    "REGISTRY",
    "DType",
    "InvalidTree",
    "Kind",
    "Node",
    "Panel",
    "Primitive",
    "RandomTreeGenerator",
    "evaluate",
    "from_dict",
    "from_json",
    "is_subtype",
    "is_valid",
    "simplify",
    "to_dict",
    "to_json",
    "validate",
]
