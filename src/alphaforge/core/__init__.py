"""Phase 1 — the strongly-typed expression-tree engine.

Encode strategies as typed trees (:mod:`tree`), built from a vectorized primitive DSL
(:mod:`primitives`, :mod:`types`), generated randomly (:mod:`generate`), evaluated over a
panel (:mod:`evaluate`, :mod:`panel`), and simplified (:mod:`simplify`).
"""

from alphaforge.core.evaluate import evaluate
from alphaforge.core.fitness import daily_ic, forward_returns, ic_ir, mean_ic, score_tree
from alphaforge.core.generate import RandomTreeGenerator
from alphaforge.core.gp import GP, GPConfig, Individual
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
    "GP",
    "REGISTRY",
    "DType",
    "GPConfig",
    "Individual",
    "InvalidTree",
    "Kind",
    "Node",
    "Panel",
    "Primitive",
    "RandomTreeGenerator",
    "daily_ic",
    "evaluate",
    "forward_returns",
    "from_dict",
    "from_json",
    "ic_ir",
    "is_subtype",
    "is_valid",
    "mean_ic",
    "score_tree",
    "simplify",
    "to_dict",
    "to_json",
    "validate",
]
