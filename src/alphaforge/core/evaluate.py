"""P1-T5 — the vectorized evaluator.

Walks a tree bottom-up over a :class:`~alphaforge.core.panel.Panel`. Operand leaves
return their panel field, ephemeral leaves return their stored constant, and operators
apply their vectorized implementation to the evaluated children. A SERIES/SIGNAL node
evaluates to a wide ``date x symbol`` DataFrame; SCALAR/WINDOW nodes to a float/int.
"""

from __future__ import annotations

from alphaforge.core.panel import Panel
from alphaforge.core.primitives import Kind
from alphaforge.core.tree import EvalResult, Node


def evaluate(node: Node, panel: Panel) -> EvalResult:
    """Evaluate ``node`` against ``panel``."""
    prim = node.primitive
    if prim.kind is Kind.OPERAND:
        assert prim.panel_field is not None
        return panel[prim.panel_field]
    if prim.kind is Kind.EPHEMERAL:
        assert node.value is not None
        return node.value
    assert prim.fn is not None
    args = [evaluate(child, panel) for child in node.children]
    return prim.fn(*args)
