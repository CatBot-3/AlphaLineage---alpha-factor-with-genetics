"""P1-T5 - the vectorized evaluator (with an optional C++ backend).

``evaluate_python`` is the pure-Python recursive baseline (the correctness reference): operand
leaves return their panel field, ephemeral leaves their constant, operators apply their vectorized
implementation, and user macros expand. ``evaluate`` is the public dispatcher - it uses the C++
backend when it is built and the tree is supported (``ALPHALINEAGE_EVALUATOR=auto`` default), and
otherwise falls back transparently to ``evaluate_python``. The two are pinned identical by the
parity test, so the backend choice never changes a result.
"""

from __future__ import annotations

from alphalineage.core import cpp
from alphalineage.core.extensions import expand
from alphalineage.core.panel import Panel
from alphalineage.core.primitives import Kind
from alphalineage.core.tree import EvalResult, Node


def evaluate_python(node: Node, panel: Panel) -> EvalResult:
    """Pure-Python recursive evaluation (the correctness baseline)."""
    prim = node.primitive
    if prim.macro_body is not None:  # user operator: expand the macro, then evaluate
        return evaluate_python(expand(node, prim.macro_body), panel)
    if prim.kind is Kind.OPERAND:
        assert prim.panel_field is not None
        return panel[prim.panel_field]
    if prim.kind is Kind.EPHEMERAL:
        assert node.value is not None
        return node.value
    assert prim.fn is not None
    args = [evaluate_python(child, panel) for child in node.children]
    return prim.fn(*args)


def evaluate(node: Node, panel: Panel) -> EvalResult:
    """Evaluate ``node`` against ``panel``, using the C++ backend when available + supported."""
    if cpp.backend_enabled():
        try:
            result = cpp.evaluate_cpp(node, panel)
            if result is not None:
                return result
        except Exception:  # noqa: BLE001 - any backend failure falls back to Python
            pass
    return evaluate_python(node, panel)
