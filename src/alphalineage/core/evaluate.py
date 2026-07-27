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
    memo: dict[Node, EvalResult] = {}

    def visit(current: Node) -> EvalResult:
        cached = memo.get(current)
        if cached is not None:
            return cached
        prim = current.primitive
        if prim.macro_body is not None:  # formula: expand once and memoize its result
            result = visit(expand(current, prim.macro_body))
        elif prim.kind is Kind.OPERAND:
            assert prim.panel_field is not None
            result = panel[prim.panel_field]
        elif prim.kind is Kind.EPHEMERAL:
            assert current.value is not None
            result = current.value
        else:
            assert prim.fn is not None
            result = prim.fn(*(visit(child) for child in current.children))
        memo[current] = result
        return result

    return visit(node)


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
