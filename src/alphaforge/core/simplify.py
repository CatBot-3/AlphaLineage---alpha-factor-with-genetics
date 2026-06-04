"""P1-T6 — algebraic simplification / anti-bloat.

Collapses semantics-preserving no-ops bottom-up to a fixpoint: identity scalar/window
ops, idempotent unaries, and double negation. Used to shrink trees before display and to
support the Phase 2 node-count (anti-bloat) penalty. Every rule preserves the evaluated
output exactly.
"""

from __future__ import annotations

from alphaforge.core.tree import Node

# Time-series ops where a window of 1 is the identity (f(x, 1) == x).
_TS_WINDOW1_IDENTITY = frozenset({"ts_mean", "ts_sum", "ts_min", "ts_max", "decay_linear"})


def _is_const(node: Node, value: float) -> bool:
    return node.name == "const" and node.value is not None and float(node.value) == value


def _is_window(node: Node, value: int) -> bool:
    return node.name == "window" and node.value is not None and int(node.value) == value


def _rewrite(node: Node) -> Node:
    """Apply one layer of rewrites to a node whose children are already simplified."""
    name = node.name
    ch = node.children

    # scalar identities
    if name == "add_scalar" and _is_const(ch[1], 0.0):
        return ch[0]
    if name == "mul_scalar" and _is_const(ch[1], 1.0):
        return ch[0]
    if name == "signed_power" and _is_const(ch[1], 1.0):
        return ch[0]

    # window identities
    if name == "delay" and _is_window(ch[1], 0):
        return ch[0]
    if name in _TS_WINDOW1_IDENTITY and _is_window(ch[1], 1):
        return ch[0]

    # idempotent / inverse unaries
    if name == "abs" and ch[0].name in ("abs", "neg"):
        return Node("abs", ch[0].children)
    if name == "sign" and ch[0].name == "sign":
        return ch[0]
    if name == "neg" and ch[0].name == "neg":
        return ch[0].children[0]

    return node


def simplify(node: Node) -> Node:
    """Return a semantically-equivalent, reduced copy of ``node``."""
    if node.children:
        node = Node(node.name, tuple(simplify(c) for c in node.children), node.value)
    rewritten = _rewrite(node)
    if rewritten != node:
        return simplify(rewritten)
    return node
