"""P1-T4 - strongly-typed random tree generation (ramped half-and-half).

Fills a typed hole recursively, choosing only primitives whose output can satisfy the
hole and that fit the remaining depth/node budget. Both the depth limit and the node-count
limit are enforced *by construction* (no rejection sampling): an operator is offered only when
its cheapest completion fits the remaining budget, and each argument reserves at least the
minimum nodes every later sibling needs, so the total never exceeds the cap and every hole is
always closeable.

Operator availability is gated by ``allowed_operators`` (a name allow-set): with ``None`` the
default pool excludes the ``condition`` category, so the classic numeric search space (and the
two load-bearing tests) is unchanged until conditions are explicitly opted in for a run.
"""

from __future__ import annotations

import random

from alphalineage.core.categories import DEFAULT_ENABLED_CATEGORIES, builtin_category
from alphalineage.core.primitives import OPERATORS, Primitive, operators_for, terminals_for
from alphalineage.core.tree import Node
from alphalineage.core.types import DType, is_subtype

DEFAULT_MAX_DEPTH = 5
DEFAULT_MAX_NODES = 40

_INF = 1 << 30


class GenerationError(RuntimeError):
    """Raised when a typed hole cannot be closed within the given budget (caught by callers)."""


def operator_allowed(prim: Primitive, allowed_operators: set[str] | None) -> bool:
    """Whether an operator may be used. ``None`` => default pool (condition category excluded)."""
    if allowed_operators is not None:
        return prim.name in allowed_operators
    if prim.macro_body is not None:  # user formulas default to the (enabled) 'custom' category
        return True
    return builtin_category(prim.name) in DEFAULT_ENABLED_CATEGORIES


def _min_costs(allowed_operators: set[str] | None) -> tuple[dict[DType, int], dict[DType, int]]:
    """Per-type minimum (nodes, depth) to close a hole, given the allowed operator pool.

    Fixpoint over the current registry: a type with a terminal costs (1, 1); otherwise the
    cheapest allowed operator whose output fills it. Types with no terminal and no allowed
    producer stay at ``_INF`` (e.g. BOOL when conditions are disabled) and are simply never
    offered as a fillable hole.
    """
    allowed = [op for op in OPERATORS.values() if operator_allowed(op, allowed_operators)]
    nodes = {t: (1 if terminals_for(t) else _INF) for t in DType}
    depth = {t: (1 if terminals_for(t) else _INF) for t in DType}
    changed = True
    while changed:
        changed = False
        for op in allowed:
            arg_nodes = [nodes[a] for a in op.arg_types]
            arg_depth = [depth[a] for a in op.arg_types]
            if any(n >= _INF for n in arg_nodes):
                continue
            cand_n = 1 + sum(arg_nodes)
            cand_d = 1 + (max(arg_depth) if arg_depth else 0)
            for t in DType:
                if not is_subtype(op.out_type, t):
                    continue
                if cand_n < nodes[t]:
                    nodes[t] = cand_n
                    changed = True
                if cand_d < depth[t]:
                    depth[t] = cand_d
                    changed = True
    return nodes, depth


class RandomTreeGenerator:
    """Generates type-valid random trees with bounded depth and size."""

    def __init__(
        self,
        rng: random.Random | None = None,
        *,
        max_depth: int = DEFAULT_MAX_DEPTH,
        max_nodes: int = DEFAULT_MAX_NODES,
        root_type: DType = DType.SIGNAL,
        allowed_operators: set[str] | None = None,
    ) -> None:
        self.rng = rng or random.Random()
        self.max_depth = max_depth
        self.max_nodes = max_nodes
        self.root_type = root_type
        self.allowed_operators = allowed_operators
        self._min_nodes, self._min_depth = _min_costs(allowed_operators)

    def _terminal_node(self, prim: Primitive) -> Node:
        if prim.sampler is not None:  # ephemeral constant
            return Node(prim.name, value=prim.sampler(self.rng))
        return Node(prim.name)  # operand

    def _fits(self, op: Primitive, budget: int, depth_rem: int) -> bool:
        need_nodes = 1 + sum(self._min_nodes[a] for a in op.arg_types)
        need_depth = 1 + max((self._min_depth[a] for a in op.arg_types), default=0)
        return need_nodes <= budget and need_depth <= depth_rem

    def _build(self, target: DType, depth_rem: int, budget: int, grow: bool) -> tuple[Node, int]:
        terminals = terminals_for(target)
        operators = [
            op
            for op in operators_for(target)
            if operator_allowed(op, self.allowed_operators) and self._fits(op, budget, depth_rem)
        ]

        if operators:
            choices = operators + terminals if grow else operators
            prim = self.rng.choice(choices)
        elif terminals:
            prim = self.rng.choice(terminals)
        else:
            raise GenerationError(
                f"cannot close a {target.value} hole within budget {budget}/depth {depth_rem}"
            )

        if prim.is_terminal:
            return self._terminal_node(prim), 1

        used = 1
        remaining = budget - 1
        children: list[Node] = []
        n_args = prim.arity
        for i, arg_type in enumerate(prim.arg_types):
            reserve = sum(self._min_nodes[prim.arg_types[j]] for j in range(i + 1, n_args))
            child, used_c = self._build(arg_type, depth_rem - 1, remaining - reserve, grow)
            children.append(child)
            used += used_c
            remaining -= used_c
        return Node(prim.name, tuple(children)), used

    def generate(
        self, *, grow: bool = True, max_depth: int | None = None, root_type: DType | None = None
    ) -> Node:
        node, _ = self._build(
            root_type or self.root_type, max_depth or self.max_depth, self.max_nodes, grow
        )
        return node

    def grow_subtree(
        self, target: DType, *, max_depth: int, max_nodes: int, grow: bool = True
    ) -> Node:
        """Generate a fresh, type-valid subtree of type ``target`` within the given budgets.

        Public entry point for crossover/mutation in :mod:`alphalineage.core.gp`.
        """
        node, _ = self._build(target, max(1, max_depth), max(1, max_nodes), grow)
        return node

    def ramped_half_and_half(
        self, n: int, *, min_depth: int = 2, max_depth: int | None = None
    ) -> list[Node]:
        """A population of ``n`` trees, ramped over depth limits, half grow / half full."""
        top = max_depth or self.max_depth
        depths = list(range(min_depth, top + 1)) or [top]
        return [
            self.generate(grow=(i % 2 == 0), max_depth=depths[i % len(depths)]) for i in range(n)
        ]
