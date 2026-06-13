"""P1-T4 - strongly-typed random tree generation (ramped half-and-half).

Fills a typed hole recursively, choosing only primitives whose output can satisfy the
hole and that fit the remaining depth/node budget. Both the depth limit and the node-count
limit are enforced *by construction* (no rejection sampling): when an operator is chosen,
each argument is given a budget that reserves at least one node for every sibling still to
be built, so the total never exceeds the cap and every hole is always closeable.
"""

from __future__ import annotations

import random

from alphalineage.core.primitives import Primitive, operators_for, terminals_for
from alphalineage.core.tree import Node
from alphalineage.core.types import DType

DEFAULT_MAX_DEPTH = 5
DEFAULT_MAX_NODES = 40


class RandomTreeGenerator:
    """Generates type-valid random trees with bounded depth and size."""

    def __init__(
        self,
        rng: random.Random | None = None,
        *,
        max_depth: int = DEFAULT_MAX_DEPTH,
        max_nodes: int = DEFAULT_MAX_NODES,
        root_type: DType = DType.SIGNAL,
    ) -> None:
        self.rng = rng or random.Random()
        self.max_depth = max_depth
        self.max_nodes = max_nodes
        self.root_type = root_type

    def _terminal_node(self, prim: Primitive) -> Node:
        if prim.sampler is not None:  # ephemeral constant
            return Node(prim.name, value=prim.sampler(self.rng))
        return Node(prim.name)  # operand

    def _build(self, target: DType, depth_rem: int, budget: int, grow: bool) -> tuple[Node, int]:
        terminals = terminals_for(target)
        # An operator needs 1 node for itself plus >=1 per argument.
        operators = [op for op in operators_for(target) if op.arity + 1 <= budget]
        can_branch = depth_rem > 1 and bool(operators)

        if can_branch:
            choices = operators + terminals if grow else operators
            prim = self.rng.choice(choices)
        else:
            prim = self.rng.choice(terminals)

        if prim.is_terminal:
            return self._terminal_node(prim), 1

        used = 1
        remaining = budget - 1
        children: list[Node] = []
        n_args = prim.arity
        for i, arg_type in enumerate(prim.arg_types):
            reserve = n_args - i - 1  # one node minimum for each remaining sibling
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
