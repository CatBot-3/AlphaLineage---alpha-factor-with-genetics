"""P7 - user-defined operators as safe typed macros (invariant 5: no arbitrary server code).

A user operator is a *composition of existing primitives* with argument placeholders, submitted
as **data** (a typed tree), not code. ``register_operator`` type-checks the body against the
declared signature - rejecting anything that is not a known, vectorized primitive - and registers
it as a normal ``Primitive`` the GP can search. Evaluation expands the macro through the existing
evaluator. There is no ``eval``/``exec``/``compile`` anywhere on this path.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from alphalineage.core.primitives import OPERATORS, REGISTRY, Kind, Primitive
from alphalineage.core.tree import InvalidTree, Node, from_dict, validate
from alphalineage.core.types import DType, is_subtype

#: Placeholder leaf for the i-th macro argument: ``Node(ARG, value=i)``.
ARG = "$arg"

#: User-registered operators (name -> Primitive). Process-global (single-user local app).
USER_OPERATORS: dict[str, Primitive] = {}

# User formulas are data, but a malformed or manually edited formula store can still describe
# a recursive or exponentially expanding macro graph.  Keep every expansion entry point bounded
# so preview, backtest, training, and result persistence share the same safety envelope.
# Keep these equal to the GP hard ceilings without importing ``core.gp`` (which imports the
# evaluator and would introduce a cycle).  Compact macro calls therefore cannot bypass the
# global expression limits used by training seeds and native compilation.
MAX_EXPANDED_DEPTH = 32
MAX_EXPANDED_NODES = 2_000


class InvalidOperator(ValueError):
    """Raised when a user operator's body or signature is invalid."""


def _body_node(body: Node | dict[str, Any]) -> Node:
    if isinstance(body, Node):
        return body
    try:
        return from_dict(body)
    except (AttributeError, KeyError, TypeError) as exc:
        raise InvalidOperator("operator body must be a well-formed expression tree") from exc


def infer_macro_type(body: Node, arg_types: Sequence[DType]) -> DType:
    """Type-check a macro body, treating ``$arg[i]`` as ``arg_types[i]``; return its output type.

    This is the security gate: a body may only reference ``$arg`` placeholders and **existing**
    primitives, composed with valid types. Anything else raises ``InvalidOperator``.
    """
    if body.name == ARG:
        if isinstance(body.value, bool) or not isinstance(body.value, int):
            raise InvalidOperator(f"$arg index must be an integer, got {body.value!r}")
        index = body.value
        if index < 0 or index >= len(arg_types):
            raise InvalidOperator(f"$arg index {index} out of range for {len(arg_types)} args")
        if body.children:
            raise InvalidOperator("$arg placeholder must have no children")
        return arg_types[index]

    prim = REGISTRY.get(body.name)
    if prim is None:
        raise InvalidOperator(f"unknown primitive {body.name!r} (only existing primitives allowed)")
    if prim.kind in (Kind.OPERAND, Kind.EPHEMERAL):
        try:
            validate(body)
        except InvalidTree as exc:
            raise InvalidOperator(str(exc)) from exc
        return prim.out_type

    if body.value is not None:
        raise InvalidOperator(f"operator {body.name!r} must not carry a value")
    if len(body.children) != prim.arity:
        raise InvalidOperator(f"{body.name!r} expects {prim.arity} args, got {len(body.children)}")
    for child, expected in zip(body.children, prim.arg_types, strict=True):
        actual = infer_macro_type(child, arg_types)
        if not is_subtype(actual, expected):
            raise InvalidOperator(f"{body.name!r} arg expected {expected}, got {actual}")
    return prim.out_type


def register_operator(
    name: str,
    arg_types: Sequence[DType],
    out_type: DType,
    body: Node | dict[str, Any],
) -> Primitive:
    """Validate and register a user operator (a typed macro). Returns the new primitive."""
    if not isinstance(name, str) or not name or name == ARG:
        raise InvalidOperator(f"invalid operator name {name!r}")
    if name in REGISTRY:
        raise InvalidOperator(f"{name!r} already exists; choose another name")

    types = tuple(arg_types)
    if any(not isinstance(arg_type, DType) for arg_type in types):
        raise InvalidOperator("arg_types must contain only DType values")
    if not isinstance(out_type, DType):
        raise InvalidOperator("out_type must be a DType value")
    body_node = _body_node(body)
    inferred = infer_macro_type(body_node, types)
    if not is_subtype(inferred, out_type):
        raise InvalidOperator(f"body produces {inferred}, not the declared output {out_type}")

    prim = Primitive(name, Kind.OPERATOR, out_type, types, macro_body=body_node)
    OPERATORS[name] = prim
    REGISTRY[name] = prim
    USER_OPERATORS[name] = prim
    return prim


def ensure_operator(
    name: str,
    arg_types: Sequence[DType],
    out_type: DType,
    body: Node | dict[str, Any],
) -> Primitive:
    """Idempotently register a user operator.

    No-op (returns the existing primitive) when an identical operator is already
    registered; raises ``InvalidOperator`` when the name is taken by a different
    definition or a built-in. Used to re-materialize a saved factor's operators
    before seeding a session (invariant 5: still data, never code).
    """
    types = tuple(arg_types)
    if any(not isinstance(arg_type, DType) for arg_type in types):
        raise InvalidOperator("arg_types must contain only DType values")
    if not isinstance(out_type, DType):
        raise InvalidOperator("out_type must be a DType value")
    body_node = _body_node(body)
    existing = USER_OPERATORS.get(name)
    if existing is not None:
        if (
            existing.arg_types == types
            and existing.out_type == out_type
            and existing.macro_body == body_node
        ):
            return existing
        raise InvalidOperator(f"{name!r} already registered with a different definition")
    if name in REGISTRY:  # a built-in (or other non-macro primitive) holds this name
        raise InvalidOperator(f"{name!r} already exists; choose another name")
    return register_operator(name, types, out_type, body_node)


def unregister_operator(name: str) -> None:
    """Remove a user operator (never a built-in)."""
    USER_OPERATORS.pop(name, None)
    for table in (OPERATORS, REGISTRY):
        prim = table.get(name)
        if prim is not None and prim.macro_body is not None:
            del table[name]


def clear_user_operators() -> None:
    for name in list(USER_OPERATORS):
        unregister_operator(name)


def operator_counts() -> tuple[int, int]:
    """Return ``(builtin_operator_count, user_operator_count)``."""
    user = len(USER_OPERATORS)
    return len(OPERATORS) - user, user


def expand(node: Node, body: Node) -> Node:
    """Hygienically substitute ``node``'s children into ``body``'s ``$arg`` placeholders."""

    def substitute(b: Node) -> Node:
        if b.name == ARG:
            return node.children[int(b.value) if b.value is not None else 0]
        if b.children:
            return Node(b.name, tuple(substitute(c) for c in b.children), b.value)
        return b

    return substitute(body)


def expand_all(
    node: Node,
    *,
    max_depth: int = MAX_EXPANDED_DEPTH,
    max_nodes: int = MAX_EXPANDED_NODES,
) -> Node:
    """Expand all user macros, rejecting cycles and unreasonably large expanded trees.

    The limits apply to the final built-in tree rather than the compact call-site tree.  This
    prevents nested formulas from bypassing GP complexity limits and, more importantly, makes a
    corrupt persisted cycle a visible validation error instead of an unbounded recursion.
    """
    if max_depth < 1 or max_nodes < 1:
        raise InvalidOperator("expansion limits must be positive")
    emitted = 0
    expansions = 0

    def visit(
        current: Node,
        depth: int,
        active: tuple[str, ...],
        *,
        account: bool = True,
    ) -> Node:
        nonlocal emitted, expansions
        if account and depth > max_depth:
            raise InvalidOperator(
                f"expanded expression depth exceeds the limit of {max_depth}"
            )
        prim = REGISTRY.get(current.name)
        if prim is not None and prim.macro_body is not None:
            if current.name in active:
                start = active.index(current.name)
                cycle = (*active[start:], current.name)
                raise InvalidOperator(f"formula dependency cycle: {' -> '.join(cycle)}")
            expansions += 1
            if expansions > max_nodes:
                raise InvalidOperator(
                    f"expanded expression size exceeds the limit of {max_nodes} nodes"
                )
            # Resolve call-site arguments in the caller's dependency scope. This permits normal
            # composition such as ``EMA(DIF(...))`` (and even ``SMA(SMA(...))``) without falsely
            # treating the inner finite call as a recursive formula definition.
            expanded_children = tuple(
                visit(child, depth, active, account=False) for child in current.children
            )
            call = Node(current.name, expanded_children, current.value)
            return visit(
                expand(call, prim.macro_body),
                depth,
                (*active, current.name),
                account=account,
            )

        if account:
            emitted += 1
            if emitted > max_nodes:
                raise InvalidOperator(
                    f"expanded expression size exceeds the limit of {max_nodes} nodes"
                )
        children = tuple(
            visit(child, depth + 1, active, account=account) for child in current.children
        )
        return Node(current.name, children, current.value)

    return visit(node, 1, ())
