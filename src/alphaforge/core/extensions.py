"""P7 — user-defined operators as safe typed macros (invariant 5: no arbitrary server code).

A user operator is a *composition of existing primitives* with argument placeholders, submitted
as **data** (a typed tree), not code. ``register_operator`` type-checks the body against the
declared signature — rejecting anything that is not a known, vectorized primitive — and registers
it as a normal ``Primitive`` the GP can search. Evaluation expands the macro through the existing
evaluator. There is no ``eval``/``exec``/``compile`` anywhere on this path.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from alphaforge.core.primitives import OPERATORS, REGISTRY, Kind, Primitive
from alphaforge.core.tree import Node, from_dict
from alphaforge.core.types import DType, is_subtype

#: Placeholder leaf for the i-th macro argument: ``Node(ARG, value=i)``.
ARG = "$arg"

#: User-registered operators (name -> Primitive). Process-global (single-user local app).
USER_OPERATORS: dict[str, Primitive] = {}


class InvalidOperator(ValueError):
    """Raised when a user operator's body or signature is invalid."""


def infer_macro_type(body: Node, arg_types: Sequence[DType]) -> DType:
    """Type-check a macro body, treating ``$arg[i]`` as ``arg_types[i]``; return its output type.

    This is the security gate: a body may only reference ``$arg`` placeholders and **existing**
    primitives, composed with valid types. Anything else raises ``InvalidOperator``.
    """
    if body.name == ARG:
        index = int(body.value) if body.value is not None else -1
        if index < 0 or index >= len(arg_types):
            raise InvalidOperator(f"$arg index {index} out of range for {len(arg_types)} args")
        if body.children:
            raise InvalidOperator("$arg placeholder must have no children")
        return arg_types[index]

    prim = REGISTRY.get(body.name)
    if prim is None:
        raise InvalidOperator(f"unknown primitive {body.name!r} (only existing primitives allowed)")
    if prim.kind in (Kind.OPERAND, Kind.EPHEMERAL):
        if body.children:
            raise InvalidOperator(f"{body.name!r} is a leaf and must have no children")
        return prim.out_type

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
    body_node = body if isinstance(body, Node) else from_dict(body)
    inferred = infer_macro_type(body_node, types)
    if not is_subtype(inferred, out_type):
        raise InvalidOperator(f"body produces {inferred}, not the declared output {out_type}")

    prim = Primitive(name, Kind.OPERATOR, out_type, types, macro_body=body_node)
    OPERATORS[name] = prim
    REGISTRY[name] = prim
    USER_OPERATORS[name] = prim
    return prim


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
