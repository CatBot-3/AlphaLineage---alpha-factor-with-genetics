"""P1-T3 — the Node data structure, type validation, and lossless JSON (de)serialization.

A tree is an immutable ``Node``. Internal nodes name an operator; leaf nodes name an
operand (a panel field) or an ephemeral constant (carrying its sampled ``value``).
Structural equality/hash come for free from the frozen dataclass, which makes
``from_json(to_json(t)) == t`` a meaningful round-trip check.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import pandas as pd

from alphaforge.core.primitives import REGISTRY, Kind, Primitive
from alphaforge.core.types import DType, is_subtype

#: What evaluating a node yields: a panel frame (SERIES/SIGNAL) or a leaf constant.
EvalResult = pd.DataFrame | float | int


@dataclass(frozen=True)
class Node:
    """A node in a typed expression tree."""

    name: str
    children: tuple[Node, ...] = ()
    value: float | int | None = None  # ephemeral leaves only (const -> float, window -> int)

    @property
    def primitive(self) -> Primitive:
        return REGISTRY[self.name]

    @property
    def out_type(self) -> DType:
        return self.primitive.out_type

    def depth(self) -> int:
        return 1 + max((c.depth() for c in self.children), default=0)

    def size(self) -> int:
        return 1 + sum(c.size() for c in self.children)

    def iter_nodes(self) -> Iterator[Node]:
        yield self
        for child in self.children:
            yield from child.iter_nodes()

    def __str__(self) -> str:
        prim = self.primitive
        if prim.kind is Kind.EPHEMERAL:
            return f"{self.value}"
        if prim.kind is Kind.OPERAND:
            return self.name
        return f"{self.name}({', '.join(str(c) for c in self.children)})"


class InvalidTree(ValueError):
    """Raised when a tree violates the primitive type signatures."""


def validate(node: Node) -> Node:
    """Assert ``node`` is structurally and type-consistent; return it unchanged."""
    prim = REGISTRY.get(node.name)
    if prim is None:
        raise InvalidTree(f"unknown primitive {node.name!r}")

    if prim.kind is Kind.EPHEMERAL:
        if node.children:
            raise InvalidTree(f"ephemeral {node.name!r} must have no children")
        if node.value is None:
            raise InvalidTree(f"ephemeral {node.name!r} must carry a value")
        return node
    if prim.kind is Kind.OPERAND:
        if node.children:
            raise InvalidTree(f"operand {node.name!r} must have no children")
        return node

    # operator
    if len(node.children) != prim.arity:
        raise InvalidTree(f"{node.name!r} expects {prim.arity} args, got {len(node.children)}")
    for child, expected in zip(node.children, prim.arg_types, strict=True):
        if not is_subtype(child.out_type, expected):
            raise InvalidTree(
                f"{node.name!r} arg expected {expected}, got {child.out_type} ({child.name!r})"
            )
        validate(child)
    return node


def is_valid(node: Node) -> bool:
    try:
        validate(node)
    except InvalidTree:
        return False
    return True


# --- serialization ---------------------------------------------------------------
def to_dict(node: Node) -> dict[str, Any]:
    out: dict[str, Any] = {"name": node.name}
    if node.value is not None:
        out["value"] = node.value
    if node.children:
        out["children"] = [to_dict(c) for c in node.children]
    return out


def from_dict(data: dict[str, Any]) -> Node:
    children = tuple(from_dict(c) for c in data.get("children", ()))
    return Node(name=data["name"], children=children, value=data.get("value"))


def to_json(node: Node) -> str:
    return json.dumps(to_dict(node), separators=(",", ":"), sort_keys=True)


def from_json(text: str) -> Node:
    return from_dict(json.loads(text))
