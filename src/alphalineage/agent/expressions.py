"""P10-T3 - parsing the DSL's function-call text form into a typed tree.

The frontend has had this since Phase 7 (``frontend/src/extend/formulaText.ts``); the backend
never needed it, because the API has always taken trees as JSON. An agent does need it: asking a
language model to emit a nested JSON tree is a reliable way to get malformed output, whereas
``rank(ts_std(returns, 20))`` is the notation it has actually seen. This module is the inverse of
:func:`alphalineage.explain.anatomy.formula_text`, and the two must stay in step.

Numbers are typed by their **parent slot**, exactly as the editor does it: a number in a WINDOW
slot becomes ``{"name": "window", "value": 20}``, a number in a SCALAR slot becomes
``{"name": "const", "value": 0.5}``. That is what lets the model write plain numerals.

Parsing produces a *candidate* tree; it does not trust it. Every result is handed to
``core.tree.validate``, the same boundary a human's formula-editor input crosses (invariant 5).
Nothing here executes, registers, or evaluates anything.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from alphalineage.core.primitives import REGISTRY, Kind
from alphalineage.core.tree import InvalidTree, Node, validate
from alphalineage.core.types import DType

_TOKEN_RE = re.compile(r"\s+|(-?\d+\.\d+|-?\d+)|([A-Za-z_][A-Za-z0-9_]*)|(\()|(\))|(,)")
MAX_EXPRESSION_CHARS = 4000
MAX_TOKENS = 800


class ExpressionError(ValueError):
    """A parse or type error, carrying the character offset so the model can self-correct."""

    def __init__(self, message: str, position: int = 0) -> None:
        super().__init__(message)
        self.message = message
        self.position = position

    def to_dict(self) -> dict[str, Any]:
        return {"error": self.message, "position": self.position}


@dataclass(frozen=True)
class _Token:
    kind: str  # number | ident | lparen | rparen | comma
    text: str
    pos: int


def _tokenize(text: str) -> list[_Token]:
    tokens: list[_Token] = []
    i = 0
    while i < len(text):
        match = _TOKEN_RE.match(text, i)
        if match is None or match.start() != i:
            raise ExpressionError(f"unexpected character {text[i]!r}", i)
        number, ident, lparen, rparen, comma = match.groups()
        start, i = match.start(), match.end()
        if number is not None:
            tokens.append(_Token("number", number, start))
        elif ident is not None:
            tokens.append(_Token("ident", ident, start))
        elif lparen is not None:
            tokens.append(_Token("lparen", lparen, start))
        elif rparen is not None:
            tokens.append(_Token("rparen", rparen, start))
        elif comma is not None:
            tokens.append(_Token("comma", comma, start))
        if len(tokens) > MAX_TOKENS:
            raise ExpressionError(f"expression exceeds {MAX_TOKENS} tokens", start)
    return tokens


def _numeric_leaf(token: _Token, expected: DType | None) -> Node:
    """Type a bare numeral by the slot it fills, mirroring the formula editor."""
    if expected is DType.WINDOW:
        if not re.fullmatch(r"-?\d+", token.text):
            raise ExpressionError(
                f"{token.text!r} fills a window slot, which must be a whole number", token.pos
            )
        return Node("window", value=int(token.text))
    if expected is DType.SCALAR:
        return Node("const", value=float(token.text))
    if expected is None:
        raise ExpressionError(
            "a bare number cannot be the whole expression; a factor must produce a series",
            token.pos,
        )
    raise ExpressionError(
        f"{token.text!r} fills a slot expecting {expected.value}, not a number", token.pos
    )


def parse_expression(text: str, *, allowed: set[str] | None = None) -> Node:
    """Parse the function-call form into a validated, type-checked tree.

    ``allowed`` restricts the callable names to the run's enabled operator set, so the agent
    cannot reach an operator the session's search space excludes.
    """
    if not isinstance(text, str) or not text.strip():
        raise ExpressionError("expression is empty", 0)
    if len(text) > MAX_EXPRESSION_CHARS:
        raise ExpressionError(f"expression exceeds {MAX_EXPRESSION_CHARS} characters", 0)

    tokens = _tokenize(text)
    if not tokens:
        raise ExpressionError("expression is empty", 0)
    position = 0

    def peek() -> _Token | None:
        return tokens[position] if position < len(tokens) else None

    def take() -> _Token:
        nonlocal position
        token = peek()
        if token is None:
            raise ExpressionError("expression ended unexpectedly", len(text))
        position += 1
        return token

    def parse(expected: DType | None, depth: int = 0) -> Node:
        if depth > 64:
            ahead = peek()
            raise ExpressionError(
                "expression is nested too deeply", ahead.pos if ahead is not None else 0
            )
        token = take()
        if token.kind == "number":
            return _numeric_leaf(token, expected)
        if token.kind != "ident":
            raise ExpressionError(f"expected an operator or field, found {token.text!r}", token.pos)

        name = token.text
        primitive = REGISTRY.get(name)
        if primitive is None:
            raise ExpressionError(
                f"unknown operator {name!r}. Use list_operators to see the legal vocabulary; "
                "operators cannot be invented.",
                token.pos,
            )
        if allowed is not None and name not in allowed:
            raise ExpressionError(
                f"{name!r} exists but is not enabled for this session's search space", token.pos
            )

        nxt = peek()
        if nxt is None or nxt.kind != "lparen":
            if primitive.kind is not Kind.OPERAND:
                raise ExpressionError(
                    f"{name!r} takes {primitive.arity} argument(s); write {name}(...)", token.pos
                )
            return Node(name)

        take()  # consume "("
        if primitive.kind is Kind.OPERAND:
            raise ExpressionError(f"{name!r} is a data field and takes no arguments", token.pos)

        children: list[Node] = []
        closing = peek()
        if closing is not None and closing.kind == "rparen":
            take()
        else:
            while True:
                index = len(children)
                slot = primitive.arg_types[index] if index < len(primitive.arg_types) else None
                if slot is None:
                    raise ExpressionError(
                        f"{name!r} takes {primitive.arity} argument(s), got more", token.pos
                    )
                children.append(parse(slot, depth + 1))
                separator = take()
                if separator.kind == "rparen":
                    break
                if separator.kind != "comma":
                    raise ExpressionError(
                        f"expected ',' or ')' in {name!r}, found {separator.text!r}",
                        separator.pos,
                    )
        if len(children) != primitive.arity:
            raise ExpressionError(
                f"{name!r} takes {primitive.arity} argument(s), got {len(children)}", token.pos
            )
        return Node(name, tuple(children))

    tree = parse(None)
    trailing = peek()
    if trailing is not None:
        raise ExpressionError(f"unexpected trailing input {trailing.text!r}", trailing.pos)

    try:
        validate(tree)
    except InvalidTree as exc:
        raise ExpressionError(str(exc), 0) from None
    if tree.out_type not in (DType.SERIES, DType.SIGNAL):
        raise ExpressionError(
            f"a factor must produce a series, but this expression produces {tree.out_type.value}",
            0,
        )
    return tree
