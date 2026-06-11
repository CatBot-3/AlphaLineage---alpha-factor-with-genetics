"""Optional C++ evaluator backend: flatten a tree to an IR and dispatch to the extension.

The pure-Python evaluator is the correctness baseline. This module compiles a (macro-expanded)
tree into a flat instruction list the C++ extension can walk over the panel's stacked arrays, and
selects the backend (``ALPHAFORGE_EVALUATOR=auto|python|cpp``; default ``auto`` = C++ when the
extension is importable and the tree is fully supported, else Python). ``flatten`` is pure Python
and unit-testable without any compiler.
"""

from __future__ import annotations

import os
from weakref import WeakKeyDictionary

import numpy as np
import pandas as pd

from alphaforge.core.extensions import expand_all
from alphaforge.core.panel import Panel
from alphaforge.core.primitives import OPERAND_FIELDS
from alphaforge.core.tree import Node

# The compiled extension is optional; absence => Python fallback.
try:
    from alphaforge import _evaluator as _EXT  # type: ignore[attr-defined]
except ImportError:  # pragma: no cover - exercised only when unbuilt
    _EXT = None

# Opcodes — must match cpp/evaluator.cpp.
OP_LOAD = 0
_BINARY = {"add": 1, "sub": 2, "mul": 3, "div": 4}
_SCALAR = {"mul_scalar": 5, "add_scalar": 6, "signed_power": 7}
_UNARY = {"log": 8, "abs": 9, "sign": 10, "neg": 11}
_TS = {
    "ts_mean": 12,
    "ts_std": 13,
    "ts_sum": 14,
    "ts_min": 15,
    "ts_max": 16,
    "delta": 17,
    "delay": 18,
}
_CROSS = {"rank": 19, "zscore": 20}

#: Built-in operators the C++ backend can evaluate (everything else => Python fallback).
CPP_OPCODES: dict[str, int] = {**_BINARY, **_SCALAR, **_UNARY, **_TS, **_CROSS}
_FIELD_INDEX = {name: i for i, name in enumerate(OPERAND_FIELDS)}

# One instruction = (opcode, a, b, ival, fval, field).
Instruction = tuple[int, int, int, int, float, int]


def available() -> bool:
    """True if the compiled C++ evaluator extension is importable."""
    return _EXT is not None


def backend_enabled() -> bool:
    """True if the selected backend permits C++ and the extension is available."""
    return os.environ.get("ALPHAFORGE_EVALUATOR", "auto").lower() in ("auto", "cpp") and available()


def flatten(node: Node) -> tuple[list[Instruction], int] | None:
    """Compile a tree into a post-order instruction list, or ``None`` if any op is unsupported."""
    instrs: list[Instruction] = []

    def emit(
        op: int, a: int = -1, b: int = -1, ival: int = 0, fval: float = 0.0, field: int = -1
    ) -> int:
        instrs.append((op, a, b, ival, fval, field))
        return len(instrs) - 1

    def visit(n: Node) -> int | None:
        name = n.name
        if name in _FIELD_INDEX:
            return emit(OP_LOAD, field=_FIELD_INDEX[name])
        op = CPP_OPCODES.get(name)
        if op is None:
            return None  # unsupported op -> whole-tree fallback
        if name in _BINARY:
            a, b = visit(n.children[0]), visit(n.children[1])
            return None if a is None or b is None else emit(op, a=a, b=b)
        if name in _SCALAR:
            a = visit(n.children[0])
            return None if a is None else emit(op, a=a, fval=float(n.children[1].value or 0.0))
        if name in _TS:
            a = visit(n.children[0])
            return None if a is None else emit(op, a=a, ival=int(n.children[1].value or 0))
        # unary or cross-sectional: one series child
        a = visit(n.children[0])
        return None if a is None else emit(op, a=a)

    root = visit(expand_all(node))
    return None if root is None else (instrs, root)


_PANEL_ARRAYS: WeakKeyDictionary[Panel, np.ndarray] = WeakKeyDictionary()


def _panel_arrays(panel: Panel) -> np.ndarray:
    """Stacked ``(n_fields, T, N)`` float64 array of the panel's operand fields (cached)."""
    cached = _PANEL_ARRAYS.get(panel)
    if cached is not None:
        return cached
    stacked = np.stack(
        [np.ascontiguousarray(panel[f].to_numpy(dtype="float64")) for f in OPERAND_FIELDS]
    )
    _PANEL_ARRAYS[panel] = stacked
    return stacked


def evaluate_cpp(node: Node, panel: Panel) -> pd.DataFrame | None:
    """Evaluate via the C++ extension; ``None`` if unavailable or the tree is unsupported."""
    if _EXT is None:
        return None
    plan = flatten(node)
    if plan is None:
        return None
    instrs, root = plan
    ops = np.array([i[0] for i in instrs], dtype=np.int32)
    a = np.array([i[1] for i in instrs], dtype=np.int32)
    b = np.array([i[2] for i in instrs], dtype=np.int32)
    ival = np.array([i[3] for i in instrs], dtype=np.int32)
    fval = np.array([i[4] for i in instrs], dtype=np.float64)
    field = np.array([i[5] for i in instrs], dtype=np.int32)
    result = _EXT.evaluate(_panel_arrays(panel), ops, a, b, ival, fval, field, root)
    return pd.DataFrame(result, index=panel.dates, columns=panel.symbols)
