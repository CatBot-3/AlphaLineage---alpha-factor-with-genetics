"""Optional C++ evaluator backend: flatten a tree to an IR and dispatch to the extension.

The pure-Python evaluator is the correctness baseline. This module compiles a (macro-expanded)
tree into a flat instruction list the C++ extension can walk over the panel's stacked arrays, and
selects the backend (``ALPHALINEAGE_EVALUATOR=auto|python|cpp``; default ``auto`` = C++ when the
extension is importable and the tree is fully supported, else Python). ``flatten`` is pure Python
and unit-testable without any compiler.
"""

from __future__ import annotations

import os
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from functools import lru_cache
from weakref import WeakKeyDictionary

import numpy as np
import pandas as pd

from alphalineage.core.extensions import expand_all_cached
from alphalineage.core.panel import Panel
from alphalineage.core.primitives import OPERAND_FIELDS, checked_scalar, checked_window
from alphalineage.core.tree import Node
from alphalineage.core.types import DType

# The compiled extension is optional; absence => Python fallback.
try:
    from alphalineage import _evaluator as _EXT  # type: ignore[attr-defined]
except ImportError:  # pragma: no cover - exercised only when unbuilt
    _EXT = None

_NATIVE_ABI_VERSIONS = {8, 9, 10}
_NATIVE_SCORING_ABI = 8
#: Shared-program batch scoring: one instruction list per worker with each distinct operation
#: emitted once. Older builds keep the per-tree path, which is slower but identical.
_NATIVE_SHARED_SCORING_ABI = 9
_MAX_PLAN_CACHE = 4096
# pandas releases differ at span=3 after a gap. Probe the Python baseline once,
# then pass that numerical convention into each native EMA instruction.
_EMA_DECAYED_NEW_WEIGHT = bool(
    np.isclose(pd.Series([1., 2., 3., np.nan, 5.]).ewm(span=3, adjust=False).mean().iloc[-1], 4.3125)
)

# Opcodes - must match cpp/evaluator.cpp.
OP_LOAD = 0
_BINARY = {"add": 1, "sub": 2, "mul": 3, "div": 4}
_SCALAR = {"mul_scalar": 5, "add_scalar": 6, "signed_power": 7}
_UNARY = {"log": 8, "abs": 9, "sign": 10, "neg": 11, "ts_cumsum": 38}
_TS = {
    "ts_mean": 12,
    "ts_std": 13,
    "ts_sum": 14,
    "ts_min": 15,
    "ts_max": 16,
    "delta": 17,
    "delay": 18,
    "ts_ema": 21,
    "ts_rank": 22,
    "decay_linear": 23,
    "ts_rma": 35,
    "ts_std_pop": 37,
}
_TS_INITIAL = {"ts_recursive_smooth": 36}
_BINARY_TS = {"ts_corr": 24, "ts_cov": 25}
_CROSS = {"rank": 19, "zscore": 20, "scale": 26}
_COMPARISON = {"gt": 27, "lt": 28, "ge": 29, "le": 30}
_LOGICAL_BINARY = {"and_": 31, "or_": 32}
_LOGICAL_UNARY = {"not_": 33}
_TERNARY = {"where": 34}

#: Built-in operators the C++ backend can evaluate (everything else => Python fallback).
CPP_OPCODES: dict[str, int] = {
    **_BINARY,
    **_SCALAR,
    **_UNARY,
    **_TS,
    **_TS_INITIAL,
    **_BINARY_TS,
    **_CROSS,
    **_COMPARISON,
    **_LOGICAL_BINARY,
    **_LOGICAL_UNARY,
    **_TERNARY,
}
_FIELD_INDEX = {name: i for i, name in enumerate(OPERAND_FIELDS)}

# One instruction = (opcode, a, b, ival, fval, field).
Instruction = tuple[int, int, int, int, float, int]
PlanArrays = tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray
]


@dataclass
class _CompiledPlan:
    instructions: tuple[Instruction, ...]
    root: int
    arrays: PlanArrays
    peak_buffers: int

    def native_tuple(self) -> tuple[object, ...]:
        return (*self.arrays, self.root)


def _plan_supported_by_loaded_abi(plan: _CompiledPlan) -> bool:
    """Reject plans whose operators need semantics absent from an older extension."""
    abi = int(getattr(_EXT, "ABI_VERSION", 0)) if _EXT is not None else 0
    # Older builds hard-coded the pandas 3 EMA gap rule. On pandas 2 those trees
    # must use Python until the extension is rebuilt with the explicit rule flag.
    if abi < 10 and not _EMA_DECAYED_NEW_WEIGHT:
        if any(instruction[0] == _TS["ts_ema"] for instruction in plan.instructions):
            return False
    return abi >= 5 or all(instruction[0] != 38 for instruction in plan.instructions)


def available() -> bool:
    """True when a compatible compiled evaluator extension is importable."""
    return (
        _EXT is not None
        and getattr(_EXT, "ABI_VERSION", None) in _NATIVE_ABI_VERSIONS
        and hasattr(_EXT, "evaluate_many")
        and hasattr(_EXT, "score_many")
    )


def unavailable_reason() -> str | None:
    """Return a stable diagnostic for capability APIs, or ``None`` when native is ready."""
    if _EXT is None:
        return "native evaluator extension is not installed"
    abi = getattr(_EXT, "ABI_VERSION", None)
    if abi not in _NATIVE_ABI_VERSIONS:
        expected = ", ".join(map(str, sorted(_NATIVE_ABI_VERSIONS)))
        return f"native evaluator ABI {abi!r} is incompatible (expected one of {expected})"
    if not hasattr(_EXT, "evaluate_many"):
        return "native evaluator does not provide ordered batch evaluation"
    if not hasattr(_EXT, "score_many"):
        return "native evaluator does not provide integrated batch scoring"
    return None


def native_max_workers() -> int:
    """Maximum worker count accepted by this native build (one when unavailable)."""
    if not available():
        return 1
    return max(1, int(getattr(_EXT, "MAX_WORKERS", 1)))


def native_abi_version() -> int | None:
    """Loaded evaluator ABI, included in the persisted scorer identity."""
    if not available():
        return None
    return int(_EXT.ABI_VERSION)


def supports_shared_scoring() -> bool:
    """Whether this build can score a whole batch from one shared instruction list."""
    abi = native_abi_version()
    return (
        abi is not None
        and abi >= _NATIVE_SHARED_SCORING_ABI
        and hasattr(_EXT, "score_shared")
    )


def supports_native_scoring(method: str) -> bool:
    """Whether the selected backend can preserve scoring semantics for this IC method."""
    abi = native_abi_version()
    return (
        method == "spearman"
        and backend_enabled()
        and abi is not None
        and abi >= _NATIVE_SCORING_ABI
    )


# Process-level backend override, set from the persisted UI setting. Resolved without a
# per-evaluation file read. The ``ALPHALINEAGE_EVALUATOR`` env var stays the explicit force-override
# (power users / CI); the UI setting governs whenever that env var is unset, which is every normal
# launch. ``None`` => no override.
_BACKEND_OVERRIDE: str | None = None


def set_backend(choice: str | None) -> None:
    """Set (clear with ``None``) the runtime evaluator choice: ``auto`` | ``python`` | ``cpp``."""
    global _BACKEND_OVERRIDE
    _BACKEND_OVERRIDE = choice.lower() if choice else None


def selected_backend() -> str:
    """The active evaluator selection: env var (force) > UI override > ``auto`` default."""
    env = os.environ.get("ALPHALINEAGE_EVALUATOR")
    if env:
        return env.lower()
    if _BACKEND_OVERRIDE is not None:
        return _BACKEND_OVERRIDE
    return "auto"


def backend_enabled() -> bool:
    """True if the selected backend permits C++ and the extension is available."""
    return selected_backend() in ("auto", "cpp") and available()


def flatten(node: Node) -> tuple[list[Instruction], int] | None:
    """Compile a tree into a post-order instruction list, or ``None`` if any op is unsupported."""
    plan = _compile(node)
    return None if plan is None else (list(plan.instructions), plan.root)


def _instruction_dependencies(instruction: Instruction) -> tuple[int, ...]:
    op, a, b, ival, _fval, _field = instruction
    if op == OP_LOAD:
        return ()
    name = _OPCODE_NAMES[op]
    if name in _BINARY or name in _COMPARISON or name in _LOGICAL_BINARY:
        return (a, b)
    if name in _BINARY_TS:
        return (a, b)
    if name in _TERNARY:
        return (a, b, ival)
    return (a,)


def _peak_buffers(instructions: tuple[Instruction, ...]) -> int:
    uses = [0] * len(instructions)
    for instruction in instructions:
        for dependency in _instruction_dependencies(instruction):
            uses[dependency] += 1
    live = peak = 0
    for instruction in instructions:
        live += 1  # output is acquired before its inputs are released
        peak = max(peak, live)
        for dependency in _instruction_dependencies(instruction):
            uses[dependency] -= 1
            if uses[dependency] == 0:
                live -= 1
    # Pairwise pandas parity requires pair-masked rolling means and independent std buffers.
    # Include those native temporaries in the per-worker memory guard.
    opcodes = {instruction[0] for instruction in instructions}
    pair_scratch = 7 if _BINARY_TS["ts_corr"] in opcodes else 0
    if _BINARY_TS["ts_cov"] in opcodes:
        pair_scratch = max(pair_scratch, 6)
    return max(1, peak + pair_scratch)


_OPCODE_NAMES = {opcode: name for name, opcode in CPP_OPCODES.items()}


def _emit_into(
    node: Node, instrs: list[Instruction], memo: dict[Node, int]
) -> int | None:
    """Append the instructions computing ``node`` and return its result slot.

    ``memo`` carries structural sharing. Passing one memo across several trees is what makes a
    batch program share their common subexpressions; passing a fresh one per tree reproduces
    the original per-tree plan exactly.
    """

    def emit(
        op: int, a: int = -1, b: int = -1, ival: int = 0, fval: float = 0.0, field: int = -1
    ) -> int:
        instrs.append((op, a, b, ival, fval, field))
        return len(instrs) - 1

    def visit(n: Node) -> int | None:
        if n in memo:
            return memo[n]
        name = n.name
        if name in _FIELD_INDEX:
            field_result = emit(OP_LOAD, field=_FIELD_INDEX[name])
            memo[n] = field_result
            return field_result
        op = CPP_OPCODES.get(name)
        if op is None:
            return None  # unsupported op -> whole-tree fallback
        result: int | None
        if name in _BINARY:
            a, b = visit(n.children[0]), visit(n.children[1])
            result = None if a is None or b is None else emit(op, a=a, b=b)
        elif name in _SCALAR:
            a = visit(n.children[0])
            result = (
                None
                if a is None
                else emit(op, a=a, fval=checked_scalar(n.children[1].value))
            )
        elif name in _TS:
            a = visit(n.children[0])
            result = (
                None
                if a is None
                else emit(op, a=a, ival=checked_window(n.children[1].value),
                          fval=float(_EMA_DECAYED_NEW_WEIGHT) if name == "ts_ema" else 0.)
            )
        elif name in _TS_INITIAL:
            a = visit(n.children[0])
            result = (
                None
                if a is None
                else emit(
                    op,
                    a=a,
                    ival=checked_window(n.children[1].value),
                    fval=checked_scalar(n.children[2].value),
                )
            )
        elif name in _BINARY_TS:
            a, b = visit(n.children[0]), visit(n.children[1])
            result = (
                None
                if a is None or b is None
                else emit(op, a=a, b=b, ival=checked_window(n.children[2].value))
            )
        elif name in _COMPARISON or name in _LOGICAL_BINARY:
            a, b = visit(n.children[0]), visit(n.children[1])
            result = None if a is None or b is None else emit(op, a=a, b=b)
        elif name in _TERNARY:
            condition, when_true, when_false = (visit(child) for child in n.children)
            result = (
                None
                if condition is None or when_true is None or when_false is None
                else emit(op, a=condition, b=when_true, ival=when_false)
            )
        else:
            # unary or cross-sectional: one series child
            a = visit(n.children[0])
            result = None if a is None else emit(op, a=a)
        if result is not None:
            memo[n] = result
        return result

    return visit(node)


@lru_cache(maxsize=_MAX_PLAN_CACHE)
def _compile_expanded(node: Node) -> _CompiledPlan | None:
    instrs: list[Instruction] = []
    root = _emit_into(node, instrs, {})
    if root is None:
        return None
    instructions = tuple(instrs)
    columns: PlanArrays = (
        np.ascontiguousarray([instruction[0] for instruction in instructions], dtype=np.int32),
        np.ascontiguousarray([instruction[1] for instruction in instructions], dtype=np.int32),
        np.ascontiguousarray([instruction[2] for instruction in instructions], dtype=np.int32),
        np.ascontiguousarray([instruction[3] for instruction in instructions], dtype=np.int32),
        np.ascontiguousarray([instruction[4] for instruction in instructions], dtype=np.float64),
        np.ascontiguousarray([instruction[5] for instruction in instructions], dtype=np.int32),
    )
    for column in columns:
        column.setflags(write=False)
    return _CompiledPlan(instructions, root, columns, _peak_buffers(instructions))


def _compile(node: Node) -> _CompiledPlan | None:
    return _compile_expanded(expand_all_cached(node))


@dataclass(frozen=True)
class _SharedProgram:
    """One instruction list computing several trees, each distinct operation emitted once.

    A population is not a set of unrelated expressions. Crossover and mutation reshuffle the
    same building blocks, so the trees in one scoring batch overwhelmingly share subexpressions:
    a profile of a real search found 64% of the array operations in a batch recomputing
    something the same batch had already computed, with ``sub(high, low)`` evaluated 129 times
    over a run and ``delay(close, 1)`` 103 times. Each of those is a full panel-sized pass.

    Sharing them is not free of consequence for parallelism: a shared program has internal
    dependencies, so it cannot be split across threads the way independent plans could. The
    batch is therefore partitioned into one program per worker, which keeps every thread
    independent and lock-free while still collapsing the repeats - the common subexpressions
    are common precisely because they appear in nearly every tree, so they appear in every
    partition too.
    """

    instructions: tuple[Instruction, ...]
    #: Result slot per tree, in the order the trees were given.
    roots: tuple[int, ...]
    arrays: PlanArrays
    root_array: np.ndarray
    peak_buffers: int

    def native_tuple(self) -> tuple[object, ...]:
        return (*self.arrays, self.root_array)


def _shared_peak_buffers(
    instructions: tuple[Instruction, ...], roots: tuple[int, ...]
) -> int:
    """Peak simultaneously-live buffers, with each root held until it is scored.

    The single-root version frees a slot as soon as its last consumer has run. A root has one
    extra consumer - the scorer - which is satisfied the instant the root is computed, because
    scoring happens inline. Everything else is freed exactly as aggressively as before, which
    is what keeps a shared program's working set close to a single plan's.
    """
    uses = [0] * len(instructions)
    for instruction in instructions:
        for dependency in _instruction_dependencies(instruction):
            uses[dependency] += 1
    for root in roots:
        uses[root] += 1
    root_positions = set(roots)

    live = peak = 0
    for position, instruction in enumerate(instructions):
        live += 1  # output is acquired before its inputs are released
        peak = max(peak, live)
        for dependency in _instruction_dependencies(instruction):
            uses[dependency] -= 1
            if uses[dependency] == 0:
                live -= 1
        if position in root_positions:
            # Scored immediately, so the scorer's hold is released here, not at the end.
            uses[position] -= 1
            if uses[position] == 0:
                live -= 1

    opcodes = {instruction[0] for instruction in instructions}
    pair_scratch = 7 if _BINARY_TS["ts_corr"] in opcodes else 0
    if _BINARY_TS["ts_cov"] in opcodes:
        pair_scratch = max(pair_scratch, 6)
    return max(1, peak + pair_scratch)


def compile_batch(expanded_nodes: Sequence[Node]) -> _SharedProgram | None:
    """Compile several *already expanded* trees into one instruction list without repeats.

    Returns ``None`` when any tree contains an operation the backend cannot evaluate; callers
    fall back to per-tree plans for the batch. Pure Python and independently testable, which is
    the point: the interesting logic is the sharing, not the arithmetic.
    """
    instrs: list[Instruction] = []
    memo: dict[Node, int] = {}
    roots: list[int] = []
    for node in expanded_nodes:
        root = _emit_into(node, instrs, memo)
        if root is None:
            return None
        roots.append(root)
    if not roots:
        return None

    instructions = tuple(instrs)
    columns: PlanArrays = (
        np.ascontiguousarray([instruction[0] for instruction in instructions], dtype=np.int32),
        np.ascontiguousarray([instruction[1] for instruction in instructions], dtype=np.int32),
        np.ascontiguousarray([instruction[2] for instruction in instructions], dtype=np.int32),
        np.ascontiguousarray([instruction[3] for instruction in instructions], dtype=np.int32),
        np.ascontiguousarray([instruction[4] for instruction in instructions], dtype=np.float64),
        np.ascontiguousarray([instruction[5] for instruction in instructions], dtype=np.int32),
    )
    root_array = np.ascontiguousarray(roots, dtype=np.int32)
    for column in columns:
        column.setflags(write=False)
    root_array.setflags(write=False)
    return _SharedProgram(
        instructions,
        tuple(roots),
        columns,
        root_array,
        _shared_peak_buffers(instructions, tuple(roots)),
    )


def clear_plan_cache() -> None:
    """Clear cached native IR (primarily useful after runtime registry changes in tests)."""
    _compile_expanded.cache_clear()


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
    if not available():
        return None
    plan = _compile(node)
    if plan is None or not _plan_supported_by_loaded_abi(plan):
        return None
    result = _EXT.evaluate(_panel_arrays(panel), *plan.arrays, plan.root)
    frame = pd.DataFrame(result, index=panel.dates, columns=panel.symbols)
    return frame.astype(bool, copy=False) if node.out_type is DType.BOOL else frame


def ranked_correlations(left: np.ndarray, right: np.ndarray, min_names: int):
    """Optional bounded native pairwise reranking; older builds use the Python baseline."""
    if backend_enabled() and hasattr(_EXT, "ranked_correlations"):
        return _EXT.ranked_correlations(left, right, min_names)
    return None


def evaluate_many(
    nodes: Iterable[Node],
    panel: Panel,
    *,
    workers: int,
    memory_budget_bytes: int | None = None,
) -> list[pd.DataFrame | None]:
    """Evaluate supported trees in deterministic input order with bounded native workers.

    Unsupported trees (and every tree when native evaluation is disabled/unavailable) retain a
    ``None`` marker so callers can apply the Python baseline serially.  ``memory_budget_bytes``
    limits native scratch/output per chunk; it never changes results or drops work.
    """
    materialized = list(nodes)
    results: list[pd.DataFrame | None] = [None] * len(materialized)
    if isinstance(workers, bool) or not isinstance(workers, int) or workers <= 0:
        raise ValueError("workers must be a positive integer")
    if memory_budget_bytes is not None and (
        isinstance(memory_budget_bytes, bool)
        or not isinstance(memory_budget_bytes, int)
        or memory_budget_bytes <= 0
    ):
        raise ValueError("memory_budget_bytes must be a positive integer or None")
    if not materialized or not backend_enabled():
        return results

    supported: list[tuple[int, Node, _CompiledPlan]] = []
    for index, node in enumerate(materialized):
        plan = _compile(node)
        if plan is not None and _plan_supported_by_loaded_abi(plan):
            supported.append((index, node, plan))
    if not supported:
        return results

    frame_bytes = max(1, len(panel.dates) * len(panel.symbols) * np.dtype(np.float64).itemsize)
    max_peak = max(plan.peak_buffers for _index, _node, plan in supported)
    effective_workers = min(workers, native_max_workers(), len(supported))
    chunk_size = len(supported)
    if memory_budget_bytes is not None:
        per_worker = frame_bytes * (max_peak + 1)
        effective_workers = min(
            effective_workers, max(1, memory_budget_bytes // max(1, per_worker))
        )
        scratch = effective_workers * max_peak * frame_bytes
        remaining = memory_budget_bytes - min(memory_budget_bytes, scratch)
        chunk_size = max(1, remaining // frame_bytes)
        chunk_size = max(effective_workers, chunk_size)

    fields = _panel_arrays(panel)
    for start in range(0, len(supported), chunk_size):
        chunk = supported[start : start + chunk_size]
        native = _EXT.evaluate_many(
            fields,
            [plan.native_tuple() for _index, _node, plan in chunk],
            min(effective_workers, len(chunk)),
        )
        for offset, (result_index, node, _plan) in enumerate(chunk):
            # pandas may retain only a raw pointer for a pybind-owned slice; copy so each frame
            # remains valid after the local 3-D batch owner is released/reused by the next chunk.
            owned = np.array(native[offset], dtype=np.float64, order="C", copy=True)
            frame = pd.DataFrame(owned, index=panel.dates, columns=panel.symbols)
            results[result_index] = (
                frame.astype(bool, copy=False) if node.out_type is DType.BOOL else frame
            )
    return results


def _score_row(
    row: Iterable[float],
    size: int,
    unique_size: int,
    penalty_rate: float,
    min_valid_dates: int,
) -> tuple[float, dict[str, float]]:
    """Decode one native score row into the metric tuple ``score_tree`` produces."""
    values = [float(value) for value in row]
    fitness, ic, information_ratio = values[:3]
    valid_dates = values[9]
    metrics = {"ic": ic, "ic_ir": information_ratio}
    if valid_dates >= min_valid_dates:
        metrics.update(
            {
                "raw_objective": ic,
                "expanded_complexity": float(size),
                "expanded_unique_nodes": float(unique_size),
                "complexity_penalty": float(penalty_rate * size),
                "signed_ic": values[3],
                "oriented_ic": values[4],
                "mean_abs_ic": values[5],
                "polarity": values[6],
                "oriented_ic_ir": values[7],
                "sign_consistency": values[8],
                "valid_dates": valid_dates,
                "avg_active_names": values[10],
                "min_active_names": values[11],
            }
        )
    return fitness, metrics


def _distinct_subexpressions(node: Node) -> frozenset[Node]:
    """Every distinct subtree of ``node``, which is what a shared program emits one slot for."""
    found: set[Node] = set()
    stack = [node]
    while stack:
        current = stack.pop()
        if current in found:
            continue
        found.add(current)
        stack.extend(current.children)
    return frozenset(found)


def partition_for_sharing(nodes: Sequence[Node], workers: int) -> list[list[int]]:
    """Split trees into one group per worker, putting trees that overlap in the same group.

    Sharing subexpressions puts dependencies inside a program, so a program cannot be split
    across threads: one program per worker is what keeps every thread independent and lock-free.
    That makes the worker count and the sharing pull against each other, and the pull is not
    mild. Splitting a sixteen-tree batch by index across fifteen workers leaves each program
    with one tree and shares essentially nothing, so on a large machine the tuner's own choice
    of worker count was quietly cancelling the batch-CSE work.

    Grouping by overlap rather than by position recovers a good part of it. Two trees end up
    together because they actually have subexpressions in common, not because they sit next to
    each other in a list, so a small group can share nearly as much as a larger arbitrary one.
    Measured on real populations: at fifteen workers a ninety-six tree batch keeps 29% of its
    instructions shared this way against 18% by index, and a sixteen-tree batch 8% against 1%.

    Groups stay balanced in size on purpose. A perfectly-shared group of thirty that leaves the
    other fourteen workers idle is not a win, and load imbalance is measured in wall clock while
    sharing is measured in instructions.

    The assignment is deterministic and cannot change a score: which program computes a tree
    does not change what the tree computes, and results are written back by original index.
    """
    count = len(nodes)
    if count <= 0:
        return []
    groups = max(1, min(workers, count))
    if groups == 1:
        return [list(range(count))]
    if groups >= count:
        return [[index] for index in range(count)]

    subexpressions = [_distinct_subexpressions(node) for node in nodes]
    base, extra = divmod(count, groups)
    unplaced = set(range(count))
    members: list[list[int]] = []
    for group in range(groups):
        capacity = base + (1 if group < extra else 0)
        if not capacity or not unplaced:
            break
        # Seed with the largest tree left. A big tree offers the most for the rest of the group
        # to latch onto, and placing it while every group is still open is what gives the
        # smaller trees somewhere they genuinely fit.
        seed = max(unplaced, key=lambda index: (len(subexpressions[index]), -index))
        chosen = [seed]
        unplaced.discard(seed)
        union = set(subexpressions[seed])
        while len(chosen) < capacity and unplaced:
            # Grow this group before opening the next, taking whichever tree adds the fewest new
            # slots. Filling one group at a time beats spreading trees across all of them: a
            # round-robin assignment scatters the trees that overlap most before anything has
            # been learned about what overlaps what.
            best = min(
                unplaced, key=lambda index: (len(subexpressions[index] - union), index)
            )
            chosen.append(best)
            unplaced.discard(best)
            union |= subexpressions[best]
        members.append(sorted(chosen))
    # Anything left over (only possible if a capacity was zero) goes to the smallest group.
    for index in sorted(unplaced):
        min(members, key=len).append(index)
    return [sorted(group) for group in members if group]


def _shared_partitions(
    supported: list[tuple[int, Node, _CompiledPlan, int, int]],
    workers: int,
    memory_budget_bytes: int | None,
    frame_bytes: int,
) -> list[tuple[_SharedProgram, list[tuple[int, Node, _CompiledPlan, int, int]]]] | None:
    """Compile one shared program per worker, or ``None`` to keep the per-tree path.

    A shared program holds more buffers at once than a single plan, so a group that would not
    fit the run's memory budget disqualifies the whole batch rather than quietly exceeding it.
    """
    partitions: list[tuple[_SharedProgram, list[tuple[int, Node, _CompiledPlan, int, int]]]] = []
    expanded = [expand_all_cached(node) for _i, node, _p, _s, _u in supported]
    for bounds in partition_for_sharing(expanded, workers):
        members = [supported[index] for index in bounds]
        program = compile_batch([expanded[index] for index in bounds])
        if program is None:
            return None
        if (
            memory_budget_bytes is not None
            and program.peak_buffers * frame_bytes > memory_budget_bytes
        ):
            return None
        partitions.append((program, members))
    return partitions or None


def score_many(
    nodes: Iterable[Node],
    panel: Panel,
    forward_returns: pd.DataFrame,
    *,
    method: str = "spearman",
    absolute: bool = True,
    parsimony: float = 0.0,
    complexity_penalty_mode: str = "per_node",
    complexity_penalty_value: float | None = None,
    max_nodes: int | None = None,
    min_names: int = 5,
    min_valid_dates: int = 5,
    workers: int,
    observations: dict[int, np.ndarray] | None = None,
    memory_budget_bytes: int | None = None,
) -> list[tuple[float, dict[str, float]] | None]:
    """Evaluate and IC-score trees natively without materializing factor DataFrames.

    Results remain in input order and unsupported entries are ``None``. The metric tuple exactly
    mirrors :func:`alphalineage.core.fitness.score_tree`: ``fitness`` plus ``ic``/``ic_ir``.
    """
    materialized = list(nodes)
    results: list[tuple[float, dict[str, float]] | None] = [None] * len(materialized)
    if isinstance(workers, bool) or not isinstance(workers, int) or workers <= 0:
        raise ValueError("workers must be a positive integer")
    if memory_budget_bytes is not None and (
        isinstance(memory_budget_bytes, bool)
        or not isinstance(memory_budget_bytes, int)
        or memory_budget_bytes <= 0
    ):
        raise ValueError("memory_budget_bytes must be a positive integer or None")
    if method not in {"spearman", "pearson"}:
        raise ValueError(f"unknown IC method {method!r}")
    from alphalineage.core.fitness import complexity_penalty_rate

    penalty_rate = complexity_penalty_rate(
        parsimony=parsimony,
        complexity_penalty_mode=complexity_penalty_mode,
        complexity_penalty_value=complexity_penalty_value,
        max_nodes=max_nodes,
    )
    if isinstance(min_names, bool) or not isinstance(min_names, int) or min_names <= 0:
        raise ValueError("min_names must be a positive integer")
    if (
        isinstance(min_valid_dates, bool)
        or not isinstance(min_valid_dates, int)
        or min_valid_dates <= 0
    ):
        raise ValueError("min_valid_dates must be a positive integer")
    # Spearman is the GP default and has exact average-tie integer-rank semantics in native code.
    # Pearson's degenerate constant-row IC-IR depends on NumPy reduction roundoff, so retain the
    # Python scorer rather than silently changing that metric.
    if (
        not materialized
        or not backend_enabled()
        or method == "pearson"
        or (native_abi_version() or 0) < _NATIVE_SCORING_ABI
    ):
        return results

    if observations is not None and (native_abi_version() or 0) < 10:
        return results

    supported: list[tuple[int, Node, _CompiledPlan, int, int]] = []
    for index, node in enumerate(materialized):
        expanded = expand_all_cached(node)
        plan = _compile_expanded(expanded)
        if plan is not None and _plan_supported_by_loaded_abi(plan):
            supported.append(
                (
                    index,
                    node,
                    plan,
                    expanded.size(),
                    expanded.unique_computation_size(),
                )
            )
    if not supported:
        return results

    frame_bytes = max(1, len(panel.dates) * len(panel.symbols) * np.dtype(np.float64).itemsize)
    max_peak = max(
        plan.peak_buffers + 1
        for _index, _node, plan, _size, _unique_size in supported
    )
    effective_workers = min(workers, native_max_workers(), len(supported))
    if memory_budget_bytes is not None:
        effective_workers = min(
            effective_workers,
            max(1, memory_budget_bytes // max(1, max_peak * frame_bytes)),
        )

    target = forward_returns.reindex(index=panel.dates, columns=panel.symbols)
    target_values = np.ascontiguousarray(target.to_numpy(dtype=np.float64, na_value=np.nan))

    shared = (
        _shared_partitions(supported, effective_workers, memory_budget_bytes, frame_bytes)
        if supports_shared_scoring()
        else None
    )
    observed = np.empty((len(supported), len(panel.dates), len(panel.symbols))) if observations is not None else None
    optional = {"observed": observed} if observations is not None else {}
    if shared is not None:
        native_shared = _EXT.score_shared(
            _panel_arrays(panel),
            [program.native_tuple() for program, _members in shared],
            target_values,
            np.ascontiguousarray(
                [size for _program, members in shared for _i, _n, _p, size, _u in members],
                dtype=np.int32,
            ),
            0 if method == "spearman" else 1,
            absolute,
            float(penalty_rate),
            min_names,
            min_valid_dates,
            effective_workers,
            **optional,
        )
        row = 0
        for _program, members in shared:
            for result_index, _node, _plan, size, unique_size in members:
                results[result_index] = _score_row(
                    native_shared[row], size, unique_size, penalty_rate, min_valid_dates
                )
                if observations is not None:
                    observations[result_index] = observed[row]
                row += 1
        return results

    native = _EXT.score_many(
        _panel_arrays(panel),
        [
            plan.native_tuple()
            for _index, _node, plan, _size, _unique_size in supported
        ],
        target_values,
        np.ascontiguousarray(
            [
                size
                for _index, _node, _plan, size, _unique_size in supported
            ],
            dtype=np.int32,
        ),
        0 if method == "spearman" else 1,
        absolute,
        float(penalty_rate),
        min_names,
        min_valid_dates,
        effective_workers,
        **optional,
    )
    for offset, (
        result_index,
        _node,
        _plan,
        size,
        unique_size,
    ) in enumerate(supported):
        results[result_index] = _score_row(
            native[offset], size, unique_size, penalty_rate, min_valid_dates
        )
        if observations is not None:
            observations[result_index] = observed[offset]
    return results
