"""P11 - measuring an expression tree. Arithmetic only; no judgement.

This module used to do two jobs: classify a tree against a hand-written dictionary of ~15 "aspect"
labels, and compute numbers from the tree's shape. The dictionary is gone (P11-T2). It needed a
human edit every time an operator was added, or the new operator silently got no label — it did
not grow with the system, which is exactly the property a research tool needs.

What remains is the half that *does* grow on its own, because it reads structure rather than
consulting a list:

* the compounded **effective lookback** — nested windows stack, so
  ``ts_std(ts_mean(close, 20), 20)``
  needs ~39 bars, not 20, and a real factor can reach 124 bars while its largest window says 30;
* the **window profile**, depth, size, operator counts, and which market-data fields are read;
* a coarse **unit** reading that flags adding a price to a volume;
* structural **diagnostics** with severities.

All of it works on any operator, including ones added next year. Semantic labels now come from the
model, which is what judgement is for; these numbers are what it would get wrong.

Not investment advice. Research output only (invariant 8).
"""

from __future__ import annotations

import re
import statistics
from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Any

from alphalineage.core.tree import Node

_REVISION_SUFFIX = re.compile(r"__r\d+$")

#: Bars of history each lookback operator consumes: base name -> (window arg index, mode).
#: ``delta``/``delay`` reach exactly ``w`` bars back; a rolling window of length ``w`` reaches
#: ``w - 1``.
_LOOKBACK_OPS: dict[str, tuple[int, str]] = {
    "ts_mean": (1, "window"),
    "ts_std": (1, "window"),
    "ts_std_pop": (1, "window"),
    "ts_sum": (1, "window"),
    "ts_min": (1, "window"),
    "ts_max": (1, "window"),
    "ts_rank": (1, "window"),
    "decay_linear": (1, "window"),
    "ts_corr": (2, "window"),
    "ts_cov": (2, "window"),
    "delta": (1, "shift"),
    "delay": (1, "shift"),
    # Recursive smoothers depend on all history; their practical memory is a small multiple of
    # the nominal span, which is what a user reasons about.
    "ts_ema": (1, "recursive"),
    "ts_rma": (1, "recursive"),
    "ts_recursive_smooth": (1, "recursive"),
}

#: Operators with an unbounded (expanding) memory: the value depends on where the sample starts.
_EXPANDING_OPS: frozenset[str] = frozenset({"ts_cumsum"})

#: Smoothing operators, tracked so nested smoothing can be reported as phase lag.
SMOOTHING_OPERATORS: frozenset[str] = frozenset(
    {"ts_mean", "ts_ema", "ts_rma", "ts_recursive_smooth", "decay_linear"}
)

PRICE_FIELDS: frozenset[str] = frozenset({"open", "high", "low", "close", "vwap"})
VOLUME_FIELDS: frozenset[str] = frozenset({"volume"})
RETURN_FIELDS: frozenset[str] = frozenset({"returns"})

_SHORT_WINDOW_MAX = 5
_MEDIUM_WINDOW_MAX = 20
_CONCRETE_UNITS = frozenset({"price", "volume", "return"})


def base_name(name: str) -> str:
    """Strip a catalog revision suffix so ``ta_atr__r2`` reads as ``ta_atr``."""
    return _REVISION_SUFFIX.sub("", name)


# --- report structures -----------------------------------------------------------
@dataclass(frozen=True)
class WindowProfile:
    """Every lookback the factor declares, plus how far back it actually reaches."""

    values: tuple[int, ...] = ()
    count: int = 0
    minimum: int | None = None
    median: float | None = None
    maximum: int | None = None
    short: int = 0
    medium: int = 0
    long: int = 0
    effective_lookback_bars: int = 0
    unbounded_lookback: bool = False
    recursive_smoothing: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "values": list(self.values)}


@dataclass(frozen=True)
class StructureProfile:
    """Shape facts: size, operator mix, which market-data channels are read, what repeats."""

    depth: int = 0
    node_count: int = 0
    unique_node_count: int = 0
    distinct_operators: int = 0
    operator_counts: dict[str, int] = field(default_factory=dict)
    data_fields: tuple[str, ...] = ()
    indicators: tuple[str, ...] = ()
    root: str = ""
    root_is_cross_sectional: bool = False
    scale_invariant: bool = False
    constant_leaf_fraction: float = 0.0
    repeated_subtrees: tuple[dict[str, Any], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "data_fields": list(self.data_fields),
            "indicators": list(self.indicators),
            "repeated_subtrees": [dict(item) for item in self.repeated_subtrees],
        }


@dataclass(frozen=True)
class UnitProfile:
    """Coarse dimensional reading, and any suspicious combinations."""

    output_unit: str = "unknown"
    mismatches: tuple[dict[str, str], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {"output_unit": self.output_unit, "mismatches": [dict(m) for m in self.mismatches]}


@dataclass(frozen=True)
class Diagnostic:
    """A deterministic observation. Never a verdict, never advice."""

    code: str
    severity: str  # info | caution | warning
    message: str
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Measurement:
    """The full computed report: serializable, prompt-ready, directly renderable."""

    formula: str
    expanded_formula: str
    windows: WindowProfile
    structure: StructureProfile
    units: UnitProfile
    diagnostics: tuple[Diagnostic, ...]
    horizon: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "formula": self.formula,
            "expanded_formula": self.expanded_formula,
            "horizon": self.horizon,
            "windows": self.windows.to_dict(),
            "structure": self.structure.to_dict(),
            "units": self.units.to_dict(),
            "diagnostics": [d.to_dict() for d in self.diagnostics],
        }


# --- text rendering --------------------------------------------------------------
def formula_text(node: Node) -> str:
    """Function-call rendering that never consults the registry.

    ``Node.__str__`` needs the primitive to decide leaf vs call, which fails for a factor whose
    user operators are not currently registered. Measuring must work on any stored artifact, so
    shape alone decides.
    """
    if not node.children:
        return str(node.value) if node.value is not None else node.name
    return f"{node.name}({', '.join(formula_text(c) for c in node.children)})"


# --- windows and lookback --------------------------------------------------------
def _window_values(node: Node) -> list[int]:
    return [
        int(sub.value)
        for sub in node.iter_nodes()
        if sub.name == "window" and isinstance(sub.value, (int, float))
    ]


def effective_lookback(node: Node) -> tuple[int, bool, bool]:
    """Deepest chain of lookback consumption: (bars, unbounded, uses recursive smoothing).

    The single most under-estimated property of a generated factor, and the one a language model
    is least able to work out by eye.
    """
    unbounded = False
    recursive = False

    def visit(current: Node) -> int:
        nonlocal unbounded, recursive
        name = base_name(current.name)
        child_max = max((visit(c) for c in current.children if c.name != "window"), default=0)
        if name in _EXPANDING_OPS:
            unbounded = True
            return child_max
        spec = _LOOKBACK_OPS.get(name)
        if spec is None:
            return child_max
        index, mode = spec
        window = 0
        if index < len(current.children):
            value = current.children[index].value
            if isinstance(value, (int, float)):
                window = int(value)
        if mode == "recursive":
            recursive = True
            own = max(window - 1, 0)
        elif mode == "shift":
            own = window
        else:
            own = max(window - 1, 0)
        return child_max + own

    total = visit(node)
    return total, unbounded, recursive


def window_profile(expanded: Node) -> WindowProfile:
    """Window sizes and effective lookback of an expanded tree.

    Public because callers outside the explanation path need it too: a signal snapshot
    trims history to this lookback so a refresh reads a few hundred bars, not twenty years.
    """
    values = _window_values(expanded)
    lookback, unbounded, recursive = effective_lookback(expanded)
    if not values:
        return WindowProfile(
            effective_lookback_bars=lookback,
            unbounded_lookback=unbounded,
            recursive_smoothing=recursive,
        )
    return WindowProfile(
        values=tuple(sorted(values)),
        count=len(values),
        minimum=min(values),
        median=float(statistics.median(values)),
        maximum=max(values),
        short=sum(1 for v in values if v <= _SHORT_WINDOW_MAX),
        medium=sum(1 for v in values if _SHORT_WINDOW_MAX < v <= _MEDIUM_WINDOW_MAX),
        long=sum(1 for v in values if v > _MEDIUM_WINDOW_MAX),
        effective_lookback_bars=lookback,
        unbounded_lookback=unbounded,
        recursive_smoothing=recursive,
    )


# --- structure -------------------------------------------------------------------
def _structure_profile(surface: Node, expanded: Node) -> StructureProfile:
    counts = Counter(base_name(sub.name) for sub in expanded.iter_nodes() if sub.children)
    fields = sorted(
        {
            sub.name
            for sub in expanded.iter_nodes()
            if not sub.children and sub.value is None and sub.name != "$arg"
        }
    )
    indicators = sorted(
        {
            base_name(sub.name)
            for sub in surface.iter_nodes()
            if base_name(sub.name).startswith("ta_")
        }
    )
    leaves = [sub for sub in expanded.iter_nodes() if not sub.children]
    constants = [leaf for leaf in leaves if leaf.value is not None]

    subtree_counts = Counter(formula_text(sub) for sub in expanded.iter_nodes() if sub.size() >= 3)
    repeated = tuple(
        {"expression": text, "occurrences": n} for text, n in subtree_counts.most_common(3) if n > 1
    )

    root = base_name(expanded.name)
    return StructureProfile(
        depth=expanded.depth(),
        node_count=expanded.size(),
        unique_node_count=expanded.unique_size(),
        distinct_operators=len(counts),
        operator_counts=dict(sorted(counts.items())),
        data_fields=tuple(fields),
        indicators=tuple(indicators),
        root=root,
        root_is_cross_sectional=root in {"rank", "zscore", "scale"},
        scale_invariant=root in {"rank", "zscore", "scale", "ts_rank", "sign"},
        constant_leaf_fraction=(len(constants) / len(leaves)) if leaves else 0.0,
        repeated_subtrees=repeated,
    )


# --- units -----------------------------------------------------------------------
def _unit_profile(expanded: Node) -> UnitProfile:
    mismatches: list[dict[str, str]] = []
    cache: dict[Node, str] = {}

    def unit(node: Node) -> str:
        cached = cache.get(node)
        if cached is not None:
            return cached
        cache[node] = value = _compute_unit(node)
        return value

    def _compute_unit(node: Node) -> str:  # noqa: PLR0911 - a dispatch table by another name
        name = base_name(node.name)
        if name in PRICE_FIELDS:
            return "price"
        if name in VOLUME_FIELDS:
            return "volume"
        if name in RETURN_FIELDS:
            return "return"
        if name in {"const", "window"}:
            return "dimensionless"
        if not node.children:
            return "unknown"
        series = [c for c in node.children if c.name != "window"]
        if name in {"rank", "zscore", "scale", "ts_rank", "sign", "ts_corr"}:
            return "dimensionless"
        if name in {"gt", "lt", "ge", "le", "and_", "or_", "not_"}:
            return "bool"
        if name in {"add", "sub"} and len(series) == 2:
            left, right = unit(series[0]), unit(series[1])
            if left in _CONCRETE_UNITS and right in _CONCRETE_UNITS and left != right:
                mismatches.append(
                    {
                        "expression": formula_text(node)[:160],
                        "left_unit": left,
                        "right_unit": right,
                    }
                )
                return "mixed"
            if left == right:
                return left
            return left if left in _CONCRETE_UNITS else right
        if name == "div" and len(series) == 2:
            return "dimensionless" if unit(series[0]) == unit(series[1]) else "ratio"
        if name in {"mul", "ts_cov"} and len(series) == 2:
            left, right = unit(series[0]), unit(series[1])
            if left == "dimensionless":
                return right
            if right == "dimensionless":
                return left
            return "composite"
        if name in {"log", "signed_power"}:
            return "dimensionless"
        if name == "where" and len(series) == 3:
            return unit(series[1]) if unit(series[1]) == unit(series[2]) else "mixed"
        if series:
            return unit(series[0])  # smoothers, dispersion, shifts, abs/neg preserve their input
        return "unknown"

    output = unit(expanded)
    deduped: list[dict[str, str]] = []
    for item in mismatches:
        if item not in deduped:
            deduped.append(item)
    return UnitProfile(output_unit=output, mismatches=tuple(deduped[:5]))


# --- diagnostics -----------------------------------------------------------------
def _smoothing_chain_depth(node: Node) -> int:
    def visit(current: Node, run: int) -> int:
        next_run = run + 1 if base_name(current.name) in SMOOTHING_OPERATORS else 0
        return max(
            [next_run, *(visit(c, next_run) for c in current.children if c.name != "window")]
        )

    return visit(node, 0)


def _diagnostics(
    *,
    windows: WindowProfile,
    structure: StructureProfile,
    units: UnitProfile,
    expanded: Node,
    horizon: int | None,
    train_bars: int | None,
    policy_bounds: dict[str, dict[str, Any]] | None,
    surface: Node,
) -> tuple[Diagnostic, ...]:
    out: list[Diagnostic] = []

    if not structure.root_is_cross_sectional:
        out.append(
            Diagnostic(
                "no_cross_sectional",
                "caution",
                "The expression is never compared across the universe on a given date.",
                "Fitness here is a cross-sectional rank IC, so a monotone per-date transform "
                "would not change the score — but an explicit rank() or zscore() at the root "
                "makes the signal scale-comparable across symbols and is what portfolio "
                "construction expects.",
            )
        )

    if horizon and windows.effective_lookback_bars > 20 * horizon:
        out.append(
            Diagnostic(
                "long_lookback_vs_horizon",
                "caution",
                f"Effective lookback is ~{windows.effective_lookback_bars} bars against a "
                f"{horizon}-bar forward horizon.",
                "A slow state predicting a fast return is legitimate, but the signal changes "
                "little day to day: expect low turnover, high autocorrelation of the score, and "
                "an effective sample size well below the number of observations.",
            )
        )

    if train_bars and windows.effective_lookback_bars > 0.1 * train_bars:
        out.append(
            Diagnostic(
                "lookback_vs_train_window",
                "warning",
                f"Effective lookback (~{windows.effective_lookback_bars} bars) consumes more "
                f"than 10% of the training window (~{train_bars} bars).",
                "The warm-up period is dead weight in every split, and the longest windows are "
                "fit on the fewest independent observations.",
            )
        )

    if windows.unbounded_lookback:
        out.append(
            Diagnostic(
                "unbounded_lookback",
                "caution",
                "An expanding accumulation (ts_cumsum) makes the value depend on where the "
                "sample starts.",
                "Two symbols with different history lengths accumulate different totals, and the "
                "same symbol's value shifts when the panel start date moves. Consider a "
                "fixed-length ts_sum, or a cross-sectional rank at the root to remove the level.",
            )
        )

    chain = _smoothing_chain_depth(expanded)
    if chain >= 3:
        out.append(
            Diagnostic(
                "nested_smoothing",
                "caution",
                f"{chain} smoothing operators are nested along one path.",
                "Each smoother adds phase lag. Stacked smoothing produces a very slow, very "
                "smooth series whose turning points arrive late relative to the underlying move.",
            )
        )

    if units.mismatches:
        first = units.mismatches[0]
        out.append(
            Diagnostic(
                "unit_mismatch",
                "caution",
                f"Adds or subtracts quantities in different units "
                f"({first['left_unit']} vs {first['right_unit']}).",
                "A heuristic reading, not an error: the GP will happily combine a price and a "
                "volume because the arithmetic is valid. The result is dominated by whichever "
                "input has the larger numeric scale, which is rarely what was intended.",
            )
        )

    if structure.constant_leaf_fraction > 0.4:
        out.append(
            Diagnostic(
                "constant_heavy",
                "info",
                f"{structure.constant_leaf_fraction:.0%} of leaves are tuned numbers "
                "(windows and scalars) rather than market-data fields.",
                "Every one of those is a fitted parameter. A high ratio means more tuning per "
                "unit of data, which is where overfitting concentrates and what the trial count "
                "is meant to price in.",
            )
        )

    if structure.depth > 12:
        out.append(
            Diagnostic(
                "deep_expression",
                "info",
                f"Expression depth is {structure.depth}.",
                "Depth is not itself bad, but past roughly a dozen levels a tree stops being "
                "readable, and bloat is the usual GP failure mode. Check whether the complexity "
                "penalty is doing any work.",
            )
        )

    if structure.operator_counts.get("div"):
        out.append(
            Diagnostic(
                "division_present",
                "info",
                "The expression divides by a computed series.",
                "The evaluator returns NaN rather than infinity where a denominator is ~0, so "
                "near-zero denominators silently drop names from that date's cross-section.",
            )
        )

    out.extend(_policy_bound_diagnostics(surface, policy_bounds))
    return tuple(out)


def _policy_bound_diagnostics(
    surface: Node, policy_bounds: dict[str, dict[str, Any]] | None
) -> list[Diagnostic]:
    """Flag tuned parameters sitting exactly on their allowed minimum or maximum.

    A window pinned at the edge of its search policy means the search wanted to go further and
    could not — the bound, not the data, chose the value.
    """
    if not policy_bounds:
        return []
    pinned: list[str] = []
    for sub in surface.iter_nodes():
        policy = policy_bounds.get(sub.name) or policy_bounds.get(base_name(sub.name))
        if not policy:
            continue
        for index, spec in enumerate(policy.get("inputs") or []):
            if index >= len(sub.children) or not isinstance(spec, dict):
                continue
            value = sub.children[index].value
            tuning = spec.get("tuning")
            if value is None or not isinstance(tuning, dict):
                continue
            label = str(spec.get("name") or f"input_{index + 1}")
            if tuning.get("min") is not None and value == tuning["min"]:
                pinned.append(f"{base_name(sub.name)}.{label} = {value} (policy minimum)")
            elif tuning.get("max") is not None and value == tuning["max"]:
                pinned.append(f"{base_name(sub.name)}.{label} = {value} (policy maximum)")
    if not pinned:
        return []
    return [
        Diagnostic(
            "parameter_at_policy_bound",
            "info",
            "Tuned parameters sit on the edge of their allowed range: " + "; ".join(pinned[:4]),
            "A binding bound means the local search wanted to move further in that direction. "
            "Widening the parameter policy would tell you whether the value is a real optimum.",
        )
    ]


# --- entry point -----------------------------------------------------------------
def measure(
    tree: Node,
    *,
    expanded: Node | None = None,
    horizon: int | None = None,
    train_bars: int | None = None,
    policy_bounds: dict[str, dict[str, Any]] | None = None,
) -> Measurement:
    """Measure a factor. ``expanded`` defaults to ``tree`` when macros are already inlined."""
    surface = tree
    inlined = expanded if expanded is not None else tree
    windows = window_profile(inlined)
    structure = _structure_profile(surface, inlined)
    units = _unit_profile(inlined)
    diagnostics = _diagnostics(
        windows=windows,
        structure=structure,
        units=units,
        expanded=inlined,
        horizon=horizon,
        train_bars=train_bars,
        policy_bounds=policy_bounds,
        surface=surface,
    )
    return Measurement(
        formula=formula_text(surface),
        expanded_formula=formula_text(inlined),
        windows=windows,
        structure=structure,
        units=units,
        diagnostics=diagnostics,
        horizon=horizon,
    )


def render_markdown(measurement: Measurement) -> str:
    """Compact Markdown rendering — what the model is handed as computed ground truth."""
    lines: list[str] = []
    w = measurement.windows
    lines.append("### Lookback")
    if w.count:
        lines.append(
            f"- {w.count} window(s): min {w.minimum}, median {w.median:g}, max {w.maximum} "
            f"(short {w.short} / medium {w.medium} / long {w.long})"
        )
    else:
        lines.append("- no explicit windows")
    lines.append(
        f"- effective lookback: ~{w.effective_lookback_bars} bars "
        "(nested windows compound; this is not the largest window)"
    )
    if w.unbounded_lookback:
        lines.append("- contains an expanding (unbounded) accumulation")
    if w.recursive_smoothing:
        lines.append("- contains recursive smoothing (memory exceeds the nominal span)")
    if measurement.horizon:
        lines.append(f"- forward horizon being predicted: {measurement.horizon} bar(s)")

    s = measurement.structure
    lines.append("")
    lines.append("### Structure")
    lines.append(
        f"- depth {s.depth}, {s.node_count} nodes ({s.unique_node_count} structurally distinct), "
        f"{s.distinct_operators} distinct operators"
    )
    lines.append(f"- data fields read: {', '.join(s.data_fields) or 'none'}")
    if s.indicators:
        lines.append(f"- named indicators used: {', '.join(s.indicators)}")
    lines.append(
        f"- root operator: {s.root} "
        f"({'cross-sectional' if s.root_is_cross_sectional else 'not cross-sectional'})"
    )
    for repeat in s.repeated_subtrees:
        lines.append(f"- repeated subexpression x{repeat['occurrences']}: {repeat['expression']}")

    lines.append("")
    lines.append(f"### Units\n- output reads as: {measurement.units.output_unit}")
    for mismatch in measurement.units.mismatches:
        lines.append(
            f"- mixes {mismatch['left_unit']} with {mismatch['right_unit']} in "
            f"`{mismatch['expression']}`"
        )

    lines.append("")
    lines.append("### Diagnostics")
    if measurement.diagnostics:
        for d in measurement.diagnostics:
            lines.append(f"- [{d.severity}] {d.message} {d.detail}".rstrip())
    else:
        lines.append("- none")
    return "\n".join(lines)
