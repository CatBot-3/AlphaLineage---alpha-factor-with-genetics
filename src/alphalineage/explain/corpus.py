"""P9 - the retrieval corpus: everything a model needs to know about this workspace.

Documents are built from artifacts already on disk — sessions, per-round results, lineage,
saved factors, the primitive registry, and the indicator catalog. Nothing here evaluates a tree
or reads a price panel, so building the corpus can never consume a test read (invariant 1).

The document that makes tuning advice concrete rather than generic is ``genetics``: it carries
the per-generation fitness trajectory, when improvement stalled, how population diversity moved,
and which formulas the parameter search actually explored. "Best fitness flattened at generation
7 of 20 while the unique-tree ratio fell to 0.61" is a retrievable fact, not a guess.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from alphalineage.data import paths
from alphalineage.explain.anatomy import base_name

_TOKEN_RE = re.compile(r"[a-z0-9_]+")
#: Terms that carry no discriminating power in a corpus that is entirely about alpha factors.
_STOPWORDS: frozenset[str] = frozenset(
    """a an and are as at be by для for from in into is it its of on or that the this to with
    was were will be has have had not no than then there these those which while""".split()
)

DOC_KINDS: tuple[str, ...] = ("primitive", "indicator", "session", "round", "genetics", "factor")


def tokenize(text: str) -> list[str]:
    """Lowercase word/underscore tokens, with ``ts_mean`` also emitting ``ts`` and ``mean``.

    Splitting compound operator names matters: a user asking about "moving average" should reach
    ``ts_mean`` documents, and a factor using ``ta_boll_upper`` should reach "bollinger" text.
    """
    out: list[str] = []
    for token in _TOKEN_RE.findall(text.lower()):
        if token in _STOPWORDS or len(token) < 2:
            continue
        out.append(token)
        if "_" in token:
            out.extend(
                part for part in token.split("_") if len(part) > 1 and part not in _STOPWORDS
            )
    return out


@dataclass
class Document:
    """One retrievable unit of context."""

    id: str
    kind: str
    title: str
    text: str
    #: Structural facts used for boosting; not part of the lexical index.
    operators: frozenset[str] = frozenset()
    universe: str = ""
    categories: frozenset[str] = frozenset()
    created_at: str = ""
    tokens: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.tokens:
            self.tokens = tokenize(f"{self.title} {self.text}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "kind": self.kind,
            "title": self.title,
            "text": self.text,
            "universe": self.universe,
            "created_at": self.created_at,
        }


# --- helpers ---------------------------------------------------------------------
def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _tree_operators(node: Any) -> set[str]:
    """Operator/operand names in a serialized tree, revision suffixes stripped."""
    if not isinstance(node, dict):
        return set()
    names = {base_name(str(node.get("name", "")))}
    for child in node.get("children") or ():
        names |= _tree_operators(child)
    return {name for name in names if name}


def _tree_text(node: Any) -> str:
    if not isinstance(node, dict):
        return ""
    name = str(node.get("name", ""))
    children = node.get("children") or ()
    if not children:
        value = node.get("value")
        return str(value) if value is not None else name
    return f"{name}({', '.join(_tree_text(c) for c in children)})"


def _fmt(value: Any, digits: int = 4) -> str:
    if isinstance(value, bool) or value is None:
        return str(value)
    if isinstance(value, (int,)):
        return str(value)
    if isinstance(value, float):
        return "n/a" if math.isnan(value) else f"{value:.{digits}f}"
    return str(value)


# --- document builders -----------------------------------------------------------
def primitive_documents() -> list[Document]:
    """One document per built-in operator: signature, category, and human description."""
    from alphalineage.core.categories import builtin_category
    from alphalineage.core.primitive_docs import PRIMITIVE_DOCS
    from alphalineage.core.primitives import REGISTRY

    docs: list[Document] = []
    for name, prim in sorted(REGISTRY.items()):
        if prim.macro_body is not None:
            continue  # user/catalog formulas are covered by indicator_documents
        info = PRIMITIVE_DOCS.get(name, {})
        signature = ", ".join(t.value for t in prim.arg_types) or "no arguments"
        text = (
            f"{info.get('display_name', name)}. {info.get('description', '')} "
            f"Signature: {name}({signature}) -> {prim.out_type.value}. "
            f"Category: {builtin_category(name)}."
        )
        docs.append(
            Document(
                id=f"primitive:{name}",
                kind="primitive",
                title=f"Operator {name}",
                text=text,
                operators=frozenset({name}),
                categories=frozenset({builtin_category(name)}),
            )
        )
    return docs


def indicator_documents() -> list[Document]:
    """One document per catalog indicator: meaning, parameters, and tuning policy."""
    from alphalineage.library.indicator_catalog import INDICATOR_CATALOG

    docs: list[Document] = []
    for item in INDICATOR_CATALOG:
        name = str(item.get("name", ""))
        if not name or item.get("status") != "active":
            continue
        params = []
        for spec in item.get("inputs") or []:
            tuning = spec.get("tuning") or {}
            bounds = (
                f" (default {tuning.get('default')}, allowed {tuning.get('min')}"
                f"-{tuning.get('max')})"
                if tuning
                else ""
            )
            params.append(f"{spec.get('name')}: {spec.get('description', '')}{bounds}")
        text = (
            f"{item.get('display_name', name)}. {item.get('description', '')} "
            f"Also known as: {', '.join(item.get('aliases') or [])}. "
            f"Family: {item.get('family')}. "
            f"Parameters: {'; '.join(params) if params else 'none'}. "
            f"Definition: {_tree_text(item.get('body'))}."
        )
        docs.append(
            Document(
                id=f"indicator:{name}",
                kind="indicator",
                title=f"Indicator {name}",
                text=text,
                operators=frozenset({name}) | _tree_operators(item.get("body")),
                categories=frozenset({str(item.get("category", ""))}),
            )
        )
    return docs


def _session_text(session: dict[str, Any]) -> str:
    config = session.get("config") or {}
    boundaries = session.get("boundaries") or {}
    rounds = session.get("rounds") or []
    lines = [
        f"Session {session.get('name')} (id {session.get('id')}) on universe "
        f"{session.get('universe')} as of {session.get('as_of')}.",
        f"Split boundaries: train ends {boundaries.get('train_end')}, validation "
        f"{boundaries.get('valid_start')} to {boundaries.get('valid_end')}, test starts "
        f"{boundaries.get('test_start')}, embargo {boundaries.get('embargo')} bars. "
        "The time boundary is frozen at session creation and is never relocated.",
        f"Honesty ledger: cumulative trials {session.get('cumulative_trials')}, "
        f"trial baseline {session.get('trial_baseline')}, test reads {session.get('test_reads')}, "
        f"session holdout reads {session.get('session_holdout_reads')}, "
        f"{len(rounds)} segment(s) run.",
        "Search configuration: "
        + ", ".join(
            f"{key}={config[key]}"
            for key in (
                "population_size",
                "generations",
                "max_depth",
                "max_nodes",
                "min_depth",
                "tournament_size",
                "elitism",
                "crossover_rate",
                "point_mutation_rate",
                "subtree_mutation_rate",
                "parsimony",
                "complexity_penalty_mode",
                "complexity_penalty_value",
                "exploration_profile",
                "parameter_neighbor_fraction",
                "horizon",
                "ic_method",
                "validation_folds",
                "min_names",
                "seed",
            )
            if key in config
        ),
        f"Operator categories enabled: {', '.join(config.get('enabled_categories') or [])}.",
    ]
    formulas = config.get("enabled_formula_names") or []
    if formulas:
        lines.append(f"Formulas enabled ({len(formulas)}): {', '.join(formulas)}.")
    if session.get("seed_factor_ids"):
        lines.append(f"Seeded from saved factors: {', '.join(session['seed_factor_ids'])}.")
    for summary in rounds:
        lines.append(
            f"Segment {summary.get('index')}: generations {summary.get('gen_start')}"
            f"->{summary.get('gen_end')}, status {summary.get('status')}, "
            f"evidence {summary.get('evidence_status')}, "
            f"termination {summary.get('termination_reason')}, "
            f"validity {summary.get('validity')}."
        )
    return " ".join(lines)


def session_documents(sessions_root: Path | None = None) -> list[Document]:
    root = sessions_root or paths.sessions_dir()
    if not root.exists():
        return []
    docs: list[Document] = []
    for directory in sorted(p for p in root.iterdir() if p.is_dir()):
        session = _read_json(directory / "session.json")
        if not isinstance(session, dict):
            continue
        config = session.get("config") or {}
        docs.append(
            Document(
                id=f"session:{session.get('id', directory.name)}",
                kind="session",
                title=f"Session {session.get('name', directory.name)}",
                text=_session_text(session),
                universe=str(session.get("universe", "")),
                categories=frozenset(config.get("enabled_categories") or []),
                created_at=str(session.get("created_at", "")),
                operators=frozenset(config.get("enabled_formula_names") or []),
            )
        )
    return docs


def _round_text(result: dict[str, Any], session_id: str, index: int) -> str:
    context = result.get("context") or {}
    selection = result.get("selection") or {}
    training = selection.get("training_metrics") or {}
    validation = selection.get("validation_metrics") or {}
    timings = result.get("timings") or {}
    lines = [
        f"Segment {index} of session {session_id} on universe {context.get('universe')}, "
        f"horizon {context.get('horizon')}, IC method {context.get('ic_method')}, "
        f"weighting {context.get('weighting_scheme')}, quantile {context.get('quantile')}, "
        f"costs {context.get('commission_bps')}bps commission + "
        f"{context.get('slippage_bps')}bps slippage.",
        f"Ran {result.get('generations')} generations, terminated because "
        f"{result.get('termination_reason')}; evidence status "
        f"{result.get('evidence_status')}; cumulative trials "
        f"{result.get('cumulative_trials')}; test reads {result.get('test_reads')}.",
        f"Selected expression: {_tree_text(_parse_tree(result.get('best_factor')))}",
    ]
    if training:
        lines.append(
            "Training metrics (in-sample, not reportable): "
            + ", ".join(f"{k}={_fmt(v)}" for k, v in sorted(training.items()))
        )
    if validation:
        lines.append(
            "Validation metrics (out-of-sample within the validation window): "
            + ", ".join(f"{k}={_fmt(v)}" for k, v in sorted(validation.items()))
        )
    if selection.get("first_seen") is not None:
        lines.append(
            f"The selected individual first appeared at trial {selection.get('first_seen')} "
            f"with polarity {selection.get('polarity')}."
        )
    if timings:
        lines.append(
            "Timings: " + ", ".join(f"{k}={_fmt(v, 1)}s" for k, v in sorted(timings.items()))
        )
    if result.get("report") is None:
        lines.append(
            "No locked-test report was produced for this segment; it is validation-only, so "
            "the test split remains unread for it."
        )
    return " ".join(lines)


def _parse_tree(value: Any) -> Any:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return None
    return value


def _genetics_text(result: dict[str, Any], session_id: str, index: int) -> str:
    history = [h for h in (result.get("history") or []) if isinstance(h, dict)]
    if not history:
        return ""
    first, last = history[0], history[-1]
    best = [
        float(value) for h in history if isinstance(value := h.get("best_fitness"), (int, float))
    ]
    peak = max(best) if best else None
    peak_gen = (
        next(
            (h.get("generation") for h in history if h.get("best_fitness") == peak),
            None,
        )
        if peak is not None
        else None
    )
    stall = last.get("generations_since_improvement")
    ratios = [
        float(value)
        for h in history
        if isinstance(value := h.get("unique_tree_ratio"), (int, float))
    ]
    warnings = sum(1 for h in history if h.get("diversity_warning"))

    lines = [
        f"Evolution trajectory for segment {index} of session {session_id} over "
        f"{len(history)} recorded generations.",
        f"Best fitness moved from {_fmt(first.get('best_fitness'))} at generation "
        f"{first.get('generation')} to {_fmt(last.get('best_fitness'))} at generation "
        f"{last.get('generation')}; peak {_fmt(peak)} first reached at generation {peak_gen}.",
        f"Mean population fitness moved from {_fmt(first.get('mean_fitness'))} to "
        f"{_fmt(last.get('mean_fitness'))}.",
        f"At the end, {stall} generation(s) had passed with no improvement and the champion "
        f"was {last.get('champion_age')} generation(s) old.",
    ]
    if ratios:
        lines.append(
            f"Population diversity (unique-tree ratio) ranged {_fmt(min(ratios), 3)} to "
            f"{_fmt(max(ratios), 3)}, ending at {_fmt(ratios[-1], 3)}; "
            f"{warnings} generation(s) raised a diversity warning."
        )
    lines.append(
        f"Final generation produced {last.get('novel_offspring_count')} novel offspring, "
        f"{last.get('parameter_neighbor_count')} parameter-neighbour probes, and "
        f"{last.get('duplicate_count')} duplicates out of "
        f"{last.get('unique_tree_count')} unique trees."
    )

    if peak_gen is not None and isinstance(peak_gen, (int, float)) and len(history) > 1:
        fraction = float(peak_gen) / max(float(last.get("generation") or 1), 1.0)
        if fraction <= 0.5:
            lines.append(
                "The best fitness was reached in the first half of the run and did not improve "
                "afterwards, which is the signature of premature convergence: the remaining "
                "generations spent budget without finding anything better."
            )
        elif fraction >= 0.9:
            lines.append(
                "The best fitness was still improving at the end of the run, which suggests the "
                "generation budget, not the search, was the binding constraint."
            )

    exploration = last.get("formula_exploration")
    if isinstance(exploration, dict) and exploration:
        ranked = sorted(
            ((name, stats) for name, stats in exploration.items() if isinstance(stats, dict)),
            key=lambda item: item[1].get("calls_searched", 0),
            reverse=True,
        )[:8]
        lines.append(
            "Parameter search by formula: "
            + "; ".join(
                f"{name}: {stats.get('calls_searched')} calls over "
                f"{stats.get('distinct_parameter_tuples')} distinct parameter tuples, "
                f"best training {_fmt(stats.get('best_training_score'))}, "
                f"best validation {_fmt(stats.get('best_validation_score'))}"
                for name, stats in ranked
            )
            + "."
        )

    ops = _lineage_operator_mix(result)
    if ops:
        lines.append("Genetic operator mix across the recorded lineage: " + ops + ".")
    return " ".join(lines)


#: Genetic-operation label -> family. Raw labels are far too granular to summarize (a single run
#: emits ~80 distinct ones, one per formula per parameter axis), so they are folded into the
#: families a user would actually retune.
_OPERATOR_FAMILIES: tuple[tuple[str, str], ...] = (
    ("parameter_beam", "parameter search"),
    ("parameter_frontier", "parameter search"),
    ("crossover", "crossover"),
    ("composition_crossover", "crossover"),
    ("stepping_stone", "stepping stone"),
    ("two_edit", "mutation"),
    ("point", "mutation"),
    ("subtree", "mutation"),
    ("insertion", "mutation"),
    ("reproduction", "reproduction"),
    ("elite", "elitism"),
    ("init", "seeding"),
    ("formula_default", "seeding"),
    ("immigrant", "immigration"),
)


def _operator_family(op: str) -> str:
    head = op.split(":", 1)[0].split("+", 1)[0]
    for prefix, family in _OPERATOR_FAMILIES:
        if head == prefix or head.startswith(prefix):
            return family
    return "other"


def _lineage_operator_mix(result: dict[str, Any]) -> str:
    """Which genetic operations produced individuals, and which produced the best ones.

    Reported by family and capped: the raw per-formula labels run to thousands of characters and
    would eat the whole retrieval budget for a single document.
    """
    lineage = result.get("lineage")
    nodes = lineage.get("nodes") if isinstance(lineage, dict) else None
    if not isinstance(nodes, list) or not nodes:
        return ""
    counts: dict[str, int] = {}
    best: dict[str, float] = {}
    for node in nodes:
        if not isinstance(node, dict):
            continue
        family = _operator_family(str(node.get("op", "unknown")))
        counts[family] = counts.get(family, 0) + 1
        fitness = node.get("fitness")
        if isinstance(fitness, (int, float)) and not math.isnan(fitness):
            if family not in best or fitness > best[family]:
                best[family] = float(fitness)
    total = sum(counts.values()) or 1
    ranked = sorted(counts.items(), key=lambda item: -item[1])[:8]
    return ", ".join(
        f"{family} produced {n} individuals ({n / total:.0%}), "
        f"best fitness {_fmt(best.get(family))}"
        for family, n in ranked
    )


def round_documents(sessions_root: Path | None = None) -> list[Document]:
    """One ``round`` and one ``genetics`` document per completed segment."""
    root = sessions_root or paths.sessions_dir()
    if not root.exists():
        return []
    docs: list[Document] = []
    for directory in sorted(p for p in root.iterdir() if p.is_dir()):
        session = _read_json(directory / "session.json")
        session_id = str(session.get("id")) if isinstance(session, dict) else directory.name
        universe = str(session.get("universe", "")) if isinstance(session, dict) else ""
        rounds_root = directory / "rounds"
        if not rounds_root.exists():
            continue
        for path in sorted(rounds_root.glob("*.json")):
            result = _read_json(path)
            if not isinstance(result, dict):
                continue
            index = int(result.get("round_index", result.get("segment", 0)) or 0)
            tree = _parse_tree(result.get("best_factor"))
            operators = _tree_operators(tree)
            created = str((result.get("round_metadata") or {}).get("completed_at", ""))
            docs.append(
                Document(
                    id=f"round:{session_id}:{index}",
                    kind="round",
                    title=f"Segment {index} of session {session_id}",
                    text=_round_text(result, session_id, index),
                    operators=frozenset(operators),
                    universe=universe,
                    created_at=created,
                )
            )
            genetics = _genetics_text(result, session_id, index)
            if genetics:
                docs.append(
                    Document(
                        id=f"genetics:{session_id}:{index}",
                        kind="genetics",
                        title=f"Evolution trajectory, segment {index} of session {session_id}",
                        text=genetics,
                        operators=frozenset(operators),
                        universe=universe,
                        created_at=created,
                    )
                )
    return docs


def factor_documents(factors_root: Path | None = None) -> list[Document]:
    """One document per saved factor, including its computed shape so it is searchable."""
    from alphalineage.core.tree import from_dict as tree_from_dict
    from alphalineage.explain.anatomy import measure

    root = factors_root or paths.factors_dir()
    if not root.exists():
        return []
    docs: list[Document] = []
    for path in sorted(root.glob("*.json")):
        data = _read_json(path)
        if not isinstance(data, dict) or "tree" not in data:
            continue
        tree_dict = data["tree"]
        expanded_dict = data.get("expanded_tree") or tree_dict
        shape = ""
        try:
            measurement = measure(tree_from_dict(tree_dict), expanded=tree_from_dict(expanded_dict))
            shape = (
                f"reads {', '.join(measurement.structure.data_fields) or 'nothing'}; "
                f"~{measurement.windows.effective_lookback_bars} bar lookback; "
                f"{measurement.structure.node_count} nodes; "
                f"root {measurement.structure.root}"
            )
        except (KeyError, TypeError, ValueError):
            pass
        metrics = data.get("metrics") or {}
        provenance = data.get("provenance") or {}
        text = (
            f"Saved factor '{data.get('name')}' (kind {data.get('kind', 'training')}), "
            f"saved {data.get('saved_at')}. "
            f"Expression: {_tree_text(tree_dict)}. "
            f"Shape: {shape or 'unmeasured'}. "
            f"Metrics: "
            f"{', '.join(f'{k}={_fmt(v)}' for k, v in sorted(metrics.items())) or 'none'}. "
            f"Provenance: session {provenance.get('session_id')}, cumulative trials "
            f"{provenance.get('cumulative_trials')}, test reads {provenance.get('test_reads')}. "
            f"{data.get('notes', '')}"
        )
        docs.append(
            Document(
                id=f"factor:{data.get('id', path.stem)}",
                kind="factor",
                title=f"Saved factor {data.get('name', path.stem)}",
                text=text,
                operators=frozenset(_tree_operators(tree_dict) | _tree_operators(expanded_dict)),
                universe=str((data.get("configuration") or {}).get("universe") or ""),
                created_at=str(data.get("saved_at", "")),
            )
        )
    return docs


def build_corpus(
    *,
    sessions_root: Path | None = None,
    factors_root: Path | None = None,
    include: Iterable[str] | None = None,
) -> list[Document]:
    """Assemble every document kind. Deterministic order, safe on a cold/empty workspace."""
    wanted = set(include) if include is not None else set(DOC_KINDS)
    builders: list[tuple[str, Any]] = [
        ("primitive", primitive_documents),
        ("indicator", indicator_documents),
        ("session", lambda: session_documents(sessions_root)),
        ("round", lambda: round_documents(sessions_root)),
        ("genetics", lambda: round_documents(sessions_root)),
        ("factor", lambda: factor_documents(factors_root)),
    ]
    seen: set[str] = set()
    docs: list[Document] = []
    for kind, builder in builders:
        if kind not in wanted:
            continue
        try:
            produced = builder()
        except Exception:  # noqa: BLE001 - a malformed artifact must not break explanation
            continue
        for doc in produced:
            if doc.kind in wanted and doc.id not in seen:
                seen.add(doc.id)
                docs.append(doc)
    return docs


def iter_kinds(docs: Iterable[Document]) -> Iterator[tuple[str, int]]:
    counts: dict[str, int] = {}
    for doc in docs:
        counts[doc.kind] = counts.get(doc.kind, 0) + 1
    yield from sorted(counts.items())
