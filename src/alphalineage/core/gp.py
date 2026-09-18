"""P2-T1, T3-T7 - the genetic-programming search.

A population of typed trees evolved by tournament selection, type-safe subtree crossover,
and mutation, scored by |mean signed rank IC| (invariants 3 and 4). The loop is deterministic
(one RNG), budget-bounded (generations and/or wall-clock), and checkpointable/resumable:
a checkpoint captures the RNG state *after* a generation is scored and *before* the next is
bred, so resuming reproduces the run bit-for-bit.
"""

from __future__ import annotations

import dataclasses
import json
import math
import random
import time
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from alphalineage.core.categories import CUSTOM, builtin_category
from alphalineage.core.extensions import InvalidOperator, expand_all
from alphalineage.core.fitness import (
    COMPLEXITY_PENALTY_MODES,
    DEFAULT_NORMALIZED_COMPLEXITY_PENALTY,
    forward_returns,
    score_trees,
    validate_execution,
)
from alphalineage.core.generate import GenerationError, RandomTreeGenerator, operator_allowed
from alphalineage.core.panel import Panel
from alphalineage.core.primitives import OPERANDS, OPERATORS, Kind
from alphalineage.core.simplify import simplify
from alphalineage.core.tree import Node, from_dict, from_json, to_dict, to_json, validate
from alphalineage.core.types import DType, is_subtype

Path_ = str | Path
Position = tuple[tuple[int, ...], DType, Node]

# Hard ceilings keep malformed API/config input from creating an effectively unbounded local job.
# The defaults (200 x 25, depth 6, 40 nodes) remain far below these desktop-oriented limits.
MAX_POPULATION_SIZE = 5_000
MAX_GENERATIONS = 1_000
MAX_SEARCH_EVALUATIONS = 1_000_000
MAX_TREE_DEPTH = 32
MAX_TREE_NODES = 2_000
MAX_MIN_NAMES = 10_000
MAX_HORIZON = 252
MAX_TIME_BUDGET_S = 7 * 24 * 60 * 60
MAX_TRAINING_WORKERS = 32

# Checkpoints carry the scorer version separately from the scientific GP configuration.  A
# worker-count change never invalidates a checkpoint, while an algorithm change does: continuing
# an older checkpoint first re-scores its current population so one run cannot mix score kernels.
SCORER_VERSION = 9
# Variation semantics are versioned separately from numerical scoring. Version 5 adds
# protected stepping stones, deterministic exploration lanes, and pre-cache simplification.
EVOLUTION_VERSION = 5

MAX_CANONICAL_COPIES = 2
DUPLICATE_RETRY_LIMIT = 8
DIVERSITY_WARNING_THRESHOLD = 0.60
EXPLORATION_PROFILES = frozenset({"classic", "balanced", "aggressive"})
STEPPING_STONE_TTL = 4
STEPPING_STONE_NICHE_CAP = 2
PARAMETER_BEAM_WIDTH = 4
STAGNATION_BOOST_AFTER = 4


@dataclass(frozen=True)
class ExplorationSettings:
    """Resolved, checkpointable search-lane policy."""

    profile: str
    stepping_stone_fraction: float
    protected_parent_fraction: float
    two_edit_fraction: float
    composition_fraction: float
    insertion_fraction: float
    parameter_step_multipliers: tuple[int, ...]
    stepping_stone_ttl: int = STEPPING_STONE_TTL
    niche_cap: int = STEPPING_STONE_NICHE_CAP
    beam_width: int = PARAMETER_BEAM_WIDTH

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["parameter_step_multipliers"] = list(
            self.parameter_step_multipliers
        )
        return payload

    def with_stagnation_boost(self, generations_without_improvement: int) -> ExplorationSettings:
        if self.profile == "classic" or generations_without_improvement < STAGNATION_BOOST_AFTER:
            return self
        return dataclasses.replace(
            self,
            protected_parent_fraction=min(0.50, self.protected_parent_fraction + 0.10),
            two_edit_fraction=min(0.30, self.two_edit_fraction + 0.10),
        )


def resolve_exploration_settings(profile: str | None) -> ExplorationSettings:
    """Resolve a public profile, with missing legacy input retaining classic search."""
    resolved = profile or "classic"
    if resolved == "classic":
        return ExplorationSettings(resolved, 0.0, 0.0, 0.0, 0.0, 0.0, (1,))
    if resolved == "balanced":
        return ExplorationSettings(resolved, 0.10, 0.20, 0.10, 0.15, 0.10, (1, 2, 4))
    if resolved == "aggressive":
        return ExplorationSettings(resolved, 0.20, 0.40, 0.20, 0.15, 0.10, (1, 2, 4))
    raise ValueError(
        "exploration_profile must be 'classic', 'balanced', 'aggressive', or None"
    )


def active_scorer_backend(method: str) -> str:
    """Return the numerical kernel identity persisted beside ``SCORER_VERSION``."""
    from alphalineage.core import cpp

    if cpp.supports_native_scoring(method):
        return f"native_spearman_abi{cpp.native_abi_version()}"
    return "python"


class TrainingCancelled(RuntimeError):
    """Raised when cancellation arrives before an initial population can be committed."""


@dataclass
class GPConfig:
    population_size: int = 200
    generations: int = 25
    tournament_size: int = 3
    crossover_rate: float = 0.8
    subtree_mutation_rate: float = 0.1
    point_mutation_rate: float = 0.1
    max_depth: int = 6
    max_nodes: int = 40
    # Deprecated compatibility coefficient. It remains the per-node value when the new fields
    # are absent, so old request bodies and direct-library callers retain their exact semantics.
    parsimony: float = 1e-3
    complexity_penalty_mode: str = "per_node"
    complexity_penalty_value: float | None = None
    parameter_neighbor_fraction: float = 0.0
    validation_folds: int = 3
    # Missing/None is the compatibility path for checkpoints and direct library callers.
    exploration_profile: str | None = None
    elitism: int = 1
    ic_method: str = "spearman"
    min_names: int = 5
    horizon: int = 1
    # When a signal is traded (see ``fitness.EXECUTION_TIMINGS``). Missing means the legacy
    # same-close assumption, so stored sessions and checkpoints keep their exact meaning.
    execution: str = "close"
    min_depth: int = 2
    seed: int = 0
    time_budget_s: float | None = None
    # Operator categories the GP may draw from. ``None`` => the default pool (condition excluded).
    enabled_categories: list[str] | None = None
    # Logical formula allow-list. ``None`` keeps legacy category-wide formula selection.
    enabled_formula_names: list[str] | None = None

    def __post_init__(self) -> None:
        positive_ints = (
            "population_size",
            "generations",
            "tournament_size",
            "max_depth",
            "max_nodes",
            "min_names",
            "horizon",
            "min_depth",
            "validation_folds",
        )
        for name in positive_ints:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer, got {value!r}")

        upper_bounds = {
            "population_size": MAX_POPULATION_SIZE,
            "generations": MAX_GENERATIONS,
            "max_depth": MAX_TREE_DEPTH,
            "max_nodes": MAX_TREE_NODES,
            "min_names": MAX_MIN_NAMES,
            "horizon": MAX_HORIZON,
            "min_depth": MAX_TREE_DEPTH,
            "validation_folds": 10,
        }
        for name, upper in upper_bounds.items():
            value = getattr(self, name)
            if value > upper:
                raise ValueError(f"{name} must be at most {upper}, got {value!r}")
        if self.population_size * self.generations > MAX_SEARCH_EVALUATIONS:
            raise ValueError(
                "population_size * generations must not exceed "
                f"{MAX_SEARCH_EVALUATIONS:,} evaluations"
            )

        if isinstance(self.elitism, bool) or not isinstance(self.elitism, int):
            raise ValueError(f"elitism must be an integer, got {self.elitism!r}")
        if not 0 <= self.elitism < self.population_size:
            raise ValueError("elitism must be in [0, population_size)")
        if self.tournament_size > self.population_size:
            raise ValueError("tournament_size must not exceed population_size")
        if self.min_depth > self.max_depth:
            raise ValueError("min_depth must not exceed max_depth")
        if self.max_nodes < self.min_depth:
            raise ValueError("max_nodes must be at least min_depth")
        if self.validation_folds < 2:
            raise ValueError("validation_folds must be at least 2")
        if (
            self.exploration_profile is not None
            and self.exploration_profile not in EXPLORATION_PROFILES
        ):
            raise ValueError(
                "exploration_profile must be 'classic', 'balanced', "
                "'aggressive', or None"
            )

        for name in (
            "crossover_rate",
            "subtree_mutation_rate",
            "point_mutation_rate",
            "parameter_neighbor_fraction",
        ):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or not 0.0 <= float(value) <= 1.0
            ):
                raise ValueError(f"{name} must be a finite number in [0, 1], got {value!r}")

        if (
            isinstance(self.parsimony, bool)
            or not isinstance(self.parsimony, (int, float))
            or not math.isfinite(float(self.parsimony))
            or self.parsimony < 0
        ):
            raise ValueError(
                f"parsimony must be a finite non-negative number, got {self.parsimony!r}"
            )
        if self.complexity_penalty_mode not in COMPLEXITY_PENALTY_MODES:
            raise ValueError(
                "complexity_penalty_mode must be 'per_node' or 'normalized_budget'"
            )
        if self.complexity_penalty_value is not None and (
            isinstance(self.complexity_penalty_value, bool)
            or not isinstance(self.complexity_penalty_value, (int, float))
            or not math.isfinite(float(self.complexity_penalty_value))
            or self.complexity_penalty_value < 0
        ):
            raise ValueError(
                "complexity_penalty_value must be a finite non-negative number or None"
            )
        validate_execution(self.execution)
        if self.ic_method not in {"pearson", "spearman"}:
            raise ValueError("ic_method must be 'pearson' or 'spearman'")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise ValueError(f"seed must be an integer, got {self.seed!r}")
        if not -(2**63) <= self.seed < 2**63:
            raise ValueError("seed must fit in a signed 64-bit integer")
        if self.time_budget_s is not None and (
            isinstance(self.time_budget_s, bool)
            or not isinstance(self.time_budget_s, (int, float))
            or not math.isfinite(float(self.time_budget_s))
            or self.time_budget_s < 0
        ):
            raise ValueError(
                f"time_budget_s must be a finite non-negative number or None, "
                f"got {self.time_budget_s!r}"
            )
        if self.time_budget_s is not None and self.time_budget_s > MAX_TIME_BUDGET_S:
            raise ValueError(f"time_budget_s must be at most {MAX_TIME_BUDGET_S}")
        if self.enabled_categories is not None:
            if not isinstance(self.enabled_categories, list) or any(
                not isinstance(category, str) or not category.strip()
                for category in self.enabled_categories
            ):
                raise ValueError("enabled_categories must be a list of non-empty strings or None")
            if len(self.enabled_categories) > 64 or any(
                len(category) > 64 for category in self.enabled_categories
            ):
                raise ValueError("enabled_categories may contain at most 64 names of 64 characters")
            if len(set(self.enabled_categories)) != len(self.enabled_categories):
                raise ValueError("enabled_categories must not contain duplicates")
        if self.enabled_formula_names is not None:
            if not isinstance(self.enabled_formula_names, list) or any(
                not isinstance(name, str) or not name.strip()
                for name in self.enabled_formula_names
            ):
                raise ValueError(
                    "enabled_formula_names must be a list of non-empty strings or None"
                )
            if len(self.enabled_formula_names) > 128 or any(
                len(name) > 64 for name in self.enabled_formula_names
            ):
                raise ValueError(
                    "enabled_formula_names may contain at most 128 names of 64 characters"
                )
            if len(set(self.enabled_formula_names)) != len(self.enabled_formula_names):
                raise ValueError("enabled_formula_names must not contain duplicates")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def resolved_complexity_penalty_value(self) -> float:
        """Compatibility-resolved value persisted through the scorer call."""
        if self.complexity_penalty_value is not None:
            return float(self.complexity_penalty_value)
        if self.complexity_penalty_mode == "normalized_budget":
            return DEFAULT_NORMALIZED_COMPLEXITY_PENALTY
        return float(self.parsimony)

    @property
    def resolved_exploration_profile(self) -> str:
        return resolve_exploration_settings(self.exploration_profile).profile

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> GPConfig:
        if not isinstance(data, dict):
            raise ValueError("GP config must be an object")
        names = {f.name for f in dataclasses.fields(cls)}
        unknown = sorted(set(data) - names)
        if unknown:
            raise ValueError(f"unknown GP config field(s): {', '.join(unknown)}")
        return cls(**data)


@dataclass
class Individual:
    tree: Node
    fitness: float
    metrics: dict[str, float] = field(default_factory=dict)
    birth_generation: int = 0
    stepping_stone_ancestors: tuple[str, ...] = ()


# --- type-aware tree surgery -----------------------------------------------------
def iter_positions(tree: Node, root_type: DType) -> list[Position]:
    """Every node with the type its parent slot requires (root slot = ``root_type``)."""
    out: list[Position] = []

    def walk(node: Node, path: tuple[int, ...], required: DType) -> None:
        out.append((path, required, node))
        prim = node.primitive
        if prim.kind is Kind.OPERATOR:
            for i, child in enumerate(node.children):
                walk(child, (*path, i), prim.arg_types[i])

    walk(tree, (), root_type)
    return out


def iter_variation_positions(tree: Node, root_type: DType) -> list[Position]:
    """Structural-edit positions, treating policy-owned parameter children as atomic."""
    out: list[Position] = []

    def walk(node: Node, path: tuple[int, ...], required: DType) -> None:
        out.append((path, required, node))
        prim = node.primitive
        if prim.kind is not Kind.OPERATOR:
            return
        inputs = (prim.macro_policy or {}).get("inputs") or []
        for index, child in enumerate(node.children):
            item = (
                inputs[index]
                if index < len(inputs) and isinstance(inputs[index], dict)
                else {}
            )
            if item.get("role") == "parameter":
                continue
            walk(child, (*path, index), prim.arg_types[index])

    walk(tree, (), root_type)
    return out


def replace_at(tree: Node, path: tuple[int, ...], new: Node) -> Node:
    """Return a copy of ``tree`` with the subtree at ``path`` replaced by ``new``."""
    if not path:
        return new
    children = list(tree.children)
    children[path[0]] = replace_at(children[path[0]], path[1:], new)
    return Node(tree.name, tuple(children), tree.value)


def validate_seed(
    tree: Node, *, max_depth: int, max_nodes: int, root_type: DType = DType.SIGNAL
) -> Node:
    """A seed for the GP population is data, never code: every node must be a registered
    primitive, the root must satisfy ``root_type``, and depth/size must fit the config.

    Raises ``ValueError`` naming the offending primitive or the violated bound.
    """
    validate(tree)  # raises InvalidTree naming the offending primitive
    if not is_subtype(tree.out_type, root_type):
        raise ValueError(
            f"seed root must produce {root_type.name}, got {tree.out_type.name} ({tree.name!r})"
        )
    # Limits apply after formula expansion. Otherwise a one-node saved formula could smuggle an
    # arbitrarily deep/large built-in expression into a tightly bounded GP run.
    compact_depth = tree.depth()
    compact_size = tree.size()
    if compact_depth > max_depth or compact_size > max_nodes:
        raise ValueError(
            f"seed compact expression exceeds depth/nodes {max_depth}/{max_nodes}"
        )
    expansion_depth = _expansion_depth_limit(tree, max_depth)
    expand_all(tree, max_depth=expansion_depth, max_nodes=max_nodes)
    return tree


def _expansion_depth_limit(tree: Node, configured: int) -> int:
    """Give managed catalog calls their published intrinsic depth, up to the global guard.

    The generator's compact tree still obeys the user's depth budget. This allowance only keeps a
    single atomic indicator (RSI/KDJ/ADX/MFI) usable at the default depth six; expanded distinct
    nodes still spend the ordinary node budget, so nesting cannot evade parsimony.
    """
    managed = any(
        node.primitive.macro_body is not None
        and (node.primitive.macro_policy or {}).get("catalog_revision") is not None
        for node in tree.iter_nodes()
    )
    return MAX_TREE_DEPTH if managed else configured


class GP:
    """A genetic-programming run over a panel."""

    def __init__(
        self,
        config: GPConfig,
        panel: Panel,
        fwd: pd.DataFrame | None = None,
        *,
        root_type: DType = DType.SIGNAL,
        recorder: Any | None = None,
        allowed_operators: set[str] | None = None,
        workers: int = 1,
        memory_budget_bytes: int | None = None,
    ) -> None:
        if (
            isinstance(workers, bool)
            or not isinstance(workers, int)
            or not 1 <= workers <= MAX_TRAINING_WORKERS
        ):
            raise ValueError(f"workers must be an integer in [1, {MAX_TRAINING_WORKERS}]")
        if memory_budget_bytes is not None and (
            isinstance(memory_budget_bytes, bool)
            or not isinstance(memory_budget_bytes, int)
            or memory_budget_bytes <= 0
        ):
            raise ValueError("memory_budget_bytes must be a positive integer or None")
        self.config = config
        self.panel = panel
        self.fwd = (
            fwd
            if fwd is not None
            else forward_returns(panel, config.horizon, config.execution)
        )
        self.root_type = root_type
        # Optional lineage recorder (duck-typed: on_init(trees), on_generation(gen, entries)).
        self.recorder = recorder
        # Operator name allow-set (None => default pool, condition category excluded).
        self.allowed_operators = allowed_operators
        self.workers = workers
        self.memory_budget_bytes = memory_budget_bytes
        self.scorer_backend = active_scorer_backend(config.ic_method)
        self.rng = random.Random(config.seed)
        self.generator = RandomTreeGenerator(
            self.rng,
            max_depth=config.max_depth,
            max_nodes=config.max_nodes,
            root_type=root_type,
            allowed_operators=allowed_operators,
        )
        self.population: list[Individual] = []
        self.generation = 0
        self.history: list[dict[str, Any]] = []
        self._cache: dict[str, tuple[float, dict[str, float]]] = {}
        # Trials counted before this object's cache existed (resumes, invalidated caches).
        # Monotone by construction: it only ever grows, so deflation never softens.
        self._prior_trials = 0
        self._requires_rescore = False
        self._pending_rescore_trees: list[Node] = []
        self.termination_reason = "completed"
        self._champion_key: str | None = None
        self._champion_age = 0
        self._best_ever = -math.inf
        self._generations_since_improvement = 0
        self._last_novel_offspring = 0
        self._last_parameter_neighbors = 0
        self._last_formula_default_injections = 0
        self._last_exploration_diagnostics: dict[str, Any] = {}
        self._parameter_beam_cursor = 0
        self._parameter_frontier_cursor = 0
        self._parameter_lane_cursor = 0
        self._protected_parent_cursor = 0
        self._formula_move_counts: dict[str, Counter[str]] = {}
        self._pre_cache_simplifications = 0
        self._complexity_rejections = 0
        self._managed_formula_defaults: list[Node] = []
        self._pending_formula_defaults: list[Node] = []
        self._formula_validation_best: dict[str, float] = {}

    @property
    def trial_count(self) -> int:
        """Distinct factors scored so far - the deflation's number of trials."""
        return self._prior_trials + len(self._cache)

    def searched_individuals(self) -> list[Individual]:
        """All distinct scored trees in deterministic first-seen order."""
        return [
            Individual(from_json(key), fitness, dict(metrics), 0)
            for key, (fitness, metrics) in self._cache.items()
        ]

    def _base_exploration_settings(self) -> ExplorationSettings:
        return resolve_exploration_settings(self.config.exploration_profile)

    def _generation_exploration_settings(self) -> ExplorationSettings:
        return self._base_exploration_settings().with_stagnation_boost(
            self._generations_since_improvement
        )

    @staticmethod
    def _cache_key(tree: Node) -> str:
        return to_json(simplify(tree))

    def _prepare_tree(self, tree: Node) -> Node:
        """Normalize a candidate before identity, feasibility, scoring, or persistence."""
        prepared = simplify(tree)
        if prepared != tree:
            self._pre_cache_simplifications += 1
        return prepared

    @staticmethod
    def _formula_calls(tree: Node) -> list[Node]:
        """Managed formula calls in stable depth-first expression order."""
        return [
            node
            for node in tree.iter_nodes()
            if node.primitive.macro_body is not None
            and (node.primitive.macro_policy or {}).get("catalog_revision") is not None
        ]

    @staticmethod
    def _parameter_tuple(call: Node) -> tuple[float | int | None, ...]:
        inputs = (call.primitive.macro_policy or {}).get("inputs") or []
        return tuple(
            call.children[index].value
            for index, item in enumerate(inputs)
            if index < len(call.children)
            and isinstance(item, dict)
            and item.get("role") == "parameter"
        )

    def formula_exploration(self) -> dict[str, dict[str, float | int | None]]:
        """Per-managed-formula coverage over every distinct searched compact tree."""
        names = list(
            dict.fromkeys(call.name for call in self._managed_formula_defaults)
        )
        calls: Counter[str] = Counter()
        parameter_values: dict[str, set[tuple[float | int | None, ...]]] = {}
        best_training: dict[str, float] = {}
        for key, (fitness, _metrics) in self._cache.items():
            for call in self._formula_calls(from_json(key)):
                if call.name not in names:
                    names.append(call.name)
                calls[call.name] += 1
                parameter_values.setdefault(call.name, set()).add(
                    self._parameter_tuple(call)
                )
                best_training[call.name] = max(
                    best_training.get(call.name, -math.inf), float(fitness)
                )
        pending = {to_json(call) for call in self._pending_formula_defaults}
        beam_members: Counter[str] = Counter()
        for individual in sorted(
            self.population,
            key=lambda candidate: candidate.fitness,
            reverse=True,
        ):
            for name in dict.fromkeys(
                call.name for call in self._formula_calls(individual.tree)
            ):
                if beam_members[name] < PARAMETER_BEAM_WIDTH:
                    beam_members[name] += 1
        return {
            name: {
                "calls_searched": int(calls[name]),
                "distinct_parameter_tuples": len(parameter_values.get(name, set())),
                "best_training_score": (
                    best_training[name]
                    if name in best_training and math.isfinite(best_training[name])
                    else None
                ),
                "best_validation_score": self._formula_validation_best.get(name),
                "default_evaluated": int(
                    any(
                        call.name == name and to_json(call) in self._cache
                        for call in self._managed_formula_defaults
                    )
                ),
                "default_pending": int(
                    any(
                        call.name == name and to_json(call) in pending
                        for call in self._managed_formula_defaults
                    )
                ),
                "beam_members": int(beam_members[name]),
                "neighbor_moves_searched": int(
                    sum(self._formula_move_counts.get(name, Counter()).values())
                ),
                "diagonal_moves_searched": int(
                    self._formula_move_counts.get(name, Counter()).get(
                        "diagonal", 0
                    )
                ),
            }
            for name in names
        }

    def record_formula_validation_scores(
        self, candidates: Sequence[tuple[Node, float]]
    ) -> None:
        """Attach validation evidence without altering fitness or the population.

        The caller may save a checkpoint again after validation to retain these values. The
        latest generation history snapshot is refreshed in place for round reporting.
        """
        for tree, score in candidates:
            value = float(score)
            if not math.isfinite(value):
                continue
            for call in self._formula_calls(tree):
                self._formula_validation_best[call.name] = max(
                    self._formula_validation_best.get(call.name, -math.inf), value
                )
        if self.history and int(self.history[-1].get("generation", -1)) == self.generation:
            self.history[-1]["formula_exploration"] = self.formula_exploration()

    def _parameter_distance_from_defaults(self) -> dict[str, dict[str, float | int]]:
        """Summarize searched parameter displacement in declared policy-step units."""
        distances: dict[str, list[float]] = {}
        for key in self._cache:
            for call in self._formula_calls(from_json(key)):
                inputs = (call.primitive.macro_policy or {}).get("inputs") or []
                distance = 0.0
                found = False
                for index, item in enumerate(inputs):
                    child_value = (
                        call.children[index].value
                        if index < len(call.children)
                        else None
                    )
                    if (
                        index >= len(call.children)
                        or not isinstance(item, dict)
                        or item.get("role") != "parameter"
                        or item.get("default") is None
                        or not isinstance(item.get("tuning"), dict)
                        or child_value is None
                    ):
                        continue
                    step = float(item["tuning"].get("step", 1.0))
                    if not math.isfinite(step) or step <= 0.0:
                        continue
                    distance += abs(
                        float(child_value) - float(item["default"])
                    ) / step
                    found = True
                if found:
                    distances.setdefault(call.name, []).append(distance)
        return {
            name: {
                "calls": len(values),
                "distinct_distances": len(set(values)),
                "mean_l1_steps": float(sum(values) / len(values)),
                "max_l1_steps": float(max(values)),
            }
            for name, values in sorted(distances.items())
        }

    # --- scoring -----------------------------------------------------------------
    def _score(self, tree: Node) -> tuple[float, dict[str, float]]:
        tree = self._prepare_tree(tree)
        key = to_json(tree)
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        self._ensure_scorer_backend()
        result = score_trees(
            [tree],
            self.panel,
            self.fwd,
            method=self.config.ic_method,
            parsimony=self.config.parsimony,
            complexity_penalty_mode=self.config.complexity_penalty_mode,
            complexity_penalty_value=self.config.complexity_penalty_value,
            max_nodes=self.config.max_nodes,
            min_names=self.config.min_names,
            workers=1,
            memory_budget_bytes=self.memory_budget_bytes,
        )[0]
        self._cache[key] = result
        return result

    def _individual(self, tree: Node, *, birth_generation: int | None = None) -> Individual:
        prepared = self._prepare_tree(tree)
        fitness, metrics = self._score(prepared)
        return Individual(
            prepared,
            fitness,
            metrics,
            self.generation if birth_generation is None else birth_generation,
        )

    def _notify_scoring(self, phase: str, done: int, total: int, started: float) -> None:
        callback = getattr(self.recorder, "on_scoring", None)
        if callback is not None:
            elapsed = max(time.monotonic() - started, 1e-9)
            callback(
                phase=phase,
                generation=self.generation,
                done=done,
                total=total,
                factors_per_second=done / elapsed,
            )

    def _individuals(
        self,
        trees: Sequence[Node],
        *,
        phase: str,
        stop: Callable[[], bool] | None = None,
        birth_generation: int | None = None,
    ) -> list[Individual]:
        """Score ``trees`` as one deterministic transaction.

        Cache hits and duplicates retain input order.  Unique misses are evaluated in bounded
        chunks, but only committed to the coordinator-owned cache after every chunk succeeds.
        Thus cancellation cannot leave a half-generation counted as searched.
        """
        prepared_trees = [self._prepare_tree(tree) for tree in trees]
        keys = [to_json(tree) for tree in prepared_trees]
        missing: dict[str, Node] = {}
        for key, tree in zip(keys, prepared_trees, strict=True):
            if key not in self._cache and key not in missing:
                missing[key] = tree

        pending = list(missing.items())
        if pending:
            self._ensure_scorer_backend()
        staged: dict[str, tuple[float, dict[str, float]]] = {}
        started = time.monotonic()
        total = len(pending)
        self._notify_scoring(phase, 0, total, started)
        # More than two waves gives cancellation/progress useful granularity while keeping native
        # call overhead negligible on the small default population.
        chunk_size = max(1, min(64, self.workers * 2))
        for offset in range(0, total, chunk_size):
            if stop is not None and stop():
                raise TrainingCancelled("training cancelled before scoring completed")
            chunk = pending[offset : offset + chunk_size]
            scored = score_trees(
                [tree for _, tree in chunk],
                self.panel,
                self.fwd,
                method=self.config.ic_method,
                parsimony=self.config.parsimony,
                complexity_penalty_mode=self.config.complexity_penalty_mode,
                complexity_penalty_value=self.config.complexity_penalty_value,
                max_nodes=self.config.max_nodes,
                min_names=self.config.min_names,
                workers=min(self.workers, len(chunk)),
                memory_budget_bytes=self.memory_budget_bytes,
            )
            for (key, _), result in zip(chunk, scored, strict=True):
                staged[key] = result
            self._notify_scoring(phase, min(offset + len(chunk), total), total, started)
            if stop is not None and stop():
                raise TrainingCancelled("training cancelled before scoring completed")

        self._cache.update(staged)
        born = self.generation if birth_generation is None else birth_generation
        return [
            Individual(tree, *self._cache[key], born)
            for tree, key in zip(prepared_trees, keys, strict=True)
        ]

    def _ensure_scorer_backend(self) -> None:
        """Refuse a mid-run evaluator switch instead of mixing numerical kernels."""
        current = active_scorer_backend(self.config.ic_method)
        if current != self.scorer_backend:
            raise RuntimeError(
                "training evaluator changed during the run; restart or continue from the last "
                "checkpoint so the population can be rescored consistently"
            )

    # --- selection & variation ---------------------------------------------------
    def _tournament(
        self, eligible_indices: Sequence[int] | None = None
    ) -> tuple[Individual, int]:
        # Index-based selection (same RNG draws as choice(population)) so we can record parents.
        n = len(self.population)
        pool: Sequence[int] = range(n) if eligible_indices is None else eligible_indices
        if not pool:
            pool = range(n)
        idxs = [self.rng.choice(pool) for _ in range(self.config.tournament_size)]
        best = max(idxs, key=lambda i: self.population[i].fitness)
        return self.population[best], best

    def _fits_complexity(self, tree: Node) -> bool:
        tree = simplify(tree)
        try:
            validate(tree)
            if tree.depth() > self.config.max_depth or tree.size() > self.config.max_nodes:
                self._complexity_rejections += 1
                return False
            expand_all(
                tree,
                max_depth=_expansion_depth_limit(tree, self.config.max_depth),
                max_nodes=self.config.max_nodes,
            )
        except (InvalidOperator, ValueError):
            self._complexity_rejections += 1
            return False
        return True

    @staticmethod
    def _canonical_key(tree: Node) -> str:
        """Expanded-expression identity used for population diversity limits."""
        return to_json(expand_all(simplify(tree)))

    def _accept_under_copy_limit(self, tree: Node, counts: Counter[str]) -> bool:
        key = self._canonical_key(tree)
        if counts[key] >= MAX_CANONICAL_COPIES:
            return False
        counts[key] += 1
        return True

    @staticmethod
    def _managed_default_call(name: str) -> Node | None:
        primitive = OPERATORS.get(name)
        if primitive is None or primitive.macro_body is None:
            return None
        policy = primitive.macro_policy or {}
        if policy.get("catalog_revision") is None or policy.get("status") != "active":
            return None
        inputs = policy.get("inputs") or []
        children: list[Node] = []
        for index, arg_type in enumerate(primitive.arg_types):
            item = inputs[index] if index < len(inputs) else None
            if (
                not isinstance(item, dict)
                or item.get("role") != "parameter"
                or item.get("default") is None
            ):
                return None
            default = item["default"]
            if arg_type is DType.WINDOW:
                children.append(Node("window", value=int(default)))
            elif arg_type is DType.SCALAR:
                children.append(Node("const", value=float(default)))
            else:
                return None
        call = Node(name, tuple(children))
        validate(call)
        return call

    def _configured_formula_defaults(self) -> list[Node]:
        """Published defaults for explicitly enabled managed formulas, in catalog order."""
        selected = self.config.enabled_formula_names
        if selected is None:
            return []
        logical_names = set(selected)
        defaults: list[Node] = []
        matched: set[str] = set()
        for primitive in OPERATORS.values():
            if self.allowed_operators is not None and primitive.name not in self.allowed_operators:
                continue
            logical = next(
                (
                    name
                    for name in logical_names
                    if primitive.name == name or primitive.name.startswith(f"{name}__r")
                ),
                None,
            )
            if self.allowed_operators is None and logical is None:
                continue
            call = self._managed_default_call(primitive.name)
            if call is None:
                continue
            if not self._fits_complexity(call):
                raise ValueError(
                    f"enabled managed formula {logical or primitive.name!r} exceeds "
                    "the configured expanded depth/node limits"
                )
            defaults.append(call)
            if logical is not None:
                matched.add(logical)
        if self.allowed_operators is None:
            missing = sorted(logical_names - matched)
            if missing:
                raise ValueError(
                    f"enabled managed formula(s) unavailable: {', '.join(missing)}"
                )
        return defaults

    def _take_pending_formula_default(self, counts: Counter[str]) -> Node | None:
        """Take the next admissible default without letting one blocked item starve others."""
        for _ in range(len(self._pending_formula_defaults)):
            candidate = self._prepare_tree(self._pending_formula_defaults.pop(0))
            if self._cache_key(candidate) in self._cache:
                continue
            if self._accept_under_copy_limit(candidate, counts):
                return candidate
            self._pending_formula_defaults.append(candidate)
        return None

    def _random_immigrant(self, counts: Counter[str]) -> Node:
        """Generate a bounded, canonical-novel immigrant using the run's single RNG."""
        limit = max(1_000, self.config.population_size * 200)
        for attempt in range(limit):
            candidate = self.generator.generate(
                grow=(attempt % 2 == 0),
                max_depth=self.config.max_depth,
            )
            candidate = self._prepare_tree(candidate)
            if not self._fits_complexity(candidate):
                continue
            if self._accept_under_copy_limit(candidate, counts):
                return candidate
        raise GenerationError(
            "could not generate a population within the canonical duplicate limit"
        )

    @staticmethod
    def _node_at(tree: Node, path: tuple[int, ...]) -> Node:
        current = tree
        for index in path:
            current = current.children[index]
        return current

    def _mutate_policy_call(self, tree: Node, path: tuple[int, ...]) -> Node:
        call = self._node_at(tree, path)
        policy = call.primitive.macro_policy or {}
        inputs = policy.get("inputs") or []
        tunable = [
            index
            for index, item in enumerate(inputs)
            if index < len(call.children)
            and isinstance(item, dict)
            and isinstance(item.get("tuning"), dict)
            and item["tuning"].get("enabled", True)
        ]
        if not tunable:
            return tree
        index = self.rng.choice(tunable)
        tuning = inputs[index]["tuning"]
        child = call.children[index]
        if child.value is None:
            return tree
        step = tuning["step"]
        radius = max(1, int(tuning.get("radius", 1)))
        deltas = [step * offset for offset in range(-radius, radius + 1) if offset]
        self.rng.shuffle(deltas)
        for delta in deltas:
            value = child.value + delta
            if value < tuning["min"] or value > tuning["max"]:
                continue
            value = int(value) if child.name == "window" else float(value)
            children = list(call.children)
            children[index] = Node(child.name, value=value)
            candidate = replace_at(tree, path, Node(call.name, tuple(children)))
            if self._fits_complexity(candidate):
                return candidate
        return tree

    def _policy_move_candidates(
        self,
        tree: Node,
        path: tuple[int, ...],
        *,
        multipliers: Sequence[int],
        include_diagonal: bool,
    ) -> list[tuple[Node, str]]:
        """Enumerate bounded parameter moves without ever materializing an invalid call."""
        call = self._node_at(tree, path)
        inputs = (call.primitive.macro_policy or {}).get("inputs") or []
        tunable = [
            index
            for index, item in enumerate(inputs)
            if index < len(call.children)
            and isinstance(item, dict)
            and isinstance(item.get("tuning"), dict)
            and item["tuning"].get("enabled", True)
            and call.children[index].value is not None
        ]
        moves: list[tuple[Node, str]] = []
        seen: set[str] = set()

        def append_move(changes: dict[int, float | int], kind: str) -> None:
            children = list(call.children)
            for index, value in changes.items():
                child = children[index]
                children[index] = Node(
                    child.name,
                    value=int(value) if child.name == "window" else float(value),
                )
            candidate = self._prepare_tree(
                replace_at(tree, path, Node(call.name, tuple(children)))
            )
            key = self._cache_key(candidate)
            if candidate == tree or key in seen or not self._fits_complexity(candidate):
                return
            seen.add(key)
            moves.append((candidate, kind))

        signed_multipliers = [
            signed
            for multiplier in multipliers
            for signed in (multiplier, -multiplier)
        ]
        for index in tunable:
            item = inputs[index]
            tuning = item["tuning"]
            child = call.children[index]
            assert child.value is not None
            for multiplier in signed_multipliers:
                value = child.value + tuning["step"] * multiplier
                if value < tuning["min"] or value > tuning["max"]:
                    continue
                append_move({index: value}, "axis")

        if include_diagonal and len(tunable) >= 2:
            for left_offset in range(len(tunable) - 1):
                for right_offset in range(left_offset + 1, len(tunable)):
                    left_index = tunable[left_offset]
                    right_index = tunable[right_offset]
                    left_tuning = inputs[left_index]["tuning"]
                    right_tuning = inputs[right_index]["tuning"]
                    left_child = call.children[left_index]
                    right_child = call.children[right_index]
                    assert left_child.value is not None and right_child.value is not None
                    for multiplier in multipliers:
                        for left_sign, right_sign in (
                            (1, 1),
                            (-1, -1),
                            (1, -1),
                            (-1, 1),
                        ):
                            left_value = (
                                left_child.value
                                + left_tuning["step"] * multiplier * left_sign
                            )
                            right_value = (
                                right_child.value
                                + right_tuning["step"] * multiplier * right_sign
                            )
                            if (
                                left_value < left_tuning["min"]
                                or left_value > left_tuning["max"]
                                or right_value < right_tuning["min"]
                                or right_value > right_tuning["max"]
                            ):
                                continue
                            append_move(
                                {
                                    left_index: left_value,
                                    right_index: right_value,
                                },
                                "diagonal",
                            )
        return moves

    @staticmethod
    def _tunable_formula_paths(tree: Node) -> list[tuple[int, ...]]:
        paths: list[tuple[int, ...]] = []

        def walk(node: Node, path: tuple[int, ...]) -> None:
            inputs = (node.primitive.macro_policy or {}).get("inputs") or []
            if any(
                isinstance(item, dict)
                and isinstance(item.get("tuning"), dict)
                and item["tuning"].get("enabled", True)
                for item in inputs
            ):
                paths.append(path)
            for index, child in enumerate(node.children):
                walk(child, (*path, index))

        walk(tree, ())
        return paths

    def _parameter_beam_sources(
        self,
        *,
        eligible_indices: set[int] | None = None,
        width: int = PARAMETER_BEAM_WIDTH,
    ) -> list[tuple[str, int, Node, tuple[int, ...]]]:
        """Best live formula calls per catalog niche, with stable first-seen tie-breaking."""
        formula_order = {
            call.name: index for index, call in enumerate(self._managed_formula_defaults)
        }
        sources: list[tuple[str, int, Node, tuple[int, ...]]] = []
        per_formula: Counter[str] = Counter()
        seen_tuples: set[tuple[str, tuple[float | int | None, ...]]] = set()
        ordered_indices = sorted(
            range(len(self.population)),
            key=lambda index: (-self.population[index].fitness, index),
        )
        for index in ordered_indices:
            if eligible_indices is not None and index not in eligible_indices:
                continue
            individual = self.population[index]
            for path in self._tunable_formula_paths(individual.tree):
                call = self._node_at(individual.tree, path)
                if per_formula[call.name] >= width:
                    continue
                identity = (call.name, self._parameter_tuple(call))
                if identity in seen_tuples:
                    continue
                seen_tuples.add(identity)
                per_formula[call.name] += 1
                sources.append((call.name, index, individual.tree, path))
        return sorted(
            sources,
            key=lambda item: (
                formula_order.get(item[0], len(formula_order)),
                item[0],
                -self.population[item[1]].fitness,
                item[1],
                item[3],
            ),
        )

    def _parameter_frontier_sources(
        self,
        *,
        eligible_indices: set[int] | None = None,
    ) -> list[tuple[str, int, Node, tuple[int, ...]]]:
        """All distinct live tuples in catalog/tuple order, independent of quality rank."""
        formula_order = {
            call.name: index for index, call in enumerate(self._managed_formula_defaults)
        }
        sources: list[tuple[str, int, Node, tuple[int, ...]]] = []
        seen: set[tuple[str, tuple[float | int | None, ...]]] = set()
        for index, individual in enumerate(self.population):
            if eligible_indices is not None and index not in eligible_indices:
                continue
            for path in self._tunable_formula_paths(individual.tree):
                call = self._node_at(individual.tree, path)
                identity = (call.name, self._parameter_tuple(call))
                if identity in seen:
                    continue
                seen.add(identity)
                sources.append((call.name, index, individual.tree, path))
        return sorted(
            sources,
            key=lambda item: (
                formula_order.get(item[0], len(formula_order)),
                item[0],
                repr(self._parameter_tuple(self._node_at(item[2], item[3]))),
                item[1],
                item[3],
            ),
        )

    def _parameter_neighbor(
        self,
        *,
        eligible_indices: set[int] | None = None,
    ) -> tuple[Node, list[int], str] | None:
        """Take the next unseen beam/frontier move in a deterministic round-robin."""
        settings = self._generation_exploration_settings()
        lane = "beam" if self._parameter_lane_cursor % 2 == 0 else "frontier"
        self._parameter_lane_cursor += 1
        sources = (
            self._parameter_beam_sources(
                eligible_indices=eligible_indices,
                width=settings.beam_width,
            )
            if lane == "beam"
            else self._parameter_frontier_sources(
                eligible_indices=eligible_indices
            )
        )
        if not sources:
            return None
        candidates: list[tuple[Node, int, str, str]] = []
        for name, parent_index, tree, path in sources:
            moves = self._policy_move_candidates(
                tree,
                path,
                multipliers=(
                    (1,)
                    if lane == "beam"
                    else settings.parameter_step_multipliers
                ),
                include_diagonal=lane == "frontier",
            )
            candidates.extend(
                (candidate, parent_index, name, kind)
                for candidate, kind in moves
            )
        if not candidates:
            return None
        cursor_name = (
            "_parameter_beam_cursor"
            if lane == "beam"
            else "_parameter_frontier_cursor"
        )
        cursor = int(getattr(self, cursor_name))
        for offset in range(len(candidates)):
            candidate, parent_index, name, kind = candidates[
                (cursor + offset) % len(candidates)
            ]
            if self._cache_key(candidate) in self._cache:
                continue
            setattr(self, cursor_name, cursor + offset + 1)
            return (
                candidate,
                [parent_index],
                f"parameter_{lane}:{name}:{kind}",
            )
        setattr(self, cursor_name, cursor + len(candidates))
        return None

    def _mutate_formula_coefficient(
        self, tree: Node, path: tuple[int, ...], node: Node
    ) -> Node | None:
        """Locally tune a scalar weighting a formula subtree; None means not applicable."""
        if node.name != "const" or not path or path[-1] != 1:
            return None
        parent = self._node_at(tree, path[:-1])
        if parent.name != "mul_scalar" or parent.children[0].primitive.macro_body is None:
            return None
        # Scaling a standalone factor does not alter its ranks/IC and only wastes a trial. Weight
        # tuning is meaningful when the weighted formula is one term in an add/sub combination.
        if len(path) < 2:
            return tree
        combination = self._node_at(tree, path[:-2])
        if combination.name not in {"add", "sub"}:
            return tree
        assert node.value is not None
        deltas = [-0.25, 0.25]
        self.rng.shuffle(deltas)
        for delta in deltas:
            value = max(-3.0, min(3.0, float(node.value) + delta))
            if value == node.value:
                continue
            candidate = replace_at(tree, path, Node("const", value=value))
            if self._fits_complexity(candidate):
                return candidate
        return tree

    def _crossover(self, a: Node, b: Node) -> Node:
        path, required, _ = self.rng.choice(iter_variation_positions(a, self.root_type))
        donors = [
            sub
            for (_, _, sub) in iter_variation_positions(b, self.root_type)
            if is_subtype(sub.out_type, required)
        ]
        if not donors:
            return a
        for _ in range(8):
            child = replace_at(a, path, self.rng.choice(donors))
            if self._fits_complexity(child):
                return self._prepare_tree(child)
        return a

    def _composition_crossover(self, a: Node, b: Node) -> Node:
        """Compose both complete parents, preserving each as an intact building block."""
        choices = [
            name
            for name in ("add", "sub", "mul")
            if name in OPERATORS
            and operator_allowed(OPERATORS[name], self.allowed_operators)
            and is_subtype(a.out_type, OPERATORS[name].arg_types[0])
            and is_subtype(b.out_type, OPERATORS[name].arg_types[1])
            and is_subtype(OPERATORS[name].out_type, self.root_type)
        ]
        self.rng.shuffle(choices)
        for name in choices:
            candidate = self._prepare_tree(Node(name, (a, b)))
            if self._fits_complexity(candidate):
                return candidate
        return a

    def _subtree_mutation(self, tree: Node) -> Node:
        path, required, sub = self.rng.choice(iter_variation_positions(tree, self.root_type))
        depth_budget = self.config.max_depth - len(path)
        node_budget = self.config.max_nodes - (tree.size() - sub.size())
        if depth_budget < 1 or node_budget < 1:
            return tree
        try:
            fresh = self.generator.grow_subtree(
                required, max_depth=depth_budget, max_nodes=node_budget, grow=True
            )
        except GenerationError:  # budget too small to close this typed hole; leave the tree as-is
            return tree
        candidate = replace_at(tree, path, fresh)
        return self._prepare_tree(candidate) if self._fits_complexity(candidate) else tree

    def _insertion_mutation(self, tree: Node) -> Node:
        """Wrap a chosen subtree while preserving it as an input to the new operator."""
        path, required, subtree = self.rng.choice(
            iter_variation_positions(tree, self.root_type)
        )
        depth_budget = self.config.max_depth - len(path)
        node_budget = self.config.max_nodes - (tree.size() - subtree.size())
        if depth_budget < 2 or node_budget <= subtree.size():
            return tree
        operators = [
            primitive
            for primitive in OPERATORS.values()
            if primitive.macro_body is None
            and primitive.arity > 0
            and operator_allowed(primitive, self.allowed_operators)
            and is_subtype(primitive.out_type, required)
            and any(
                is_subtype(subtree.out_type, arg_type)
                for arg_type in primitive.arg_types
            )
        ]
        self.rng.shuffle(operators)
        for primitive in operators[:16]:
            anchor_indices = [
                index
                for index, arg_type in enumerate(primitive.arg_types)
                if is_subtype(subtree.out_type, arg_type)
            ]
            self.rng.shuffle(anchor_indices)
            for anchor in anchor_indices:
                children: list[Node] = []
                remaining = node_budget - 1 - subtree.size()
                failed = False
                for index, arg_type in enumerate(primitive.arg_types):
                    if index == anchor:
                        children.append(subtree)
                        continue
                    remaining_arguments = sum(
                        1
                        for later in range(index + 1, primitive.arity)
                        if later != anchor
                    )
                    budget = max(1, remaining - remaining_arguments)
                    try:
                        child = self.generator.grow_subtree(
                            arg_type,
                            max_depth=max(1, depth_budget - 1),
                            max_nodes=budget,
                            grow=True,
                        )
                    except GenerationError:
                        failed = True
                        break
                    children.append(child)
                    remaining -= child.size()
                if failed:
                    continue
                inserted = Node(primitive.name, tuple(children))
                candidate = self._prepare_tree(replace_at(tree, path, inserted))
                if candidate != tree and self._fits_complexity(candidate):
                    return candidate
        return tree

    def _point_mutation(self, tree: Node) -> Node:
        path, _, node = self.rng.choice(iter_positions(tree, self.root_type))
        # Selecting a managed call or one of its immediate parameters mutates that call as one
        # unit, so constraints such as MACD fast < slow cannot be broken transiently.
        if node.primitive.macro_policy:
            mutated = self._mutate_policy_call(tree, path)
            if mutated != tree:
                return mutated
        if path:
            parent_path = path[:-1]
            parent = self._node_at(tree, parent_path)
            if parent.primitive.macro_policy:
                mutated = self._mutate_policy_call(tree, parent_path)
                if mutated != tree:
                    return mutated
        prim = node.primitive
        if prim.kind is Kind.EPHEMERAL:
            coefficient = self._mutate_formula_coefficient(tree, path, node)
            if coefficient is not None:
                return coefficient
            assert prim.sampler is not None
            candidate = replace_at(tree, path, Node(node.name, value=prim.sampler(self.rng)))
            return self._prepare_tree(candidate) if self._fits_complexity(candidate) else tree
        if prim.kind is Kind.OPERAND:
            others = [p.name for p in OPERANDS.values() if p.name != node.name]
            candidate = replace_at(tree, path, Node(self.rng.choice(others)))
            return self._prepare_tree(candidate) if self._fits_complexity(candidate) else tree
        # operator -> a different operator with an identical signature (children stay valid)
        same = [
            p
            for p in OPERATORS.values()
            if p.arg_types == prim.arg_types
            and p.out_type == prim.out_type
            and p.name != node.name
            and operator_allowed(p, self.allowed_operators)
        ]
        if not same:
            return tree
        candidate = replace_at(
            tree,
            path,
            Node(self.rng.choice(same).name, node.children, node.value),
        )
        return self._prepare_tree(candidate) if self._fits_complexity(candidate) else tree

    def _offspring(
        self,
        *,
        protected_indices: Sequence[int] | None = None,
    ) -> tuple[Node, list[int], str]:
        ops: list[str] = []
        if self.rng.random() < self.config.crossover_rate:
            (parent_a, idx_a), (parent_b, idx_b) = (
                self._tournament(protected_indices),
                self._tournament(),
            )
            tree = self._crossover(parent_a.tree, parent_b.tree)
            parents = [idx_a, idx_b]
            ops.append("crossover")
        else:
            parent, idx = self._tournament(protected_indices)
            tree, parents = parent.tree, [idx]
            ops.append("reproduction")
        if self.rng.random() < self.config.subtree_mutation_rate:
            tree = self._subtree_mutation(tree)
            ops.append("subtree_mut")
        if self.rng.random() < self.config.point_mutation_rate:
            tree = self._point_mutation(tree)
            ops.append("point_mut")
        return self._prepare_tree(tree), parents, "+".join(ops)

    def _composition_offspring(
        self,
        *,
        protected_indices: Sequence[int] | None = None,
    ) -> tuple[Node, list[int], str]:
        parent_a, index_a = self._tournament(protected_indices)
        parent_b, index_b = self._tournament()
        child = self._composition_crossover(parent_a.tree, parent_b.tree)
        return child, [index_a, index_b], "composition_crossover"

    def _insertion_offspring(
        self,
        *,
        protected_indices: Sequence[int] | None = None,
    ) -> tuple[Node, list[int], str]:
        parent, index = self._tournament(protected_indices)
        return (
            self._insertion_mutation(parent.tree),
            [index],
            "insertion_mutation",
        )

    def _two_edit_offspring(
        self,
        *,
        protected_indices: Sequence[int] | None = None,
    ) -> tuple[Node, list[int], str]:
        parent, index = self._tournament(protected_indices)
        tree = parent.tree
        operations: list[str] = []
        for _ in range(2):
            operation = self.rng.choice(("point", "subtree", "insertion"))
            if operation == "point":
                tree = self._point_mutation(tree)
            elif operation == "subtree":
                tree = self._subtree_mutation(tree)
            else:
                tree = self._insertion_mutation(tree)
            operations.append(operation)
        return (
            self._prepare_tree(tree),
            [index],
            f"two_edit:{'+'.join(operations)}",
        )

    @staticmethod
    def _complexity_band(size: int) -> str:
        if size <= 5:
            return "01-05"
        if size <= 10:
            return "06-10"
        if size <= 20:
            return "11-20"
        return "21+"

    def _exploration_niche(self, tree: Node) -> tuple[tuple[str, ...], str, str]:
        """Formula revision, root category, and size band define a protected niche."""
        calls = self._formula_calls(simplify(tree))
        formula = (calls[0].name,) if calls else ()
        root = tree.primitive
        root_category = (
            str((root.macro_policy or {}).get("category") or CUSTOM)
            if root.macro_body is not None
            else builtin_category(root.name)
        )
        expanded_size = expand_all(simplify(tree)).size()
        return formula, root_category, self._complexity_band(expanded_size)

    @staticmethod
    def _format_niche(niche: tuple[tuple[str, ...], str, str]) -> str:
        formula, category, size_band = niche
        return f"{formula[0] if formula else 'none'}|{category}|{size_band}"

    def _niche_occupancy(self) -> dict[str, int]:
        occupancy: Counter[str] = Counter(
            self._format_niche(self._exploration_niche(individual.tree))
            for individual in self.population
        )
        return dict(sorted(occupancy.items()))

    def _stepping_stone_indices(
        self,
        order: Sequence[int],
        *,
        excluded: set[int],
        target: int,
        settings: ExplorationSettings,
    ) -> list[int]:
        """Select young, non-elite survivors round-robin across capped niches."""
        if target <= 0:
            return []
        niches: dict[tuple[tuple[str, ...], str, str], list[int]] = {}
        for index in order:
            if index in excluded:
                continue
            individual = self.population[index]
            if self.generation - individual.birth_generation >= settings.stepping_stone_ttl:
                continue
            key = self._exploration_niche(individual.tree)
            niches.setdefault(key, []).append(index)

        protected: dict[tuple[tuple[str, ...], str, str], list[int]] = {}
        for key, candidates in niches.items():
            best = candidates[0]
            selected = [best]
            best_identity = self._canonical_key(self.population[best].tree)
            newest = sorted(
                candidates[1:],
                key=lambda index: (
                    -self.population[index].birth_generation,
                    index,
                ),
            )
            for index in newest:
                if self._canonical_key(self.population[index].tree) == best_identity:
                    continue
                selected.append(index)
                break
            protected[key] = selected[: settings.niche_cap]

        selected_indices: list[int] = []
        keys = list(protected)
        offset = 0
        while len(selected_indices) < target:
            advanced = False
            for key in keys:
                bucket = protected[key]
                if offset < len(bucket):
                    selected_indices.append(bucket[offset])
                    advanced = True
                    if len(selected_indices) >= target:
                        break
            if not advanced:
                break
            offset += 1
        return selected_indices

    @staticmethod
    def _lane_targets(
        slots: int,
        settings: ExplorationSettings,
        parameter_fraction: float,
    ) -> dict[str, int]:
        """Resolve exact per-generation lane budgets without stochastic rounding."""
        return {
            "parameter": min(slots, int(round(slots * parameter_fraction))),
            "two_edit": min(slots, int(round(slots * settings.two_edit_fraction))),
            "composition": min(
                slots, int(round(slots * settings.composition_fraction))
            ),
            "insertion": min(slots, int(round(slots * settings.insertion_fraction))),
        }

    @staticmethod
    def _next_lane(
        scheduled: Counter[str],
        targets: dict[str, int],
        cursor: int,
    ) -> tuple[str, int]:
        lanes = ("parameter", "two_edit", "composition", "insertion")
        for offset in range(len(lanes)):
            lane = lanes[(cursor + offset) % len(lanes)]
            if scheduled[lane] < targets[lane]:
                return lane, cursor + offset + 1
        return "standard", cursor + 1

    # --- the loop ----------------------------------------------------------------
    def _record(self) -> None:
        fits = [ind.fitness for ind in self.population]
        best = max(self.population, key=lambda ind: ind.fitness)
        keys = [self._canonical_key(ind.tree) for ind in self.population]
        unique_count = len(set(keys))
        best_key = self._canonical_key(best.tree)
        if best_key == self._champion_key:
            self._champion_age += 1
        else:
            self._champion_key = best_key
            self._champion_age = 1
        if best.fitness > self._best_ever + 1e-15:
            self._best_ever = best.fitness
            self._generations_since_improvement = 0
        elif self.history:
            self._generations_since_improvement += 1
        diversity_ratio = unique_count / len(self.population)
        exploration = dict(self._last_exploration_diagnostics)
        exploration["niche_occupancy"] = self._niche_occupancy()
        exploration["parameter_distance_from_defaults"] = (
            self._parameter_distance_from_defaults()
        )
        exploration["champion_stepping_stone_ancestors"] = list(
            best.stepping_stone_ancestors
        )
        self._last_exploration_diagnostics = exploration
        self.history.append(
            {
                "generation": self.generation,
                "best_fitness": float(best.fitness),
                "mean_fitness": float(sum(fits) / len(fits)),
                "best_ic": float(best.metrics.get("ic", 0.0)),
                "best_signed_ic": float(best.metrics.get("signed_ic", 0.0)),
                "unique_tree_count": float(unique_count),
                "unique_tree_ratio": float(diversity_ratio),
                "duplicate_count": float(len(self.population) - unique_count),
                "novel_offspring_count": float(self._last_novel_offspring),
                "parameter_neighbor_count": float(self._last_parameter_neighbors),
                "formula_default_injection_count": float(
                    self._last_formula_default_injections
                ),
                "champion_age": float(self._champion_age),
                "generations_since_improvement": float(
                    self._generations_since_improvement
                ),
                "diversity_warning": float(
                    diversity_ratio < DIVERSITY_WARNING_THRESHOLD
                ),
                "formula_exploration": self.formula_exploration(),
                "exploration": exploration,
            }
        )
        if diversity_ratio < DIVERSITY_WARNING_THRESHOLD:
            callback = getattr(self.recorder, "on_diversity", None)
            if callback is not None:
                callback(
                    generation=self.generation,
                    unique_tree_ratio=diversity_ratio,
                    duplicate_count=len(self.population) - unique_count,
                )

    def _validate_seed(self, tree: Node) -> Node:
        return validate_seed(
            self._prepare_tree(tree),
            max_depth=self.config.max_depth,
            max_nodes=self.config.max_nodes,
            root_type=self.root_type,
        )

    def initialize(
        self,
        seeds: Sequence[Node] = (),
        *,
        stop: Callable[[], bool] | None = None,
    ) -> None:
        rng_state = self.rng.getstate()
        simplifications_before = self._pre_cache_simplifications
        complexity_rejections_before = self._complexity_rejections
        explicit_seed_trees = [self._validate_seed(s) for s in seeds]
        if len(explicit_seed_trees) > self.config.population_size:
            raise ValueError(
                f"{len(explicit_seed_trees)} seeds exceed "
                f"population_size {self.config.population_size}"
            )
        self._managed_formula_defaults = self._configured_formula_defaults()
        formula_capacity = min(
            self.config.population_size // 2,
            self.config.population_size - len(explicit_seed_trees),
        )
        formula_seed_trees = self._managed_formula_defaults[:formula_capacity]
        self._pending_formula_defaults = list(
            self._managed_formula_defaults[formula_capacity:]
        )
        seed_trees = [*explicit_seed_trees, *formula_seed_trees]
        generated_count = self.config.population_size - len(seed_trees)
        generated = self.generator.ramped_half_and_half(
            generated_count,
            min_depth=self.config.min_depth,
            max_depth=self.config.max_depth,
        )
        generated = [
            prepared
            for tree in generated
            if self._fits_complexity(prepared := self._prepare_tree(tree))
        ]
        attempts = 0
        attempt_limit = max(1_000, generated_count * 200)
        depths = list(range(self.config.min_depth, self.config.max_depth + 1)) or [
            self.config.max_depth
        ]
        while len(generated) < generated_count and attempts < attempt_limit:
            if stop is not None and stop():
                self.rng.setstate(rng_state)
                raise TrainingCancelled("training cancelled during initialization")
            candidate = self.generator.generate(
                grow=(attempts % 2 == 0),
                max_depth=depths[attempts % len(depths)],
            )
            candidate = self._prepare_tree(candidate)
            attempts += 1
            if self._fits_complexity(candidate):
                generated.append(candidate)
        if len(generated) < generated_count:
            self.rng.setstate(rng_state)
            raise GenerationError(
                "could not generate a population within the expanded formula complexity limits"
            )
        candidates = [*seed_trees, *generated]
        counts: Counter[str] = Counter()
        trees: list[Node] = []
        for tree in candidates:
            if self._accept_under_copy_limit(tree, counts):
                trees.append(tree)
        while len(trees) < self.config.population_size:
            if stop is not None and stop():
                self.rng.setstate(rng_state)
                raise TrainingCancelled("training cancelled during initialization")
            trees.append(self._random_immigrant(counts))
        try:
            population = self._individuals(
                trees,
                phase="initializing",
                stop=stop,
                birth_generation=0,
            )
        except TrainingCancelled:
            self.rng.setstate(rng_state)
            raise
        self.population = population
        self._pending_formula_defaults = [
            tree
            for tree in self._managed_formula_defaults
            if self._cache_key(tree) not in self._cache
        ]
        self.generation = 0
        base_settings = self._base_exploration_settings()
        self._last_exploration_diagnostics = {
            "profile": base_settings.profile,
            "resolved": base_settings.to_dict(),
            "stagnation_boost": False,
            "stepping_stone_survivors": 0,
            "protected_parent_offspring": 0,
            "parameter_beam_neighbors": 0,
            "parameter_frontier_neighbors": 0,
            "diagonal_parameter_moves": 0,
            "composition_attempts": 0,
            "composition_successes": 0,
            "insertion_attempts": 0,
            "insertion_successes": 0,
            "two_edit_attempts": 0,
            "two_edit_successes": 0,
            "duplicate_rejections": 0,
            "hard_limit_rejections": (
                self._complexity_rejections - complexity_rejections_before
            ),
            "operation_attempts": {},
            "operation_successes": {},
            "operation_rejections": {},
            "parent_contribution": {},
            "no_op_crossover_rate": 0.0,
            "complexity_rejections": (
                self._complexity_rejections - complexity_rejections_before
            ),
            "pre_cache_simplifications": (
                self._pre_cache_simplifications - simplifications_before
            ),
        }
        if self.recorder is not None:
            seed_keys = Counter(
                self._canonical_key(tree) for tree in explicit_seed_trees
            )
            formula_keys = Counter(
                self._canonical_key(tree) for tree in formula_seed_trees
            )
            ops = []
            for tree in trees:
                key = self._canonical_key(tree)
                if seed_keys[key] > 0:
                    ops.append("seed")
                    seed_keys[key] -= 1
                elif formula_keys[key] > 0:
                    ops.append("formula_default")
                    formula_keys[key] -= 1
                else:
                    ops.append("init")
            self.recorder.on_init(
                [ind.tree for ind in self.population],
                fitnesses=[ind.fitness for ind in self.population],
                ops=ops,
            )
        self._record()

    def _step(self, *, stop: Callable[[], bool] | None = None) -> None:
        rng_state = self.rng.getstate()
        pending_defaults_state = list(self._pending_formula_defaults)
        beam_cursor_state = self._parameter_beam_cursor
        frontier_cursor_state = self._parameter_frontier_cursor
        parameter_lane_cursor_state = self._parameter_lane_cursor
        protected_parent_cursor_state = self._protected_parent_cursor
        simplification_state = self._pre_cache_simplifications
        complexity_rejection_state = self._complexity_rejections
        formula_move_counts_state = {
            name: Counter(counts)
            for name, counts in self._formula_move_counts.items()
        }
        simplifications_before = self._pre_cache_simplifications
        complexity_rejections_before = self._complexity_rejections
        base_settings = self._base_exploration_settings()
        settings = self._generation_exploration_settings()
        n = len(self.population)
        order = sorted(
            range(n),
            key=lambda index: (-self.population[index].fitness, index),
        )
        entries: list[tuple[Node, list[int], str, float]] = []
        next_pop: list[Individual] = []
        counts: Counter[str] = Counter()
        elite_indices: set[int] = set()
        for i in order[: self.config.elitism]:
            elite = self.population[i]
            if self._accept_under_copy_limit(elite.tree, counts):
                next_pop.append(elite)
                entries.append((elite.tree, [i], "elite", elite.fitness))
                elite_indices.add(i)

        survivor_target = min(
            self.config.population_size - len(next_pop),
            int(round(self.config.population_size * settings.stepping_stone_fraction)),
        )
        stepping_indices = self._stepping_stone_indices(
            order,
            excluded=elite_indices,
            target=survivor_target,
            settings=settings,
        )
        accepted_stepping_indices: list[int] = []
        for index in stepping_indices:
            survivor = self.population[index]
            if not self._accept_under_copy_limit(survivor.tree, counts):
                continue
            ancestry = tuple(
                dict.fromkeys(
                    (
                        *survivor.stepping_stone_ancestors,
                        self._canonical_key(survivor.tree),
                    )
                )
            )
            carried = dataclasses.replace(
                survivor,
                stepping_stone_ancestors=ancestry,
            )
            next_pop.append(carried)
            entries.append(
                (carried.tree, [index], "stepping_stone", carried.fitness)
            )
            accepted_stepping_indices.append(index)

        previous_keys = {self._canonical_key(ind.tree) for ind in self.population}
        offspring: list[tuple[Node, list[int], str]] = []
        novel_offspring = 0
        available_slots = self.config.population_size - len(next_pop)
        formula_injection_budget = min(
            len(self._pending_formula_defaults),
            max(1, self.config.population_size // 2),
            available_slots,
        )
        formula_injections = 0
        while formula_injections < formula_injection_budget:
            formula_default = self._take_pending_formula_default(counts)
            if formula_default is None:
                break
            formula_entry: tuple[Node, list[int], str] = (
                formula_default,
                [],
                "formula_default",
            )
            formula_injections += 1
            if self._canonical_key(formula_default) not in previous_keys:
                novel_offspring += 1
            offspring.append(formula_entry)

        variable_slots = self.config.population_size - len(next_pop) - len(offspring)
        lane_targets = self._lane_targets(
            variable_slots,
            settings,
            self.config.parameter_neighbor_fraction,
        )
        protected_parent_target = min(
            variable_slots,
            int(round(variable_slots * settings.protected_parent_fraction)),
        )
        scheduled: Counter[str] = Counter()
        actual: Counter[str] = Counter()
        lane_cursor = 0
        duplicate_rejections = 0
        protected_parent_offspring = 0
        parameter_neighbors = 0
        parent_contribution: Counter[int] = Counter()
        operation_attempts: Counter[str] = Counter()
        operation_successes: Counter[str] = Counter()
        operation_rejections: Counter[str] = Counter()

        while len(next_pop) + len(offspring) < self.config.population_size:
            lane, lane_cursor = self._next_lane(scheduled, lane_targets, lane_cursor)
            protected_pool: list[int] | None = None
            if (
                protected_parent_offspring < protected_parent_target
                and accepted_stepping_indices
            ):
                protected_pool = [
                    accepted_stepping_indices[
                        self._protected_parent_cursor
                        % len(accepted_stepping_indices)
                    ]
                ]
                self._protected_parent_cursor += 1
            accepted: tuple[Node, list[int], str] | None = None
            for _ in range(DUPLICATE_RETRY_LIMIT):
                operation_attempts[lane] += 1
                if lane == "parameter":
                    neighbor = self._parameter_neighbor(
                        eligible_indices=(
                            set(protected_pool) if protected_pool is not None else None
                        )
                    )
                    if neighbor is None:
                        neighbor = self._parameter_neighbor()
                    if neighbor is not None:
                        candidate, parents, op = neighbor
                    elif protected_pool is None:
                        candidate, parents, op = self._offspring()
                    else:
                        candidate, parents, op = self._offspring(
                            protected_indices=protected_pool
                        )
                elif lane == "two_edit":
                    actual["two_edit_attempts"] += 1
                    candidate, parents, op = self._two_edit_offspring(
                        protected_indices=protected_pool
                    )
                elif lane == "composition":
                    actual["composition_attempts"] += 1
                    candidate, parents, op = self._composition_offspring(
                        protected_indices=protected_pool
                    )
                elif lane == "insertion":
                    actual["insertion_attempts"] += 1
                    candidate, parents, op = self._insertion_offspring(
                        protected_indices=protected_pool
                    )
                else:
                    if protected_pool is None:
                        candidate, parents, op = self._offspring()
                    else:
                        candidate, parents, op = self._offspring(
                            protected_indices=protected_pool
                        )
                candidate = self._prepare_tree(candidate)
                if "crossover" in op:
                    actual["crossover_attempts"] += 1
                    if (
                        parents
                        and self._canonical_key(candidate)
                        == self._canonical_key(self.population[parents[0]].tree)
                    ):
                        actual["no_op_crossovers"] += 1
                if self._accept_under_copy_limit(candidate, counts):
                    accepted = (candidate, parents, op)
                    operation_successes[lane] += 1
                    break
                duplicate_rejections += 1
                operation_rejections[lane] += 1
            if accepted is None:
                candidate = self._random_immigrant(counts)
                accepted = (candidate, [], "immigrant")
                operation_successes["immigrant"] += 1
            tree, parents, op = accepted
            scheduled[lane] += 1
            if op.startswith("parameter_"):
                parameter_neighbors += 1
                mode, formula_name, move_kind = op.split(":", 2)
                actual[f"{mode}_neighbors"] += 1
                if move_kind == "diagonal":
                    actual["diagonal_parameter_moves"] += 1
                self._formula_move_counts.setdefault(
                    formula_name, Counter()
                )[move_kind] += 1
            if op.startswith("two_edit:") and parents:
                actual["two_edit_successes"] += int(
                    self._canonical_key(tree)
                    != self._canonical_key(self.population[parents[0]].tree)
                )
            elif op == "composition_crossover" and parents:
                actual["composition_successes"] += int(
                    self._canonical_key(tree)
                    != self._canonical_key(self.population[parents[0]].tree)
                )
            elif op == "insertion_mutation" and parents:
                actual["insertion_successes"] += int(
                    self._canonical_key(tree)
                    != self._canonical_key(self.population[parents[0]].tree)
                )
            if (
                protected_pool is not None
                and parents
                and parents[0] in set(protected_pool)
            ):
                protected_parent_offspring += 1
            parent_contribution.update(parents)
            if self._canonical_key(tree) not in previous_keys:
                novel_offspring += 1
            offspring.append((tree, parents, op))

        # Variation uses the RNG serially above.  Only pure scoring is parallel, and none of the
        # staged state below becomes visible until the full generation has completed.
        try:
            children = self._individuals(
                [tree for tree, _, _ in offspring],
                phase="training",
                stop=stop,
                birth_generation=self.generation + 1,
            )
        except Exception:
            self.rng.setstate(rng_state)
            self._pending_formula_defaults = pending_defaults_state
            self._parameter_beam_cursor = beam_cursor_state
            self._parameter_frontier_cursor = frontier_cursor_state
            self._parameter_lane_cursor = parameter_lane_cursor_state
            self._protected_parent_cursor = protected_parent_cursor_state
            self._pre_cache_simplifications = simplification_state
            self._complexity_rejections = complexity_rejection_state
            self._formula_move_counts = {
                str(name): Counter(counts)
                for name, counts in formula_move_counts_state.items()
            }
            raise
        stepping_index_set = set(accepted_stepping_indices)
        for (_tree, parents, op), child in zip(offspring, children, strict=True):
            ancestors: dict[str, None] = {}
            for parent_index in parents:
                for ancestor in self.population[parent_index].stepping_stone_ancestors:
                    ancestors.setdefault(ancestor, None)
                if parent_index in stepping_index_set:
                    ancestors.setdefault(
                        self._canonical_key(self.population[parent_index].tree),
                        None,
                    )
            child = dataclasses.replace(
                child,
                stepping_stone_ancestors=tuple(ancestors),
            )
            next_pop.append(child)
            entries.append((child.tree, parents, op, child.fitness))
        self.population = next_pop
        self._last_novel_offspring = novel_offspring
        self._last_parameter_neighbors = parameter_neighbors
        self._last_formula_default_injections = formula_injections
        self.generation += 1
        self._last_exploration_diagnostics = {
            "profile": settings.profile,
            "resolved": settings.to_dict(),
            "stagnation_boost": settings != base_settings,
            "requested_quotas": {
                "stepping_stones": survivor_target,
                "protected_parents": protected_parent_target,
                **lane_targets,
            },
            "stepping_stone_survivors": len(accepted_stepping_indices),
            "protected_parent_offspring": protected_parent_offspring,
            "parameter_beam_neighbors": int(actual["parameter_beam_neighbors"]),
            "parameter_frontier_neighbors": int(
                actual["parameter_frontier_neighbors"]
            ),
            "diagonal_parameter_moves": int(
                actual["diagonal_parameter_moves"]
            ),
            "composition_attempts": int(actual["composition_attempts"]),
            "composition_successes": int(actual["composition_successes"]),
            "insertion_attempts": int(actual["insertion_attempts"]),
            "insertion_successes": int(actual["insertion_successes"]),
            "two_edit_attempts": int(actual["two_edit_attempts"]),
            "two_edit_successes": int(actual["two_edit_successes"]),
            "duplicate_rejections": duplicate_rejections,
            "hard_limit_rejections": (
                self._complexity_rejections - complexity_rejections_before
            ),
            "operation_attempts": dict(sorted(operation_attempts.items())),
            "operation_successes": dict(sorted(operation_successes.items())),
            "operation_rejections": dict(sorted(operation_rejections.items())),
            "parent_contribution": {
                str(index): count
                for index, count in sorted(parent_contribution.items())
            },
            "no_op_crossover_rate": (
                float(actual["no_op_crossovers"] / actual["crossover_attempts"])
                if actual["crossover_attempts"]
                else 0.0
            ),
            "complexity_rejections": (
                self._complexity_rejections - complexity_rejections_before
            ),
            "pre_cache_simplifications": (
                self._pre_cache_simplifications - simplifications_before
            ),
        }
        if self.recorder is not None:
            self.recorder.on_generation(self.generation, entries)
        self._record()

    def run(
        self,
        generations: int | None = None,
        checkpoint_path: Path_ | None = None,
        *,
        seeds: Sequence[Node] = (),
        stop: Callable[[], bool] | None = None,
    ) -> Individual:
        target = generations if generations is not None else self.config.generations
        self.termination_reason = "completed"
        if self._requires_rescore and self.population:
            self._rescore_after_scorer_upgrade(stop=stop)
        if not self.population:
            try:
                self.initialize(seeds, stop=stop)
            except TrainingCancelled:
                self.termination_reason = "user_stopped"
                raise
            if checkpoint_path is not None:
                self.save_checkpoint(checkpoint_path)
        start = time.monotonic()

        def should_stop() -> bool:
            if stop is not None and stop():
                self.termination_reason = "user_stopped"
                return True
            if (
                self.config.time_budget_s is not None
                and time.monotonic() - start >= self.config.time_budget_s
            ):
                self.termination_reason = "time_budget"
                return True
            return False

        while self.generation < target:
            if should_stop():
                break
            try:
                self._step(stop=should_stop)
            except TrainingCancelled:
                # The staged generation and its score-cache additions were not committed.
                break
            if checkpoint_path is not None:
                self.save_checkpoint(checkpoint_path)
            if should_stop():
                break
        return self.best()

    def _rescore_after_scorer_upgrade(
        self, *, stop: Callable[[], bool] | None = None
    ) -> None:
        """Re-score a legacy checkpoint once without inflating its historical trial count."""
        previous_trials = self.trial_count
        old_cache, old_prior = self._cache, self._prior_trials
        self._cache = {}
        self._prior_trials = 0
        try:
            historical = [
                self._validate_seed(tree)
                for tree in self._pending_rescore_trees
            ]
            if historical:
                self._individuals(
                    historical,
                    phase="initializing",
                    stop=stop,
                )
            scored_population = self._individuals(
                [self._validate_seed(individual.tree) for individual in self.population],
                phase="initializing",
                stop=stop,
            )
            population = [
                dataclasses.replace(
                    scored,
                    birth_generation=original.birth_generation,
                    stepping_stone_ancestors=original.stepping_stone_ancestors,
                )
                for scored, original in zip(
                    scored_population, self.population, strict=True
                )
            ]
        except Exception:
            self._cache, self._prior_trials = old_cache, old_prior
            raise
        self.population = population
        self._prior_trials = max(0, previous_trials - len(self._cache))
        self._requires_rescore = False
        self._pending_rescore_trees = []
        if self.history and int(self.history[-1].get("generation", -1)) == self.generation:
            fits = [individual.fitness for individual in self.population]
            best = max(self.population, key=lambda individual: individual.fitness)
            updated = dict(self.history[-1])
            updated.update(
                {
                    "generation": self.generation,
                    "best_fitness": float(best.fitness),
                    "mean_fitness": float(sum(fits) / len(fits)),
                    "best_ic": float(best.metrics.get("ic", 0.0)),
                    "best_signed_ic": float(best.metrics.get("signed_ic", 0.0)),
                }
            )
            self.history[-1] = updated

    def rescore_population(self, fwd: pd.DataFrame | None = None) -> None:
        """Re-score the current population against the current panel/config.

        Used when a continued session changes the universe or a scoring-relevant
        setting: stale cached scores are invalidated, but the trials they counted
        are folded into the prior baseline first - the count never shrinks.
        """
        self._prior_trials += len(self._cache)
        self._cache.clear()
        self.fwd = (
            fwd
            if fwd is not None
            else forward_returns(self.panel, self.config.horizon, self.config.execution)
        )
        scored_population = self._individuals(
            [self._validate_seed(individual.tree) for individual in self.population],
            phase="initializing",
        )
        self.population = [
            dataclasses.replace(
                scored,
                birth_generation=original.birth_generation,
                stepping_stone_ancestors=original.stepping_stone_ancestors,
            )
            for scored, original in zip(
                scored_population, self.population, strict=True
            )
        ]

    def best(self, *, simplified: bool = True) -> Individual:
        top = max(self.population, key=lambda ind: ind.fitness)
        if simplified:
            return Individual(
                simplify(top.tree),
                top.fitness,
                top.metrics,
                top.birth_generation,
                top.stepping_stone_ancestors,
            )
        return top

    # --- checkpointing -----------------------------------------------------------
    def save_checkpoint(self, path: Path_) -> None:
        version, internal, gauss = self.rng.getstate()
        used_formula_policies: dict[str, dict[str, Any]] = {}

        def collect_policy(node: Node) -> None:
            if node.name == "$arg":
                return
            prim = node.primitive
            if prim.macro_body is not None and prim.name not in used_formula_policies:
                if prim.macro_policy is not None:
                    used_formula_policies[prim.name] = prim.macro_policy
                collect_policy(prim.macro_body)
            for child in node.children:
                collect_policy(child)

        for individual in self.population:
            collect_policy(individual.tree)
        state = {
            "scorer_version": SCORER_VERSION,
            "evolution_version": EVOLUTION_VERSION,
            "scorer_backend": self.scorer_backend,
            "generation": self.generation,
            "rng_state": [version, list(internal), gauss],
            "config": self.config.to_dict(),
            "trials": self.trial_count,
            "history": self.history,
            "resolved_exploration": self._base_exploration_settings().to_dict(),
            # Validation selection is over every distinct searched expression, not only
            # the population that survived the latest tournament.  Persist insertion order
            # so checkpoint/resume retains deterministic first-seen tie-breaking.
            "score_cache": [
                {
                    "tree": to_dict(from_json(key)),
                    "fitness": fitness,
                    "metrics": metrics,
                }
                for key, (fitness, metrics) in self._cache.items()
            ],
            "diversity_state": {
                "champion_key": self._champion_key,
                "champion_age": self._champion_age,
                "best_ever": self._best_ever,
                "generations_since_improvement": self._generations_since_improvement,
                "last_novel_offspring": self._last_novel_offspring,
                "last_parameter_neighbors": self._last_parameter_neighbors,
                "last_formula_default_injections": (
                    self._last_formula_default_injections
                ),
            },
            "formula_exploration": self.formula_exploration(),
            "formula_exploration_state": {
                "managed_defaults": [
                    to_dict(tree) for tree in self._managed_formula_defaults
                ],
                "pending_defaults": [
                    to_dict(tree) for tree in self._pending_formula_defaults
                ],
                "validation_best": self._formula_validation_best,
                "move_counts": {
                    name: dict(counts)
                    for name, counts in self._formula_move_counts.items()
                },
            },
            "exploration_state": {
                "parameter_beam_cursor": self._parameter_beam_cursor,
                "parameter_frontier_cursor": self._parameter_frontier_cursor,
                "parameter_lane_cursor": self._parameter_lane_cursor,
                "protected_parent_cursor": self._protected_parent_cursor,
                "pre_cache_simplifications": self._pre_cache_simplifications,
                "complexity_rejections": self._complexity_rejections,
                "last_diagnostics": self._last_exploration_diagnostics,
            },
            "formula_policies": used_formula_policies,
            "population": [
                {
                    "tree": to_dict(ind.tree),
                    "fitness": ind.fitness,
                    "metrics": ind.metrics,
                    "birth_generation": ind.birth_generation,
                    "stepping_stone_ancestors": list(
                        ind.stepping_stone_ancestors
                    ),
                }
                for ind in self.population
            ],
        }
        Path(path).write_text(json.dumps(state), encoding="utf-8")

    @classmethod
    def from_checkpoint(
        cls,
        path: Path_,
        panel: Panel,
        fwd: pd.DataFrame | None = None,
        *,
        recorder: Any | None = None,
        allowed_operators: set[str] | None = None,
        workers: int = 1,
        memory_budget_bytes: int | None = None,
    ) -> GP:
        state = json.loads(Path(path).read_text(encoding="utf-8"))
        gp = cls(
            GPConfig.from_dict(state["config"]),
            panel,
            fwd,
            recorder=recorder,
            allowed_operators=allowed_operators,
            workers=workers,
            memory_budget_bytes=memory_budget_bytes,
        )
        version, internal, gauss = state["rng_state"]
        gp.rng.setstate((version, tuple(internal), gauss))
        gp.generation = int(state["generation"])
        gp.history = state["history"]
        gp.population = [
            Individual(
                simplify(from_dict(p["tree"])),
                float(p["fitness"]),
                dict(p["metrics"]),
                int(p.get("birth_generation", gp.generation)),
                tuple(
                    str(value)
                    for value in p.get("stepping_stone_ancestors", ())
                ),
            )
            for p in state["population"]
        ]
        diversity_state = state.get("diversity_state") or {}
        gp._champion_key = diversity_state.get("champion_key")
        gp._champion_age = int(diversity_state.get("champion_age", 0))
        gp._best_ever = float(
            diversity_state.get(
                "best_ever",
                max((individual.fitness for individual in gp.population), default=-math.inf),
            )
        )
        gp._generations_since_improvement = int(
            diversity_state.get("generations_since_improvement", 0)
        )
        gp._last_novel_offspring = int(
            diversity_state.get("last_novel_offspring", 0)
        )
        gp._last_parameter_neighbors = int(
            diversity_state.get("last_parameter_neighbors", 0)
        )
        gp._last_formula_default_injections = int(
            diversity_state.get("last_formula_default_injections", 0)
        )
        formula_state = state.get("formula_exploration_state") or {}
        gp._managed_formula_defaults = [
            from_dict(tree) for tree in formula_state.get("managed_defaults") or []
        ]
        gp._pending_formula_defaults = [
            from_dict(tree) for tree in formula_state.get("pending_defaults") or []
        ]
        gp._formula_validation_best = {
            str(name): float(value)
            for name, value in (formula_state.get("validation_best") or {}).items()
            if math.isfinite(float(value))
        }
        gp._formula_move_counts = {
            str(name): Counter(
                {
                    str(kind): int(count)
                    for kind, count in dict(counts).items()
                }
            )
            for name, counts in (formula_state.get("move_counts") or {}).items()
        }
        exploration_state = state.get("exploration_state") or {}
        gp._parameter_beam_cursor = int(
            exploration_state.get("parameter_beam_cursor", 0)
        )
        gp._parameter_frontier_cursor = int(
            exploration_state.get("parameter_frontier_cursor", 0)
        )
        gp._parameter_lane_cursor = int(
            exploration_state.get("parameter_lane_cursor", 0)
        )
        gp._protected_parent_cursor = int(
            exploration_state.get("protected_parent_cursor", 0)
        )
        gp._pre_cache_simplifications = int(
            exploration_state.get("pre_cache_simplifications", 0)
        )
        gp._complexity_rejections = int(
            exploration_state.get("complexity_rejections", 0)
        )
        gp._last_exploration_diagnostics = dict(
            exploration_state.get("last_diagnostics") or {}
        )
        for name, policy in (state.get("formula_policies") or {}).items():
            primitive = OPERATORS.get(name)
            if primitive is None or primitive.macro_policy != policy:
                raise ValueError(
                    f"checkpoint formula policy for {name!r} is unavailable or has changed"
                )
        saved_trials = int(state.get("trials", 0))
        saved_scorer = int(state.get("scorer_version", 1))
        saved_evolution = int(state.get("evolution_version", 1))
        saved_backend = state.get("scorer_backend")
        if saved_evolution != EVOLUTION_VERSION:
            raise ValueError(
                "checkpoint uses an incompatible evolution version; "
                "restart with the same setup"
            )
        saved_exploration = state.get("resolved_exploration")
        if (
            saved_exploration is not None
            and saved_exploration != gp._base_exploration_settings().to_dict()
        ):
            raise ValueError(
                "checkpoint uses incompatible resolved exploration settings; "
                "restart with the same setup"
            )
        saved_cache: list[Individual] = []
        for cached in state.get("score_cache") or []:
            try:
                saved_cache.append(
                    Individual(
                        simplify(from_dict(cached["tree"])),
                        float(cached["fitness"]),
                        dict(cached["metrics"]),
                        0,
                    )
                )
            except (KeyError, TypeError, ValueError):
                saved_cache = []
                break
        if (
            saved_scorer == SCORER_VERSION
            and saved_backend == gp.scorer_backend
        ):
            # New checkpoints restore all searched formulas; legacy checkpoints at least retain
            # free current-population hits.
            for individual in saved_cache or gp.population:
                gp._cache.setdefault(
                    gp._cache_key(individual.tree),
                    (individual.fitness, individual.metrics),
                )
            gp._prior_trials = max(0, saved_trials - len(gp._cache))
        else:
            gp._prior_trials = saved_trials
            gp._requires_rescore = True
            gp._pending_rescore_trees = [
                individual.tree for individual in saved_cache
            ]
        return gp
