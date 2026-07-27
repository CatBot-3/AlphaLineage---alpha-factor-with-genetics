"""P2-T1, T3-T7 - the genetic-programming search.

A population of typed trees evolved by tournament selection, type-safe subtree crossover,
and mutation, scored by mean |rank IC| (invariants 3 and 4). The loop is deterministic
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
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from alphalineage.core.extensions import InvalidOperator, expand_all
from alphalineage.core.fitness import forward_returns, score_trees
from alphalineage.core.generate import GenerationError, RandomTreeGenerator, operator_allowed
from alphalineage.core.panel import Panel
from alphalineage.core.primitives import OPERANDS, OPERATORS, Kind
from alphalineage.core.simplify import simplify
from alphalineage.core.tree import Node, from_dict, to_dict, to_json, validate
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
SCORER_VERSION = 6
# Variation semantics are versioned separately from numerical scoring.  Version 2 introduces
# formula-owned local parameter policies and atomic cross-parameter constraints.
EVOLUTION_VERSION = 2


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
    parsimony: float = 1e-3
    elitism: int = 1
    ic_method: str = "spearman"
    min_names: int = 5
    horizon: int = 1
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

        for name in ("crossover_rate", "subtree_mutation_rate", "point_mutation_rate"):
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
        self.fwd = fwd if fwd is not None else forward_returns(panel, config.horizon)
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
        self.history: list[dict[str, float]] = []
        self._cache: dict[str, tuple[float, dict[str, float]]] = {}
        # Trials counted before this object's cache existed (resumes, invalidated caches).
        # Monotone by construction: it only ever grows, so deflation never softens.
        self._prior_trials = 0
        self._requires_rescore = False
        self.termination_reason = "completed"

    @property
    def trial_count(self) -> int:
        """Distinct factors scored so far - the deflation's number of trials."""
        return self._prior_trials + len(self._cache)

    # --- scoring -----------------------------------------------------------------
    def _score(self, tree: Node) -> tuple[float, dict[str, float]]:
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
            min_names=self.config.min_names,
            workers=1,
            memory_budget_bytes=self.memory_budget_bytes,
        )[0]
        self._cache[key] = result
        return result

    def _individual(self, tree: Node) -> Individual:
        fitness, metrics = self._score(tree)
        return Individual(tree, fitness, metrics)

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
    ) -> list[Individual]:
        """Score ``trees`` as one deterministic transaction.

        Cache hits and duplicates retain input order.  Unique misses are evaluated in bounded
        chunks, but only committed to the coordinator-owned cache after every chunk succeeds.
        Thus cancellation cannot leave a half-generation counted as searched.
        """
        keys = [to_json(tree) for tree in trees]
        missing: dict[str, Node] = {}
        for key, tree in zip(keys, trees, strict=True):
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
        return [Individual(tree, *self._cache[key]) for tree, key in zip(trees, keys, strict=True)]

    def _ensure_scorer_backend(self) -> None:
        """Refuse a mid-run evaluator switch instead of mixing numerical kernels."""
        current = active_scorer_backend(self.config.ic_method)
        if current != self.scorer_backend:
            raise RuntimeError(
                "training evaluator changed during the run; restart or continue from the last "
                "checkpoint so the population can be rescored consistently"
            )

    # --- selection & variation ---------------------------------------------------
    def _tournament(self) -> tuple[Individual, int]:
        # Index-based selection (same RNG draws as choice(population)) so we can record parents.
        n = len(self.population)
        idxs = [self.rng.choice(range(n)) for _ in range(self.config.tournament_size)]
        best = max(idxs, key=lambda i: self.population[i].fitness)
        return self.population[best], best

    def _fits_complexity(self, tree: Node) -> bool:
        try:
            validate(tree)
            if tree.depth() > self.config.max_depth or tree.size() > self.config.max_nodes:
                return False
            expand_all(
                tree,
                max_depth=_expansion_depth_limit(tree, self.config.max_depth),
                max_nodes=self.config.max_nodes,
            )
        except (InvalidOperator, ValueError):
            return False
        return True

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
                return child
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
        return candidate if self._fits_complexity(candidate) else tree

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
            return candidate if self._fits_complexity(candidate) else tree
        if prim.kind is Kind.OPERAND:
            others = [p.name for p in OPERANDS.values() if p.name != node.name]
            candidate = replace_at(tree, path, Node(self.rng.choice(others)))
            return candidate if self._fits_complexity(candidate) else tree
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
        return candidate if self._fits_complexity(candidate) else tree

    def _offspring(self) -> tuple[Node, list[int], str]:
        ops: list[str] = []
        if self.rng.random() < self.config.crossover_rate:
            (parent_a, idx_a), (parent_b, idx_b) = self._tournament(), self._tournament()
            tree = self._crossover(parent_a.tree, parent_b.tree)
            parents = [idx_a, idx_b]
            ops.append("crossover")
        else:
            parent, idx = self._tournament()
            tree, parents = parent.tree, [idx]
            ops.append("reproduction")
        if self.rng.random() < self.config.subtree_mutation_rate:
            tree = self._subtree_mutation(tree)
            ops.append("subtree_mut")
        if self.rng.random() < self.config.point_mutation_rate:
            tree = self._point_mutation(tree)
            ops.append("point_mut")
        return tree, parents, "+".join(ops)

    # --- the loop ----------------------------------------------------------------
    def _record(self) -> None:
        fits = [ind.fitness for ind in self.population]
        best = max(self.population, key=lambda ind: ind.fitness)
        self.history.append(
            {
                "generation": self.generation,
                "best_fitness": float(best.fitness),
                "mean_fitness": float(sum(fits) / len(fits)),
                "best_ic": float(best.metrics.get("ic", 0.0)),
            }
        )

    def _validate_seed(self, tree: Node) -> Node:
        return validate_seed(
            tree,
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
        seed_trees = [self._validate_seed(s) for s in seeds]
        if len(seed_trees) > self.config.population_size:
            raise ValueError(
                f"{len(seed_trees)} seeds exceed population_size {self.config.population_size}"
            )
        generated_count = self.config.population_size - len(seed_trees)
        generated = self.generator.ramped_half_and_half(
            generated_count,
            min_depth=self.config.min_depth,
            max_depth=self.config.max_depth,
        )
        generated = [tree for tree in generated if self._fits_complexity(tree)]
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
            attempts += 1
            if self._fits_complexity(candidate):
                generated.append(candidate)
        if len(generated) < generated_count:
            self.rng.setstate(rng_state)
            raise GenerationError(
                "could not generate a population within the expanded formula complexity limits"
            )
        trees = [*seed_trees, *generated]
        try:
            population = self._individuals(trees, phase="initializing", stop=stop)
        except TrainingCancelled:
            self.rng.setstate(rng_state)
            raise
        self.population = population
        self.generation = 0
        if self.recorder is not None:
            ops = ["seed"] * len(seed_trees) + ["init"] * (len(trees) - len(seed_trees))
            self.recorder.on_init(
                [ind.tree for ind in self.population],
                fitnesses=[ind.fitness for ind in self.population],
                ops=ops,
            )
        self._record()

    def _step(self, *, stop: Callable[[], bool] | None = None) -> None:
        rng_state = self.rng.getstate()
        n = len(self.population)
        order = sorted(range(n), key=lambda i: self.population[i].fitness, reverse=True)
        entries: list[tuple[Node, list[int], str, float]] = []
        next_pop: list[Individual] = []
        for i in order[: self.config.elitism]:
            elite = self.population[i]
            next_pop.append(elite)
            entries.append((elite.tree, [i], "elite", elite.fitness))
        offspring: list[tuple[Node, list[int], str]] = []
        while len(next_pop) + len(offspring) < self.config.population_size:
            tree, parents, op = self._offspring()
            offspring.append((tree, parents, op))

        # Variation uses the RNG serially above.  Only pure scoring is parallel, and none of the
        # staged state below becomes visible until the full generation has completed.
        try:
            children = self._individuals(
                [tree for tree, _, _ in offspring], phase="training", stop=stop
            )
        except TrainingCancelled:
            self.rng.setstate(rng_state)
            raise
        for (tree, parents, op), child in zip(offspring, children, strict=True):
            next_pop.append(child)
            entries.append((tree, parents, op, child.fitness))
        self.population = next_pop
        self.generation += 1
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
            population = self._individuals(
                [self._validate_seed(individual.tree) for individual in self.population],
                phase="initializing",
                stop=stop,
            )
        except Exception:
            self._cache, self._prior_trials = old_cache, old_prior
            raise
        self.population = population
        self._prior_trials = max(0, previous_trials - len(self._cache))
        self._requires_rescore = False
        if self.history and int(self.history[-1].get("generation", -1)) == self.generation:
            fits = [individual.fitness for individual in self.population]
            best = max(self.population, key=lambda individual: individual.fitness)
            self.history[-1] = {
                "generation": self.generation,
                "best_fitness": float(best.fitness),
                "mean_fitness": float(sum(fits) / len(fits)),
                "best_ic": float(best.metrics.get("ic", 0.0)),
            }

    def rescore_population(self, fwd: pd.DataFrame | None = None) -> None:
        """Re-score the current population against the current panel/config.

        Used when a continued session changes the universe or a scoring-relevant
        setting: stale cached scores are invalidated, but the trials they counted
        are folded into the prior baseline first - the count never shrinks.
        """
        self._prior_trials += len(self._cache)
        self._cache.clear()
        self.fwd = fwd if fwd is not None else forward_returns(self.panel, self.config.horizon)
        self.population = self._individuals(
            [self._validate_seed(individual.tree) for individual in self.population],
            phase="initializing",
        )

    def best(self, *, simplified: bool = True) -> Individual:
        top = max(self.population, key=lambda ind: ind.fitness)
        if simplified:
            return Individual(simplify(top.tree), top.fitness, top.metrics)
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
            "formula_policies": used_formula_policies,
            "population": [
                {"tree": to_dict(ind.tree), "fitness": ind.fitness, "metrics": ind.metrics}
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
            Individual(from_dict(p["tree"]), float(p["fitness"]), dict(p["metrics"]))
            for p in state["population"]
        ]
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
        if (
            saved_scorer == SCORER_VERSION
            and saved_evolution == EVOLUTION_VERSION
            and saved_backend == gp.scorer_backend
        ):
            # Restore the current population into the memoization cache.  Older code retained
            # their scores on Individuals but threw away these free resume-time cache hits.
            for individual in gp.population:
                gp._cache.setdefault(
                    to_json(individual.tree), (individual.fitness, individual.metrics)
                )
            gp._prior_trials = max(0, saved_trials - len(gp._cache))
        else:
            gp._prior_trials = saved_trials
            gp._requires_rescore = True
        return gp
