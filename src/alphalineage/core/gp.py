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
import random
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from alphalineage.core.fitness import forward_returns, score_tree
from alphalineage.core.generate import GenerationError, RandomTreeGenerator, operator_allowed
from alphalineage.core.panel import Panel
from alphalineage.core.primitives import OPERANDS, OPERATORS, Kind
from alphalineage.core.simplify import simplify
from alphalineage.core.tree import Node, from_dict, to_dict, to_json, validate
from alphalineage.core.types import DType, is_subtype

Path_ = str | Path
Position = tuple[tuple[int, ...], DType, Node]


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

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> GPConfig:
        names = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in names})


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
    if tree.depth() > max_depth:
        raise ValueError(f"seed depth {tree.depth()} exceeds max_depth {max_depth}")
    if tree.size() > max_nodes:
        raise ValueError(f"seed size {tree.size()} exceeds max_nodes {max_nodes}")
    return tree


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
    ) -> None:
        self.config = config
        self.panel = panel
        self.fwd = fwd if fwd is not None else forward_returns(panel, config.horizon)
        self.root_type = root_type
        # Optional lineage recorder (duck-typed: on_init(trees), on_generation(gen, entries)).
        self.recorder = recorder
        # Operator name allow-set (None => default pool, condition category excluded).
        self.allowed_operators = allowed_operators
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
        result = score_tree(
            tree,
            self.panel,
            self.fwd,
            method=self.config.ic_method,
            parsimony=self.config.parsimony,
            min_names=self.config.min_names,
        )
        self._cache[key] = result
        return result

    def _individual(self, tree: Node) -> Individual:
        fitness, metrics = self._score(tree)
        return Individual(tree, fitness, metrics)

    # --- selection & variation ---------------------------------------------------
    def _tournament(self) -> tuple[Individual, int]:
        # Index-based selection (same RNG draws as choice(population)) so we can record parents.
        n = len(self.population)
        idxs = [self.rng.choice(range(n)) for _ in range(self.config.tournament_size)]
        best = max(idxs, key=lambda i: self.population[i].fitness)
        return self.population[best], best

    def _crossover(self, a: Node, b: Node) -> Node:
        path, required, _ = self.rng.choice(iter_positions(a, self.root_type))
        donors = [
            sub
            for (_, _, sub) in iter_positions(b, self.root_type)
            if is_subtype(sub.out_type, required)
        ]
        if not donors:
            return a
        for _ in range(8):
            child = replace_at(a, path, self.rng.choice(donors))
            if child.depth() <= self.config.max_depth and child.size() <= self.config.max_nodes:
                return child
        return a

    def _subtree_mutation(self, tree: Node) -> Node:
        path, required, sub = self.rng.choice(iter_positions(tree, self.root_type))
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
        return replace_at(tree, path, fresh)

    def _point_mutation(self, tree: Node) -> Node:
        path, _, node = self.rng.choice(iter_positions(tree, self.root_type))
        prim = node.primitive
        if prim.kind is Kind.EPHEMERAL:
            assert prim.sampler is not None
            return replace_at(tree, path, Node(node.name, value=prim.sampler(self.rng)))
        if prim.kind is Kind.OPERAND:
            others = [p.name for p in OPERANDS.values() if p.name != node.name]
            return replace_at(tree, path, Node(self.rng.choice(others)))
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
        return replace_at(tree, path, Node(self.rng.choice(same).name, node.children, node.value))

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

    def initialize(self, seeds: Sequence[Node] = ()) -> None:
        seed_trees = [self._validate_seed(s) for s in seeds]
        if len(seed_trees) > self.config.population_size:
            raise ValueError(
                f"{len(seed_trees)} seeds exceed population_size {self.config.population_size}"
            )
        trees = seed_trees + self.generator.ramped_half_and_half(
            self.config.population_size - len(seed_trees),
            min_depth=self.config.min_depth,
            max_depth=self.config.max_depth,
        )
        self.population = [self._individual(t) for t in trees]
        self.generation = 0
        if self.recorder is not None:
            ops = ["seed"] * len(seed_trees) + ["init"] * (len(trees) - len(seed_trees))
            self.recorder.on_init(
                [ind.tree for ind in self.population],
                fitnesses=[ind.fitness for ind in self.population],
                ops=ops,
            )
        self._record()

    def _step(self) -> None:
        n = len(self.population)
        order = sorted(range(n), key=lambda i: self.population[i].fitness, reverse=True)
        entries: list[tuple[Node, list[int], str, float]] = []
        next_pop: list[Individual] = []
        for i in order[: self.config.elitism]:
            elite = self.population[i]
            next_pop.append(elite)
            entries.append((elite.tree, [i], "elite", elite.fitness))
        while len(next_pop) < self.config.population_size:
            tree, parents, op = self._offspring()
            child = self._individual(tree)
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
        if not self.population:
            self.initialize(seeds)
            if checkpoint_path is not None:
                self.save_checkpoint(checkpoint_path)
        start = time.monotonic()
        while self.generation < target:
            if stop is not None and stop():
                break
            self._step()
            if checkpoint_path is not None:
                self.save_checkpoint(checkpoint_path)
            if (
                self.config.time_budget_s is not None
                and time.monotonic() - start >= self.config.time_budget_s
            ):
                break
        return self.best()

    def rescore_population(self, fwd: pd.DataFrame | None = None) -> None:
        """Re-score the current population against the current panel/config.

        Used when a continued session changes the universe or a scoring-relevant
        setting: stale cached scores are invalidated, but the trials they counted
        are folded into the prior baseline first - the count never shrinks.
        """
        self._prior_trials += len(self._cache)
        self._cache.clear()
        self.fwd = fwd if fwd is not None else forward_returns(self.panel, self.config.horizon)
        self.population = [self._individual(ind.tree) for ind in self.population]

    def best(self, *, simplified: bool = True) -> Individual:
        top = max(self.population, key=lambda ind: ind.fitness)
        if simplified:
            return Individual(simplify(top.tree), top.fitness, top.metrics)
        return top

    # --- checkpointing -----------------------------------------------------------
    def save_checkpoint(self, path: Path_) -> None:
        version, internal, gauss = self.rng.getstate()
        state = {
            "generation": self.generation,
            "rng_state": [version, list(internal), gauss],
            "config": self.config.to_dict(),
            "trials": self.trial_count,
            "history": self.history,
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
    ) -> GP:
        state = json.loads(Path(path).read_text(encoding="utf-8"))
        gp = cls(
            GPConfig.from_dict(state["config"]),
            panel,
            fwd,
            recorder=recorder,
            allowed_operators=allowed_operators,
        )
        gp._prior_trials = int(state.get("trials", 0))
        version, internal, gauss = state["rng_state"]
        gp.rng.setstate((version, tuple(internal), gauss))
        gp.generation = int(state["generation"])
        gp.history = state["history"]
        gp.population = [
            Individual(from_dict(p["tree"]), float(p["fitness"]), dict(p["metrics"]))
            for p in state["population"]
        ]
        return gp
