"""P5-T2 - lineage persistence (parent + diff), replayable.

Records every individual a GP run produces - its id, generation, the operation that made it,
and its parents - so a finished run can be reconstructed generation by generation (and later
exported to the static `demo` JSON). The full tree is stored as each node's "diff" payload
(lossless; a minimal structural diff is a future optimization). Implements the duck-typed
recorder protocol the GP calls (``on_init`` / ``on_generation``).
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from alphalineage.core.tree import Node
from alphalineage.core.tree import from_dict as tree_from_dict
from alphalineage.core.tree import to_dict as tree_to_dict


@dataclass
class LineageNode:
    id: int
    generation: int
    op: str
    parents: list[int]
    tree: Node
    fitness: float | None = None


@dataclass
class LineageStore:
    """Accumulates a run's lineage; persists to / loads from JSON."""

    run_id: str = "run"
    nodes: list[LineageNode] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    _next_id: int = 0
    _current_gen_ids: list[int] = field(default_factory=list)

    def _add(
        self,
        generation: int,
        op: str,
        parents: list[int],
        tree: Node,
        fitness: float | None = None,
    ) -> int:
        node_id = self._next_id
        self._next_id += 1
        self.nodes.append(LineageNode(node_id, generation, op, list(parents), tree, fitness))
        return node_id

    # --- recorder protocol (called by GP) ----------------------------------------
    def on_init(
        self,
        trees: Sequence[Node],
        *,
        fitnesses: Sequence[float] | None = None,
        ops: Sequence[str] | None = None,
    ) -> None:
        fits: list[float | None] = list(fitnesses) if fitnesses is not None else [None] * len(trees)
        labels = ops if ops is not None else ["init"] * len(trees)
        self._current_gen_ids = [
            self._add(0, op, [], t, fit) for t, op, fit in zip(trees, labels, fits, strict=True)
        ]

    def on_generation(
        self, generation: int, entries: Sequence[tuple[Node, list[int], str, float]]
    ) -> None:
        previous = self._current_gen_ids
        new_ids = []
        for tree, parent_indices, op, *rest in entries:
            parent_ids = [previous[i] for i in parent_indices]
            fitness = float(rest[0]) if rest else None
            new_ids.append(self._add(generation, op, parent_ids, tree, fitness))
        self._current_gen_ids = new_ids

    def continue_from(self, generation: int) -> None:
        """Re-point the recorder at ``generation`` so a resumed run's parents resolve.

        After loading a persisted lineage to keep recording into (a warm-started session
        segment), the next ``on_generation`` must map parent indices onto the node ids of
        the last recorded generation - in population order, which is append order here.
        """
        self._current_gen_ids = [n.id for n in self.nodes if n.generation == generation]

    # --- queries ------------------------------------------------------------------
    def generations(self) -> list[list[Node]]:
        """Per-generation tree lists, in population order - the replayed run."""
        by_gen: dict[int, list[Node]] = {}
        for node in self.nodes:
            by_gen.setdefault(node.generation, []).append(node.tree)
        return [by_gen[g] for g in sorted(by_gen)]

    def replay(self) -> list[list[Node]]:
        return self.generations()

    # --- persistence --------------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "metadata": self.metadata,
            "nodes": [
                {
                    "id": n.id,
                    "generation": n.generation,
                    "op": n.op,
                    "parents": n.parents,
                    "tree": tree_to_dict(n.tree),
                    "fitness": n.fitness,
                }
                for n in self.nodes
            ],
        }

    def save(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.to_dict()), encoding="utf-8")
        return target

    @classmethod
    def load(cls, path: str | Path) -> LineageStore:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        store = cls(run_id=data.get("run_id", "run"))
        store.metadata = data.get("metadata", {})
        store.nodes = [
            LineageNode(
                n["id"],
                n["generation"],
                n["op"],
                n["parents"],
                tree_from_dict(n["tree"]),
                n.get("fitness"),
            )
            for n in data["nodes"]
        ]
        store._next_id = max((n.id for n in store.nodes), default=-1) + 1
        return store
