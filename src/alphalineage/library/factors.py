"""A3 - the saved-factor library: a user's keepers, reusable as session seeds.

A ``SavedFactor`` is a named expression tree plus its research metrics and provenance
(which session/generation/universe produced it, and the cumulative trial/test-read
counts at save time, so honesty accounting carries across sessions). It also embeds the
``OperatorSpec`` of every user macro the tree references, so the factor is fully
self-describing: seeding a new session can re-register exactly the operators it needs as
data (invariant 5), with no dependence on the registry's current state.

One JSON file per factor, under a user-configurable directory (see ``paths.factors_dir``).
Not investment advice. Research output only (invariant 8).
"""

from __future__ import annotations

import hashlib
import json
import re
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from alphalineage.core.extensions import (
    MAX_EXPANDED_DEPTH,
    MAX_EXPANDED_NODES,
    USER_OPERATORS,
    InvalidOperator,
    expand,
)
from alphalineage.core.tree import Node
from alphalineage.core.tree import from_dict as tree_from_dict
from alphalineage.core.tree import to_dict as tree_to_dict
from alphalineage.data.identifiers import atomic_write_text, child_path

DISCLAIMER = "Not investment advice. Research output only."


def _operator_spec(name: str) -> dict[str, Any]:
    prim = USER_OPERATORS[name]
    spec = {
        "name": prim.name,
        "arg_types": [t.value for t in prim.arg_types],
        "out_type": prim.out_type.value,
        "body": tree_to_dict(prim.macro_body),
    }
    if prim.macro_policy is not None:
        spec["policy"] = prim.macro_policy
    return spec


def required_operators(tree: Node) -> list[dict[str, Any]]:
    """The user-operator specs a tree depends on, in dependency order (leaves first).

    Recurses through macro bodies so nested user operators are captured too; the
    resulting order is safe to re-register sequentially.
    """
    specs: dict[str, dict[str, Any]] = {}

    active: list[str] = []

    def scan(node: Node) -> None:
        for sub in node.iter_nodes():
            if sub.name in USER_OPERATORS and sub.name not in specs:
                if sub.name in active:
                    start = active.index(sub.name)
                    cycle = [*active[start:], sub.name]
                    raise InvalidOperator(f"formula dependency cycle: {' -> '.join(cycle)}")
                active.append(sub.name)
                scan(USER_OPERATORS[sub.name].macro_body)  # dependencies first
                active.pop()
                specs[sub.name] = _operator_spec(sub.name)

    scan(tree)
    return list(specs.values())


def expanded_snapshot(tree: Node, specs: list[dict[str, Any]]) -> Node:
    """Expand a factor with its embedded operator specs, independent of the live registry."""
    bodies = {item["name"]: tree_from_dict(item["body"]) for item in specs}
    emitted = 0

    def visit(node: Node, depth: int = 1, active: tuple[str, ...] = ()) -> Node:
        nonlocal emitted
        if depth > MAX_EXPANDED_DEPTH:
            raise InvalidOperator(
                f"expanded expression depth exceeds the limit of {MAX_EXPANDED_DEPTH}"
            )
        body = bodies.get(node.name)
        if body is not None:
            if node.name in active:
                start = active.index(node.name)
                cycle = (*active[start:], node.name)
                raise InvalidOperator(f"formula dependency cycle: {' -> '.join(cycle)}")
            return visit(expand(node, body), depth, (*active, node.name))
        emitted += 1
        if emitted > MAX_EXPANDED_NODES:
            raise InvalidOperator(
                f"expanded expression size exceeds the limit of {MAX_EXPANDED_NODES} nodes"
            )
        return Node(
            node.name,
            tuple(visit(child, depth + 1, active) for child in node.children),
            node.value,
        )

    return visit(tree)


@dataclass
class SavedFactor:
    id: str
    name: str
    saved_at: str
    tree: Node
    metrics: dict[str, Any] = field(default_factory=dict)
    provenance: dict[str, Any] = field(default_factory=dict)
    required_operators: list[dict[str, Any]] = field(default_factory=list)
    expanded_tree: Node | None = None
    notes: str = ""
    disclaimer: str = DISCLAIMER
    # Canonical Formula Result fields.  Old saved-factor files omit these and load as a
    # training result; backtest results persist the same self-contained expression snapshot.
    kind: str = "training"
    source: dict[str, Any] = field(default_factory=dict)
    bindings: dict[str, Any] = field(default_factory=dict)
    dependency_revisions: list[Any] = field(default_factory=list)
    expression_fingerprint: str = ""
    configuration: dict[str, Any] = field(default_factory=dict)
    data_coverage: dict[str, Any] = field(default_factory=dict)
    data_revision: str = ""
    returns: list[dict[str, Any]] = field(default_factory=list)
    normalized_equity: list[dict[str, Any]] = field(default_factory=list)
    out_type: str = ""

    def to_dict(self) -> dict[str, Any]:
        out_type = self.out_type
        if not out_type:
            try:
                out_type = (self.expanded_tree or self.tree).out_type.value
            except KeyError:
                out_type = ""
        source_formula = self.source.get("runtime_name") if self.source else None
        dependency_fingerprint = (
            hashlib.sha256(
                json.dumps(
                    self.dependency_revisions,
                    separators=(",", ":"),
                    sort_keys=True,
                ).encode("utf-8")
            )
            .hexdigest()
            if self.dependency_revisions
            else ""
        )
        return {
            "id": self.id,
            "name": self.name,
            "saved_at": self.saved_at,
            "tree": tree_to_dict(self.tree),
            "metrics": self.metrics,
            "provenance": self.provenance,
            "required_operators": self.required_operators,
            "expanded_tree": tree_to_dict(self.expanded_tree or self.tree),
            "notes": self.notes,
            "disclaimer": self.disclaimer,
            "kind": self.kind,
            "source": self.source,
            "bindings": self.bindings,
            "dependency_revisions": self.dependency_revisions,
            "expression_fingerprint": self.expression_fingerprint,
            "configuration": self.configuration,
            "data_coverage": self.data_coverage,
            "data_revision": self.data_revision,
            "returns": self.returns,
            "normalized_equity": self.normalized_equity,
            "out_type": out_type,
            "source_formula": source_formula,
            "dependency_fingerprint": dependency_fingerprint,
            "universe": self.configuration.get("universe"),
            "start": self.configuration.get("start"),
            "end": self.configuration.get("end"),
            "horizon": self.configuration.get("horizon"),
            "weighting_scheme": self.configuration.get("weighting_scheme"),
            "quantile": self.configuration.get("quantile"),
            "commission_bps": self.configuration.get("commission_bps"),
            "slippage_bps": self.configuration.get("slippage_bps"),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SavedFactor:
        required = data.get("required_operators", [])
        tree = tree_from_dict(data["tree"])
        return cls(
            id=data["id"],
            name=data["name"],
            saved_at=data["saved_at"],
            tree=tree,
            metrics=data.get("metrics", {}),
            provenance=data.get("provenance", {}),
            required_operators=required,
            expanded_tree=(
                tree_from_dict(data["expanded_tree"])
                if data.get("expanded_tree")
                else expanded_snapshot(tree, required)
            ),
            notes=data.get("notes", ""),
            disclaimer=data.get("disclaimer", DISCLAIMER),
            kind=data.get("kind", "training"),
            source=data.get("source", {}),
            bindings=data.get("bindings", {}),
            dependency_revisions=data.get("dependency_revisions", []),
            expression_fingerprint=data.get("expression_fingerprint", ""),
            configuration=data.get("configuration", {}),
            data_coverage=data.get("data_coverage", {}),
            data_revision=data.get("data_revision", ""),
            returns=data.get("returns", []),
            normalized_equity=data.get("normalized_equity", []),
            out_type=data.get("out_type", ""),
        )


def _slug(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "-", name.strip().lower()).strip("-._")
    return f"{cleaned or 'factor'}-{uuid.uuid4().hex[:8]}"


class FactorStore:
    """A directory of saved factors, one ``{id}.json`` file each."""

    def __init__(self, directory: Path | str) -> None:
        self.directory = Path(directory)

    def _path(self, factor_id: str) -> Path:
        return child_path(self.directory, factor_id, ".json", label="factor id")

    def save(
        self,
        *,
        name: str,
        tree: Node,
        metrics: dict[str, Any] | None = None,
        provenance: dict[str, Any] | None = None,
        notes: str = "",
        saved_at: str = "",
        kind: str = "training",
        source: dict[str, Any] | None = None,
        bindings: dict[str, Any] | None = None,
        dependency_revisions: list[Any] | None = None,
        expression_fingerprint: str = "",
        configuration: dict[str, Any] | None = None,
        data_coverage: dict[str, Any] | None = None,
        data_revision: str = "",
        returns: list[dict[str, Any]] | None = None,
        normalized_equity: list[dict[str, Any]] | None = None,
        out_type: str = "",
    ) -> SavedFactor:
        operator_specs = required_operators(tree)
        factor = SavedFactor(
            id=_slug(name),
            name=name,
            saved_at=saved_at,
            tree=tree,
            metrics=metrics or {},
            provenance=provenance or {},
            required_operators=operator_specs,
            expanded_tree=expanded_snapshot(tree, operator_specs),
            notes=notes,
            kind=kind,
            source=source or {},
            bindings=bindings or {},
            dependency_revisions=dependency_revisions or [],
            expression_fingerprint=expression_fingerprint,
            configuration=configuration or {},
            data_coverage=data_coverage or {},
            data_revision=data_revision,
            returns=returns or [],
            normalized_equity=normalized_equity or [],
            out_type=out_type,
        )
        self.directory.mkdir(parents=True, exist_ok=True)
        atomic_write_text(
            self._path(factor.id), json.dumps(factor.to_dict(), indent=2, sort_keys=True)
        )
        return factor

    def list(self) -> list[SavedFactor]:
        if not self.directory.exists():
            return []
        factors = []
        for path in self.directory.glob("*.json"):
            try:
                factors.append(SavedFactor.from_dict(json.loads(path.read_text(encoding="utf-8"))))
            except (OSError, json.JSONDecodeError, KeyError):
                continue
        return sorted(factors, key=lambda f: f.saved_at, reverse=True)

    def get(self, factor_id: str) -> SavedFactor | None:
        path = self._path(factor_id)
        if not path.exists():
            return None
        return SavedFactor.from_dict(json.loads(path.read_text(encoding="utf-8")))

    def delete(self, factor_id: str) -> bool:
        path = self._path(factor_id)
        if not path.exists():
            return False
        path.unlink()
        return True

    def update(
        self, factor_id: str, *, name: str | None = None, notes: str | None = None
    ) -> SavedFactor | None:
        factor = self.get(factor_id)
        if factor is None:
            return None
        if name is not None:
            factor.name = name
        if notes is not None:
            factor.notes = notes
        atomic_write_text(
            self._path(factor_id), json.dumps(factor.to_dict(), indent=2, sort_keys=True)
        )
        return factor
