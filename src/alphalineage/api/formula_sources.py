"""Resolve immutable formula sources once for saving, inspection and Signals."""

from __future__ import annotations

import hashlib
import threading
from typing import Any

from fastapi import HTTPException

from alphalineage.core.extensions import expand_all
from alphalineage.core.tree import Node, from_dict, from_json, to_dict, to_json, validate
from alphalineage.core.types import DType

SAVE_LOCK = threading.Lock()


def api():
    # Late binding keeps the API's dependency overrides and operator lifecycle authoritative.
    from alphalineage.api import app

    return app


def resolve(
    source: str,
    bindings: dict[str, Any] | None = None,
    evaluation_id: str | None = None,
    *,
    formula_specs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    a = api()
    if source.startswith("evaluation:"):
        parts = source.split(":")
        if len(parts) != 3 or not a._session_exists(parts[1]):
            raise HTTPException(404, "Unknown holdout evaluation source.")
        artifact = a.sessions.load_finalization(parts[1], parts[2])
        if artifact is None:
            raise HTTPException(404, "Unknown holdout evaluation source.")
        return {
            **resolve(f"round:{parts[1]}:{artifact['round_index']}", evaluation_id=parts[2]),
            "id": source,
        }
    if source.startswith("lineage:"):
        parts = source.split(":")
        if len(parts) != 4 or not parts[2].isdigit() or not parts[3].isdigit():
            raise HTTPException(400, "Malformed lineage source.")
        _, sid, index, node_id = parts
        base = resolve(f"round:{sid}:{index}")
        lineage = a.sessions.lineage_for_round(sid, int(index)) or {}
        node = next(
            (item for item in lineage.get("nodes", []) if int(item["id"]) == int(node_id)), None
        )
        if node is None:
            raise HTTPException(404, "This node is not part of the saved round.")
        tree = from_json(node["tree"]) if isinstance(node["tree"], str) else from_dict(node["tree"])
        return finish(
            source,
            tree,
            name=f"{base['name']} · node {node_id}",
            provenance={
                **base["provenance"],
                "evidence": "training",
                "polarity": None,
                "validation_passed": None,
                "node_id": int(node_id),
            },
            configuration=base["configuration"],
            metrics={"fitness": node.get("fitness")},
            dependency_revisions=a._formula_dependency_revisions(tree),
            data_revision=base.get("data_revision"),
        )
    if source.startswith("round:"):
        parts = source.split(":")
        if len(parts) != 3 or not parts[2].isdigit():
            raise HTTPException(400, "Malformed round source.")
        sid, index = parts[1], int(parts[2])
        if not a._session_exists(sid):
            raise HTTPException(404, "Unknown session.")
        session = a.sessions.load_session(sid)
        result = a.sessions.load_round(sid, index)
        if result is None:
            raise HTTPException(404, "Unknown round.")
        a._ensure_session_operators(session)
        meta = result.get("round_metadata") or {}
        config = dict(meta.get("config") or session["config"])
        selection = result.get("selection") or meta.get("selection") or {}
        context = result.get("context") or {}
        evaluation = None
        if evaluation_id:
            evaluation = a.sessions.load_finalization(sid, evaluation_id)
            if evaluation is None or int(evaluation.get("round_index", -1)) != index:
                raise HTTPException(404, "The holdout evaluation does not belong to this round.")
        report = (evaluation or {}).get("report") or {}
        provenance = {
            "session_id": sid,
            "round_index": index,
            "evaluation_id": evaluation_id,
            "universe": context.get("universe", session["universe"]),
            "as_of": session["as_of"],
            "execution": config.get("execution", "close"),
            "horizon": config.get("horizon", 1),
            "polarity": 1,
            "cumulative_trials": result.get(
                "cumulative_trials", session.get("cumulative_trials", 0)
            ),
            "test_reads": result.get("test_reads", 0),
            "evidence": "holdout" if evaluation else "validation",
            "validation_passed": bool(selection.get("validated", False)),
            "history_start": session.get("history_start"),
        }
        tree = from_json(result["best_factor"])
        metrics = (
            dict(report)
            if report
            else {
                "validation_median_ic": selection.get("median_oriented_ic"),
                "validation_objective": selection.get("final_objective"),
            }
        )
        primary = next(
            (
                item
                for item in (evaluation or {}).get("context", {}).get("strategies", [])
                if item.get("id") == (evaluation or {}).get("primary_strategy_id")
            ),
            {},
        )
        if not evaluation:
            plans = [
                item
                for item in a.sessions.list_finalization_plans(sid)
                if item.get("round_index") == index
            ]
            if plans:
                plan = a.sessions.load_finalization_plan(sid, plans[-1]["strategy_plan_id"]) or {}
                primary = next(
                    (
                        item
                        for item in plan.get("strategies", [])
                        if item.get("id") == plan.get("primary_strategy_id")
                    ),
                    {},
                )
        return finish(
            source,
            tree,
            name=f"{session['name']} · Round {index + 1}",
            provenance=provenance,
            configuration={**config, "universe": provenance["universe"]},
            metrics=metrics,
            dependency_revisions=a._formula_dependency_revisions(tree),
            data_revision=meta.get("panel_fingerprint", ""),
            primary_strategy=primary,
            evaluation_id=evaluation_id,
        )
    if source.startswith("result:"):
        factor = a._factor_store().get(source.split(":", 1)[1])
        if factor is None:
            raise HTTPException(404, "Unknown saved result.")
        p, c = dict(factor.provenance), dict(factor.configuration)
        # Recover legacy metadata only from a unique matching immutable round.
        if ("execution" not in c or "universe" not in c) and p.get("session_id"):
            sid = str(p["session_id"])
            if a._session_exists(sid):
                matches = []
                for row in a.sessions.list_rounds(sid):
                    candidate = resolve(f"round:{sid}:{row['index']}")
                    if to_json(candidate["tree"]) == to_json(factor.expanded_tree or factor.tree):
                        matches.append(candidate)
                if len(matches) == 1:
                    c = {**matches[0]["configuration"], **c}
                    p = {**matches[0]["provenance"], **p}
        return finish(
            source,
            factor.expanded_tree or factor.tree,
            name=factor.name,
            provenance=p,
            configuration=c,
            metrics=factor.metrics,
            dependency_revisions=factor.dependency_revisions,
            bindings=factor.bindings,
            data_revision=factor.data_revision,
            primary_strategy=factor.source.get("primary_strategy", {}),
        )
    if source.startswith("formula:"):
        if formula_specs is None:
            a._load_persisted_formulas()
        runtime = source.split(":", 1)[1]
        spec = (
            a._formula_for_runtime(runtime) if formula_specs is None else formula_specs.get(runtime)
        )
        if spec is None or spec.status != "active":
            raise HTTPException(404, "This formula revision is unavailable.")
        args = []
        resolved_bindings = {}
        for item in spec.inputs:
            name = item.name
            kind = item.type
            provided = (bindings or {}).get(name)
            if provided is None and item.default is not None:
                provided = {"kind": "literal", "value": item.default}
            if provided is None:
                raise HTTPException(400, f"Choose a value for formula input '{name}'.")
            binding = a.FormulaBinding(**provided)
            args.append(a._formula_binding_node(binding, DType(kind), input_name=name))
            resolved_bindings[name] = binding.model_dump()
        tree = validate(Node(runtime, tuple(args)))
        return finish(
            source,
            tree,
            name=spec.display_name or spec.name,
            provenance={"evidence": "unvalidated", "polarity": None},
            configuration={},
            bindings=resolved_bindings,
            dependency_revisions=a._formula_dependency_revisions(tree, formula_specs),
        )
    raise HTTPException(400, "Choose a saved result, training round, or formula revision.")


def finish(source: str, tree: Node, **data: Any) -> dict[str, Any]:
    expanded = expand_all(tree)
    p, c = data.get("provenance", {}), data.get("configuration", {})
    execution = c.get("execution") or p.get("execution")
    return {
        **data,
        "id": source,
        "tree": expanded,
        "expression_fingerprint": hashlib.sha256(to_json(expanded).encode()).hexdigest(),
        "universe": c.get("universe") or p.get("universe"),
        "execution": execution,
        "horizon": c.get("horizon") or p.get("horizon"),
        "min_names": int(c.get("min_names") or p.get("min_names") or 5),
        "as_of": p.get("as_of"),
        "history_start": p.get("history_start") or c.get("start"),
        "evidence": p.get("evidence", "saved"),
        "validation_passed": p.get("validation_passed"),
        "polarity": p.get("polarity"),
        "missing_context": [
            key
            for key, value in {"execution": execution, "direction": p.get("polarity")}.items()
            if value is None
        ],
    }


def public(resolved: dict[str, Any]) -> dict[str, Any]:
    return {**resolved, "tree": to_dict(resolved["tree"])}


def save(
    source: str,
    name: str | None = None,
    evaluation_id: str | None = None,
    bindings: dict[str, Any] | None = None,
) -> dict[str, Any]:
    a = api()
    resolved = resolve(source, bindings, evaluation_id)
    if name is not None and not name.strip():
        raise HTTPException(400, "Enter a name for this result.")
    identity = {
        "id": source,
        "evaluation_id": evaluation_id,
        "expression_fingerprint": resolved["expression_fingerprint"],
        "bindings": resolved.get("bindings", {}),
    }
    with SAVE_LOCK:
        for existing in a._factor_store().list():
            if existing.source.get("identity") == identity:
                return existing.to_dict()
        saved = a._factor_store().save(
            name=(name or resolved["name"]).strip(),
            tree=resolved["tree"],
            metrics=resolved.get("metrics"),
            provenance=resolved["provenance"],
            configuration=resolved["configuration"],
            saved_at=a._now_iso(),
            source={"identity": identity, "primary_strategy": resolved.get("primary_strategy", {})},
            bindings=resolved.get("bindings"),
            dependency_revisions=resolved.get("dependency_revisions"),
            expression_fingerprint=resolved["expression_fingerprint"],
            data_revision=str(resolved.get("data_revision") or ""),
        )
    return saved.to_dict()


def freeze_references() -> dict[str, Any]:
    """No panels or labels are read here; expansion pins every dependency revision."""
    a = api()
    references, skipped = [], []
    latest = a._load_persisted_formulas()
    formula_specs = {spec.runtime_name: spec for spec in a._all_formula_specs()}
    sources = [f"result:{item.id}" for item in a._factor_store().list()]
    sources += [
        f"formula:{item['runtime_name']}"
        for item in latest
        if item.get("status", "active") == "active"
        and not item.get("error")
        and item.get("out_type") in {"series", "signal"}
    ]
    for source in sources:
        try:
            item = resolve(source, formula_specs=formula_specs)
            references.append(
                {
                    "key": source,
                    "name": item["name"],
                    "tree": to_dict(item["tree"]),
                    "dependency_revisions": item.get("dependency_revisions", {}),
                    "expression_fingerprint": item["expression_fingerprint"],
                }
            )
        except (HTTPException, ValueError, KeyError):
            skipped.append(source)
    import json

    return {
        "references": references,
        "skipped": skipped,
        "fingerprint": hashlib.sha256(json.dumps(references, sort_keys=True).encode()).hexdigest(),
    }
