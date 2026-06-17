"""PP-T2 - export a finished run to the static `demo` JSON the zero-backend build reads.

The frontend's `demo` target renders ``frontend/public/demo-run.json`` with no backend at all.
That file is exactly a ``RunResult`` (best factor, OOS/deflated report, history, lineage), so
exporting is a project + validate step: pull the run out of a saved workspace or session and
check it has everything the static UI needs (parseable factor tree, the honest report keys, a
lineage whose nodes carry fitness for the grouped genealogy view).
"""

from __future__ import annotations

from typing import Any

from alphalineage.core.tree import from_dict, from_json

_REPORT_KEYS = {"oos_ic", "deflated_sharpe", "pbo", "train_ic", "n_trials", "significant"}


class InvalidDemo(ValueError):
    """Raised when a run cannot produce a valid demo snapshot."""


def build_demo(run: dict[str, Any]) -> dict[str, Any]:
    """Project a ``RunResult``-shaped dict into the demo snapshot the static build consumes."""
    try:
        return {
            "best_factor": run["best_factor"],
            "report": run["report"],
            "generations": run["generations"],
            "history": run["history"],
            "lineage": run["lineage"],
        }
    except KeyError as exc:
        raise InvalidDemo(f"run is missing required field {exc}") from exc


def validate_demo(data: dict[str, Any]) -> None:
    """Assert ``data`` is a renderable demo snapshot; raise ``InvalidDemo`` otherwise."""
    report = data.get("report")
    if not isinstance(report, dict) or not _REPORT_KEYS <= set(report):
        missing = _REPORT_KEYS - set(report or {})
        raise InvalidDemo(f"report missing keys: {sorted(missing)}")

    best = data.get("best_factor")
    try:
        from_json(best) if isinstance(best, str) else from_dict(best or {})
    except Exception as exc:  # noqa: BLE001 - any unparseable tree is invalid
        raise InvalidDemo(f"best_factor is not a parseable tree: {exc}") from exc

    lineage = data.get("lineage")
    if not isinstance(lineage, dict) or not isinstance(lineage.get("nodes"), list):
        raise InvalidDemo("lineage.nodes must be a list")
    for node in lineage["nodes"]:
        for key in ("id", "generation", "op", "parents", "tree"):
            if key not in node:
                raise InvalidDemo(f"lineage node missing {key!r}")
        try:
            from_dict(node["tree"])
        except Exception as exc:  # noqa: BLE001
            node_id = node.get("id")
            raise InvalidDemo(f"lineage node {node_id} has an unparseable tree: {exc}") from exc
