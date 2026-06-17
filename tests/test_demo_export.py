"""PP-T2 acceptance: a finished run round-trips into a valid demo snapshot."""

from __future__ import annotations

import json

import pytest

from alphalineage.api.service import run_search
from alphalineage.core.gp import GPConfig
from alphalineage.library.export import InvalidDemo, build_demo, validate_demo


def test_demo_export_roundtrip(signal_panel, tmp_path):
    panel, _ = signal_panel
    config = GPConfig(population_size=16, generations=3, max_depth=4, max_nodes=20, seed=0)
    result = run_search(config, panel)

    demo = build_demo(result)
    validate_demo(demo)  # does not raise

    path = tmp_path / "demo-run.json"
    path.write_text(json.dumps(demo), encoding="utf-8")
    reloaded = json.loads(path.read_text(encoding="utf-8"))
    assert reloaded == demo  # serialization is lossless

    # the exported lineage carries fitness (powers the grouped genealogy view)
    assert all(node.get("fitness") is not None for node in demo["lineage"]["nodes"])


def test_validate_demo_rejects_incomplete_report():
    incomplete = {
        "best_factor": '{"name":"close"}',
        "report": {"oos_ic": 0.1},
        "lineage": {"nodes": []},
    }
    with pytest.raises(InvalidDemo):
        validate_demo(incomplete)


def test_build_demo_requires_run_fields():
    with pytest.raises(InvalidDemo):
        build_demo({"report": {}})
