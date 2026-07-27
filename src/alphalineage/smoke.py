"""Release-image smoke check for native evaluation and packaged research resources."""

from __future__ import annotations

import json
from typing import Any

from alphalineage.core import cpp
from alphalineage.data.universe import bundled_snapshot_specs, bundled_universe
from alphalineage.library.indicator_catalog import ACTIVE_CATALOG_NAMES, CATALOG_NAMES

_EXPECTED_SNAPSHOTS = {
    "builtin-sp500-current": 503,
    "builtin-djia-current": 30,
    "builtin-nasdaq100-current": 103,
}


def release_smoke() -> dict[str, Any]:
    """Fail fast when a release omits its accelerator or packaged catalogs."""
    if not cpp.available():
        raise RuntimeError(f"native evaluator unavailable: {cpp.unavailable_reason()}")
    native_abi = cpp.native_abi_version()
    if native_abi is None or native_abi < 5:
        raise RuntimeError(
            f"native evaluator ABI {native_abi!r} lacks the catalog-v2 cumulative kernel"
        )

    specs = {str(item["id"]): item for item in bundled_snapshot_specs()}
    if specs.keys() != _EXPECTED_SNAPSHOTS.keys():
        raise RuntimeError(f"unexpected bundled universe set: {sorted(specs)}")
    counts = {
        name: len(bundled_universe(name).all_symbols()) for name in _EXPECTED_SNAPSHOTS
    }
    if counts != _EXPECTED_SNAPSHOTS:
        raise RuntimeError(f"bundled universe counts do not match manifest: {counts}")
    if len(CATALOG_NAMES) != 37 or len(ACTIVE_CATALOG_NAMES) != 36:
        raise RuntimeError(
            "expected 37 starter formulas (36 active), found "
            f"{len(CATALOG_NAMES)} ({len(ACTIVE_CATALOG_NAMES)} active)"
        )

    return {
        "native_evaluator": True,
        "native_abi": native_abi,
        "bundled_universes": counts,
        "starter_formulas": len(CATALOG_NAMES),
        "active_starter_formulas": len(ACTIVE_CATALOG_NAMES),
    }


if __name__ == "__main__":  # pragma: no cover - exercised by wheel and Docker smoke commands
    print(json.dumps(release_smoke(), sort_keys=True))
