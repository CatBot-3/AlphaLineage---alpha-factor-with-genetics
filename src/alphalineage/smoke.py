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
#: 11 GICS sectors + 20 sub-industry themes, all filtered from the S&P 500 snapshot.
_EXPECTED_CLASSIFICATION_SUBSETS = 31
#: 37 technical indicators (36 active) + 59 published alpha factors (22 WQ101, 24 Alpha158, 13
#: academic anomalies).
_EXPECTED_STARTER_FORMULAS = 96
_EXPECTED_ACTIVE_STARTER_FORMULAS = 95


def release_smoke() -> dict[str, Any]:
    """Fail fast when a release omits its accelerator or packaged catalogs."""
    if not cpp.available():
        raise RuntimeError(f"native evaluator unavailable: {cpp.unavailable_reason()}")
    native_abi = cpp.native_abi_version()
    if native_abi is None or native_abi < 6:
        raise RuntimeError(
            f"native evaluator ABI {native_abi!r} lacks directional scoring metrics"
        )

    specs = {str(item["id"]): item for item in bundled_snapshot_specs()}
    index_specs = {name for name, item in specs.items() if not item.get("parent")}
    if index_specs != _EXPECTED_SNAPSHOTS.keys():
        raise RuntimeError(f"unexpected bundled universe set: {sorted(index_specs)}")
    counts = {
        name: len(bundled_universe(name).all_symbols()) for name in _EXPECTED_SNAPSHOTS
    }
    if counts != _EXPECTED_SNAPSHOTS:
        raise RuntimeError(f"bundled universe counts do not match manifest: {counts}")
    # Loading verifies each sector/theme subset against its manifest count and checksum.
    subsets = sorted(name for name in specs if name not in index_specs)
    for name in subsets:
        bundled_universe(name)
    if len(subsets) != _EXPECTED_CLASSIFICATION_SUBSETS:
        raise RuntimeError(
            f"expected {_EXPECTED_CLASSIFICATION_SUBSETS} sector/theme subsets, "
            f"found {len(subsets)}"
        )
    if (
        len(CATALOG_NAMES) != _EXPECTED_STARTER_FORMULAS
        or len(ACTIVE_CATALOG_NAMES) != _EXPECTED_ACTIVE_STARTER_FORMULAS
    ):
        raise RuntimeError(
            f"expected {_EXPECTED_STARTER_FORMULAS} starter formulas "
            f"({_EXPECTED_ACTIVE_STARTER_FORMULAS} active), found "
            f"{len(CATALOG_NAMES)} ({len(ACTIVE_CATALOG_NAMES)} active)"
        )

    return {
        "native_evaluator": True,
        "native_abi": native_abi,
        "bundled_universes": counts,
        "classification_subsets": len(subsets),
        "starter_formulas": len(CATALOG_NAMES),
        "active_starter_formulas": len(ACTIVE_CATALOG_NAMES),
    }


if __name__ == "__main__":  # pragma: no cover - exercised by wheel and Docker smoke commands
    print(json.dumps(release_smoke(), sort_keys=True))
