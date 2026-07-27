from __future__ import annotations

import numpy as np
from scripts.bench_training import _canonical_hash_value, _hash


def test_trajectory_hash_canonicalizes_nested_mixed_diagnostics() -> None:
    first = {
        "exploration": {
            "resolved": {"profile": "aggressive", "two_edit_fraction": 0.2},
            "stagnation_boost": False,
            "operation_successes": {"mutation": 3, "crossover": 2},
            "frontier": [None, np.float64(-0.0), np.int64(4)],
        },
        "generation": 2,
    }
    reordered = {
        "generation": 2,
        "exploration": {
            "frontier": [None, -0.0, 4],
            "operation_successes": {"crossover": 2, "mutation": 3},
            "stagnation_boost": False,
            "resolved": {"two_edit_fraction": 0.2, "profile": "aggressive"},
        },
    }

    canonical = _canonical_hash_value(first)

    assert _hash(first) == _hash(reordered)
    assert canonical["exploration"]["resolved"]["profile"] == "aggressive"
    assert canonical["exploration"]["stagnation_boost"] is False
    assert canonical["exploration"]["frontier"] == [
        None,
        {"__float_hex__": "-0x0.0p+0"},
        4,
    ]
    assert _hash([0.0]) != _hash([-0.0])
    assert _hash([1, 2]) != _hash([2, 1])
