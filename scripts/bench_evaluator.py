"""Benchmark Python, scalar native, and bounded native batch evaluation.

Run: ``python scripts/bench_evaluator.py``. Builds a large synthetic panel and times a batch of
random expression trees through both backends; prints per-tree time and the speedup. If the C++
extension is not built, it reports that and exits (the Python baseline still runs).
"""

from __future__ import annotations

import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from alphalineage.core import cpp  # noqa: E402
from alphalineage.core.evaluate import evaluate_python  # noqa: E402
from alphalineage.core.generate import RandomTreeGenerator  # noqa: E402
from alphalineage.core.panel import Panel  # noqa: E402


def _panel(t: int, n: int) -> Panel:
    rng = np.random.default_rng(0)
    dates = pd.date_range("2010-01-01", periods=t, freq="B")
    syms = [f"S{i}" for i in range(n)]
    close = pd.DataFrame(
        100.0 + np.cumsum(rng.normal(0, 1, (t, n)), axis=0), index=dates, columns=syms
    )
    open_ = close.shift(1).fillna(close.iloc[0])
    vol = pd.DataFrame(rng.uniform(1e6, 5e6, (t, n)), index=dates, columns=syms)
    return Panel.from_prices(
        open=open_, high=close * 1.01, low=close * 0.99, close=close, volume=vol
    )


def _bench(t: int, n: int, n_trees: int, note: str) -> None:
    panel = _panel(t, n)
    gen = RandomTreeGenerator(random.Random(7), max_depth=5, max_nodes=30)
    # keep only C++-evaluable trees so both backends do the same work
    trees = [
        tr
        for tr in gen.ramped_half_and_half(n_trees * 3, min_depth=3, max_depth=5)
        if cpp.flatten(tr) is not None
    ][:n_trees]

    start = time.perf_counter()
    for tr in trees:
        evaluate_python(tr, panel)
    py = time.perf_counter() - start

    python_per_tree = py / len(trees)
    line = f"{t:>5} x {n:<3}  {len(trees):>4} trees | python {1e3 * python_per_tree:6.3f} ms/tree"
    if cpp.available():
        cpp.evaluate_cpp(trees[0], panel)  # warm the cached panel arrays
        start = time.perf_counter()
        for tr in trees:
            cpp.evaluate_cpp(tr, panel)
        cc = time.perf_counter() - start
        line += f" | c++ {1e3 * cc / len(trees):6.3f} ms/tree | speedup {py / cc:4.1f}x"
        # Keep retained DataFrame results below roughly 128 MiB for the large-panel case.
        frame_bytes = t * n * np.dtype(np.float64).itemsize
        batch_count = min(len(trees), max(1, (128 * 1024**2) // max(1, frame_bytes)))
        batch_trees = trees[:batch_count]
        workers = min(8, cpp.native_max_workers(), batch_count)
        start = time.perf_counter()
        results = cpp.evaluate_many(
            batch_trees, panel, workers=workers, memory_budget_bytes=256 * 1024**2
        )
        batch = time.perf_counter() - start
        assert all(result is not None for result in results)
        batch_per_tree = batch / batch_count
        line += (
            f" | batch-{workers} {1e3 * batch_per_tree:6.3f} ms/tree"
            f" | vs python {python_per_tree / batch_per_tree:4.1f}x"
        )
    print(f"{line}  ({note})")


def main() -> int:
    if not cpp.available():
        print("C++ extension not built - run `python scripts/build_cpp.py` to compare.\n")
    # The GP hot path is dominated by many small/medium panel evaluations. Keep one large-panel
    # case as a regression guard for the linear rolling and O(N log N) rank implementations.
    _bench(252, 10, 2000, "GP hot path - small universe")
    _bench(500, 12, 1500, "small universe, longer history")
    _bench(2000, 200, 300, "large-panel regression guard")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
