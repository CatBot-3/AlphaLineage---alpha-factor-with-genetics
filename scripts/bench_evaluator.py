"""Benchmark the C++ vs pure-Python evaluator on the hot path (for the writeup).

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

from alphaforge.core import cpp  # noqa: E402
from alphaforge.core.evaluate import evaluate_python  # noqa: E402
from alphaforge.core.generate import RandomTreeGenerator  # noqa: E402
from alphaforge.core.panel import Panel  # noqa: E402


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

    line = f"{t:>5} x {n:<3}  {len(trees):>4} trees | python {1e3 * py / len(trees):6.3f} ms/tree"
    if cpp.available():
        cpp.evaluate_cpp(trees[0], panel)  # warm the cached panel arrays
        start = time.perf_counter()
        for tr in trees:
            cpp.evaluate_cpp(tr, panel)
        cc = time.perf_counter() - start
        line += f" | c++ {1e3 * cc / len(trees):6.3f} ms/tree | speedup {py / cc:4.1f}x"
    print(f"{line}  ({note})")


def main() -> int:
    if not cpp.available():
        print("C++ extension not built — run `python scripts/build_cpp.py` to compare.\n")
    # The GP hot path: a small point-in-time universe, evaluated millions of times. Here the
    # per-op Python/pandas overhead dominates and C++ wins. On very large panels, pandas'
    # vectorized C rolling/rank wins — but that is not the workload the GP actually runs.
    _bench(252, 10, 2000, "GP hot path — small universe")
    _bench(500, 12, 1500, "small universe, longer history")
    _bench(2000, 200, 300, "large panel — pandas' vectorization wins")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
