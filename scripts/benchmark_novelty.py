"""Reproducible native/Python novelty acceptance benchmark; no network or user data.

Run: .venv/Scripts/python scripts/benchmark_novelty.py --output novelty-benchmark.json
Each measurement uses a fresh process; peak RSS includes reference preparation and validation.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd
import psutil

from alphalineage.core.fitness import forward_returns
from alphalineage.core.gp import GP, GPConfig
from alphalineage.core.panel import Panel
from alphalineage.core.tree import Node, to_dict
from alphalineage.validation.selection import select_validation_candidate


def measurement(symbols, scenario, mode, count, catalog=False):
    rng = np.random.default_rng(712)
    dates = pd.bdate_range("2019-01-01", periods=504)
    names = [f"S{i}" for i in range(symbols)]
    close = pd.DataFrame(
        100 * np.exp(np.cumsum(rng.normal(0, 0.01, (len(dates), symbols)), axis=0)),
        index=dates,
        columns=names,
    )
    volume = pd.DataFrame(rng.uniform(1e5, 1e7, close.shape), index=dates, columns=names)
    panel = Panel.from_prices(
        open=close, high=close * 1.01, low=close * 0.99, close=close, volume=volume
    )
    train = Panel({name: frame.iloc[:336] for name, frame in panel.fields.items()})
    refs = [
        {"key": "known:volume", "tree": to_dict(Node("volume"))},
        {"key": "known:close", "tree": to_dict(Node("close"))},
    ]
    if catalog:
        with tempfile.TemporaryDirectory() as directory:
            os.environ["ALPHALINEAGE_DATA_DIR"] = directory
            os.environ["ALPHALINEAGE_SKIP_DOTENV"] = "1"
            from alphalineage.api.formula_sources import freeze_references

            refs = freeze_references()["references"]
    trees = []
    for i in range(count):
        window = Node("window", value=2 + i)
        if scenario == "duplicate-heavy":
            # Different monotone transforms produce the same cross-sectional ordering.
            trees.append(Node("mul", (Node("volume"), Node("const", value=1 + i))))
        elif scenario == "ordinary":
            trees.append(Node("ts_corr", (Node("returns"), Node("volume"), window)))
        else:
            trees.append(
                Node(
                    "ts_corr",
                    (
                        Node("returns"),
                        Node("delay", (Node("volume"), window)),
                        Node("window", value=10),
                    ),
                )
            )
    rss = []
    done = threading.Event()

    def sample():
        process = psutil.Process()
        while not done.wait(0.01):
            rss.append(process.memory_info().rss)

    monitor = threading.Thread(target=sample, daemon=True)
    monitor.start()
    started = time.perf_counter()
    gp = GP(
        GPConfig(population_size=count, generations=1, max_nodes=20, novelty_mode=mode),
        train,
        workers=1,
        memory_budget_bytes=512 * 1024 * 1024,
        novelty_state={"references": refs},
    )
    gp._individuals(trees, phase="initializing")
    trained = time.perf_counter()
    candidates = gp.validation_candidates()
    select_validation_candidate(
        candidates,
        panel,
        forward_returns(panel),
        dates[340:],
        min_names=5,
        validation_folds=3,
        fold_embargo=1,
    )
    elapsed = time.perf_counter() - started
    done.set()
    monitor.join()
    return {
        "symbols": symbols,
        "scenario": scenario,
        "mode": mode,
        "reference_count": len(refs),
        "candidates": count,
        "label_evaluations": gp.trial_count,
        "structural_skips": gp.structural_skips,
        "validation_candidates": len(candidates),
        "training_seconds": trained - started,
        "elapsed_seconds": elapsed,
        "peak_rss_mb": max(rss, default=0) / 1024**2,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="novelty-benchmark.json")
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--one", nargs=3)
    parser.add_argument("--catalog", action="store_true")
    parser.add_argument(
        "--scenarios", nargs="+", default=["duplicate-heavy", "ordinary", "low-overlap"]
    )
    args = parser.parse_args()
    if args.one:
        print(
            json.dumps(
                measurement(int(args.one[0]), args.one[1], args.one[2], args.count, args.catalog)
            )
        )
        return
    results = []
    for symbols in [100, 200, 500]:
        for scenario in args.scenarios:
            for mode in ["off", "balanced"]:
                command = [
                    sys.executable,
                    __file__,
                    "--count",
                    str(args.count),
                    "--one",
                    str(symbols),
                    scenario,
                    mode,
                ]
                if args.catalog:
                    command.append("--catalog")
                row = json.loads(subprocess.check_output(command, text=True))
                results.append(row)
                Path(args.output).write_text(json.dumps(results, indent=2), encoding="utf-8")
                print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
