"""Execution-timing sensitivity: how much predictive power survives a realistic trade time?

The search scores a factor computed from session ``t`` data (including the ``t`` close) against the
close-to-close return from ``t`` to ``t + 1``. That return is only earnable by trading *at* the
same close the factor already used. This diagnostic re-scores factors against the returns you
could actually capture with later execution:

* ``close_t -> close_t+1``   current scoring target (trade in the closing auction of day t)
* ``open_t+1 -> close_t+1``  trade at the next open, exit at that close (intraday only)
* ``open_t+1 -> open_t+2``   trade at the next open, hold one full day (market-on-open orders)
* ``close_t+1 -> close_t+2`` trade in the next day's closing auction (a one-day delay)

It reports mean daily rank IC, its t-statistic and the share of the current IC that survives. It
reads only the local Parquet cache, never the network, and it selects nothing: every published
alpha in the starter catalog is scored as written. By default it stops before 2021-04-13 so it
does not look at the locked test window of existing sessions.

Usage:
    python scripts/execution_timing_study.py --universe sp500-energy \
        --universe sp500-information-technology --start 2010-01-01 --end 2021-04-12
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

_SRC = Path(__file__).resolve().parents[1] / "src"
if _SRC.exists() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from alphalineage.core.evaluate import evaluate  # noqa: E402
from alphalineage.core.fitness import daily_ic  # noqa: E402
from alphalineage.core.panel import Panel  # noqa: E402
from alphalineage.core.tree import Node, from_dict  # noqa: E402
from alphalineage.data.cache import ParquetCache  # noqa: E402
from alphalineage.data.universe import bundled_snapshot_name, bundled_universe  # noqa: E402
from alphalineage.library.alpha_catalog import ALPHA_CATALOG  # noqa: E402

TARGETS = (
    "close_t->close_t+1",
    "open_t+1->close_t+1",
    "open_t+1->open_t+2",
    "close_t+1->close_t+2",
)


def forward_targets(panel: Panel) -> dict[str, pd.DataFrame]:
    close, open_ = panel["close"], panel["open"]
    return {
        "close_t->close_t+1": close.shift(-1) / close - 1.0,
        "open_t+1->close_t+1": close.shift(-1) / open_.shift(-1) - 1.0,
        "open_t+1->open_t+2": open_.shift(-2) / open_.shift(-1) - 1.0,
        "close_t+1->close_t+2": close.shift(-2) / close.shift(-1) - 1.0,
    }


def _bind_defaults(body: dict[str, Any], inputs: list[dict[str, Any]]) -> dict[str, Any]:
    if body["name"] == "$arg":
        item = inputs[int(body["value"])]
        kind = "window" if item["type"] == "window" else "const"
        value = int(item["default"]) if kind == "window" else float(item["default"])
        return {"name": kind, "value": value}
    return {**body, "children": [_bind_defaults(c, inputs) for c in body.get("children", [])]}


def catalog_trees() -> list[tuple[str, str, Node]]:
    return [
        (item["name"], item["family"], from_dict(_bind_defaults(item["body"], item["inputs"])))
        for item in ALPHA_CATALOG
    ]


def _t_stat(series: pd.Series) -> float:
    clean = series.dropna()
    if len(clean) < 3 or clean.std() == 0:
        return float("nan")
    return float(clean.mean() / clean.std() * np.sqrt(len(clean)))


def study(panel: Panel, *, min_names: int) -> pd.DataFrame:
    targets = forward_targets(panel)
    rows: list[dict[str, Any]] = []
    for name, family, tree in catalog_trees():
        factor = evaluate(tree, panel)
        if not isinstance(factor, pd.DataFrame):
            continue
        row: dict[str, Any] = {"factor": name, "family": family}
        for label in TARGETS:
            ic = daily_ic(factor, targets[label], "spearman", min_names=min_names)
            row[f"ic[{label}]"] = float(ic.mean())
            row[f"t[{label}]"] = _t_stat(ic)
        rows.append(row)
    frame = pd.DataFrame(rows).set_index("factor")
    base = frame["ic[close_t->close_t+1]"]
    for label in TARGETS[1:]:
        # Share of the current IC that survives, measured in the current IC's direction.
        frame[f"kept[{label}]"] = frame[f"ic[{label}]"] * np.sign(base) / base.abs()
    return frame


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--universe", action="append", default=[], help="bundled universe id/alias")
    parser.add_argument("--start", default="2010-01-01")
    parser.add_argument("--end", default="2021-04-12")
    parser.add_argument("--min-names", type=int, default=10)
    parser.add_argument("--min-abs-t", type=float, default=2.0, help="threshold for the summary")
    parser.add_argument("--out", default=None, help="optional CSV path for the full table")
    args = parser.parse_args(argv)

    names = args.universe or ["sp500-information-technology"]
    symbols: list[str] = []
    for name in names:
        canonical = bundled_snapshot_name(name)
        if canonical is None:
            raise SystemExit(f"unknown bundled universe {name!r}")
        symbols.extend(bundled_universe(canonical).all_symbols())
    cache = ParquetCache()
    cached = sorted({symbol for symbol in symbols if cache.has(symbol)})
    if len(cached) < args.min_names:
        raise SystemExit(f"only {len(cached)} cached symbols; download the universe first")
    panel = Panel.from_cache(cached, cache=cache, start=args.start, end=args.end)
    print(
        f"{len(cached)} cached symbols, {len(panel.dates)} sessions "
        f"{panel.dates.min().date()} to {panel.dates.max().date()}"
    )

    frame = study(panel, min_names=args.min_names)
    if args.out:
        frame.to_csv(args.out)
    significant = frame[frame["t[close_t->close_t+1]"].abs() >= args.min_abs_t]
    summary = {
        "symbols": len(cached),
        "sessions": len(panel.dates),
        "factors": len(frame),
        "significant_at_close_close": len(significant),
        "median_kept_among_significant": {
            label: float(significant[f"kept[{label}]"].median()) for label in TARGETS[1:]
        }
        if len(significant)
        else {},
        "median_kept_by_family": {
            family: {label: float(group[f"kept[{label}]"].median()) for label in TARGETS[1:]}
            for family, group in significant.groupby("family")
        },
    }
    pd.set_option("display.width", 200)
    columns = [f"ic[{label}]" for label in TARGETS] + ["t[close_t->close_t+1]"]
    print(
        frame.sort_values("t[close_t->close_t+1]", key=np.abs, ascending=False)[columns]
        .round(4)
        .head(25)
        .to_string()
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
