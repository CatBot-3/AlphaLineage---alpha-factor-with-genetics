"""Run a GP alpha search and report an honest, deflated out-of-sample verdict.

Usage:
    python scripts/run_gp.py --config configs/dev.yaml
    python scripts/run_gp.py --config configs/dev.yaml --checkpoint run.json [--resume]

The search only ever sees the TRAIN split. The TEST split is locked and scored exactly once,
at final reporting (invariant 1). Headline metrics default to out-of-sample / deflated.

Not investment advice. No brokerage. Signals are research output only.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if _SRC.exists() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import yaml  # noqa: E402

from alphaforge.core.gp import GP, GPConfig  # noqa: E402
from alphaforge.core.panel import Panel  # noqa: E402
from alphaforge.data.universe import sample_universe  # noqa: E402
from alphaforge.validation.pipeline import LockedTestSet, judge  # noqa: E402
from alphaforge.validation.splits import Split, time_split  # noqa: E402

_DISCLAIMER = "Not investment advice. Research output only; signals must survive costs (Phase 4)."


def _train_panel(panel: Panel, split: Split) -> Panel:
    return Panel({f: df.loc[df.index.isin(split.train)] for f, df in panel.fields.items()})


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a GP alpha search with an OOS verdict.")
    parser.add_argument("--config", default="configs/dev.yaml")
    parser.add_argument("--universe", default=None, help="override the config universe")
    parser.add_argument("--checkpoint", default=None, help="checkpoint path (enables resume)")
    parser.add_argument("--resume", action="store_true", help="resume from --checkpoint if present")
    args = parser.parse_args(argv)

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    universe = args.universe or cfg.get("universe", "sp500-lite")
    as_of = cfg.get("as_of", "2026-06-01")
    gp_config = GPConfig.from_dict(cfg.get("gp", {}))

    symbols = sample_universe(universe).members_asof(as_of)
    try:
        panel = Panel.from_cache(symbols)
    except ValueError:
        print("No cached data. Run: python scripts/download_universe.py --universe", universe)
        return 1

    split = time_split(
        panel.dates,
        train=float(cfg.get("train", 0.6)),
        valid=float(cfg.get("valid", 0.2)),
        embargo=int(cfg.get("embargo", 5)),
    )
    print(f"universe={universe} symbols={list(panel.symbols)}")
    print(
        f"dates train={len(split.train)} valid={len(split.valid)} test={len(split.test)} (locked)\n"
    )

    ckpt = args.checkpoint
    train_panel = _train_panel(panel, split)
    if args.resume and ckpt and Path(ckpt).exists():
        gp = GP.from_checkpoint(ckpt, train_panel)
        print(f"resumed from {ckpt} at generation {gp.generation}")
    else:
        gp = GP(gp_config, train_panel)

    best = gp.run(checkpoint_path=ckpt)
    for row in gp.history:
        print(
            f"  gen {int(row['generation']):>3}  best={row['best_fitness']:.4f}  "
            f"mean={row['mean_fitness']:.4f}  ic={row['best_ic']:.4f}"
        )

    trials = [ind.tree for ind in gp.population]
    report = judge(
        best.tree,
        trials,
        split,
        panel,
        LockedTestSet(split.test),
        n_trials=gp.trial_count,
        min_names=gp_config.min_names,
    )

    print("\nbest factor:", best.tree)
    print(f"trials searched      = {report.n_trials}")
    print(f"train    |rank IC|   = {report.train_ic:.4f}")
    print(f"OOS test |rank IC|   = {report.oos_ic:.4f}   <- default, honest metric")
    print(f"deflated Sharpe      = {report.deflated_sharpe:.4f}   (>0.95 significant)")
    print(f"PBO                  = {report.pbo:.4f}   (>=0.5 overfit red flag)")
    verdict = "PLAUSIBLE" if report.significant else "NOT SIGNIFICANT (likely overfit / luck)"
    print(f"verdict              = {verdict}")
    print("\n" + _DISCLAIMER)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
