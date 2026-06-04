"""Run a GP alpha search over the cached panel.

Usage:
    python scripts/run_gp.py --config configs/dev.yaml
    python scripts/run_gp.py --config configs/dev.yaml --checkpoint run.json [--resume]

Fits factors by mean |rank IC| on a leading time-ordered TRAIN slice; the trailing
slice is a monitoring-only holdout. The locked out-of-sample TEST split arrives in
Phase 3 and is never created here.

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

from alphaforge.core.evaluate import evaluate  # noqa: E402
from alphaforge.core.fitness import daily_ic, forward_returns, ic_ir  # noqa: E402
from alphaforge.core.gp import GP, GPConfig  # noqa: E402
from alphaforge.core.panel import Panel  # noqa: E402
from alphaforge.data.universe import sample_universe  # noqa: E402

_DISCLAIMER = "Not investment advice. Research output only; signals must survive costs (Phase 4)."


def _split(panel: Panel, train_fraction: float) -> tuple[Panel, Panel]:
    k = max(1, int(len(panel.dates) * train_fraction))
    train = Panel({f: df.iloc[:k] for f, df in panel.fields.items()})
    holdout = Panel({f: df.iloc[k:] for f, df in panel.fields.items()})
    return train, holdout


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a GP alpha search.")
    parser.add_argument("--config", default="configs/dev.yaml")
    parser.add_argument("--universe", default=None, help="override the config universe")
    parser.add_argument("--checkpoint", default=None, help="checkpoint path (enables resume)")
    parser.add_argument("--resume", action="store_true", help="resume from --checkpoint if present")
    args = parser.parse_args(argv)

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    universe = args.universe or cfg.get("universe", "sp500-lite")
    as_of = cfg.get("as_of", "2026-06-01")
    train_fraction = float(cfg.get("train_fraction", 0.7))
    gp_config = GPConfig.from_dict(cfg.get("gp", {}))

    symbols = sample_universe(universe).members_asof(as_of)
    try:
        panel = Panel.from_cache(symbols)
    except ValueError:
        print("No cached data. Run: python scripts/download_universe.py --universe", universe)
        return 1
    train, holdout = _split(panel, train_fraction)
    print(f"universe={universe} symbols={list(panel.symbols)}")
    print(f"train dates={len(train.dates)} holdout dates={len(holdout.dates)}\n")

    ckpt = args.checkpoint
    if args.resume and ckpt and Path(ckpt).exists():
        gp = GP.from_checkpoint(ckpt, train)
        print(f"resumed from {ckpt} at generation {gp.generation}")
    else:
        gp = GP(gp_config, train)

    best = gp.run(checkpoint_path=ckpt)
    for row in gp.history:
        print(
            f"  gen {int(row['generation']):>3}  "
            f"best={row['best_fitness']:.4f}  mean={row['mean_fitness']:.4f}  "
            f"ic={row['best_ic']:.4f}"
        )

    # Evaluate on the FULL panel so trailing windows warm up across the split boundary
    # (legitimate — uses only past data), then score on the holdout dates.
    factor_full = evaluate(best.tree, panel)
    fwd_full = forward_returns(panel, gp.config.horizon)
    hd = holdout.dates
    h_ic = daily_ic(
        factor_full.loc[hd], fwd_full.loc[hd], gp.config.ic_method, min_names=gp.config.min_names
    )
    holdout_ic = float(h_ic.abs().mean()) if h_ic.notna().any() else 0.0

    train_ic, train_ir = best.metrics.get("ic", 0.0), best.metrics.get("ic_ir", 0.0)
    print("\nbest factor:", best.tree)
    print(f"train   |rank IC| = {train_ic:.4f}  (IC IR {train_ir:.3f})")
    print(f"holdout |rank IC| = {holdout_ic:.4f}  (IC IR {ic_ir(h_ic):.3f})")
    print("(Phase 3 adds the locked OOS test, deflated Sharpe, and PBO to judge this honestly.)")
    print("\n" + _DISCLAIMER)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
