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

import yaml  # type: ignore[import-untyped]  # noqa: E402

from alphalineage.api.resources import (  # noqa: E402
    TRAINING_SCHEDULER,
    ResourcePolicy,
    resolve_resources,
)
from alphalineage.api.service import build_report  # noqa: E402
from alphalineage.backtest.costs import TransactionCostModel  # noqa: E402
from alphalineage.backtest.portfolio import QuantileLongShort  # noqa: E402
from alphalineage.core import cpp  # noqa: E402
from alphalineage.core.gp import GP, GPConfig  # noqa: E402
from alphalineage.core.panel import Panel  # noqa: E402
from alphalineage.data.universe import sample_universe  # noqa: E402
from alphalineage.validation.splits import Split, time_split  # noqa: E402

_DISCLAIMER = "Not investment advice. Research output only; signals must survive costs (Phase 4)."


def _metric(value: object) -> str:
    return f"{value:.4f}" if isinstance(value, (int, float)) else "unavailable"


def _train_panel(panel: Panel, split: Split) -> Panel:
    return Panel({f: df.loc[df.index.isin(split.train)] for f, df in panel.fields.items()})


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a GP alpha search with an OOS verdict.")
    parser.add_argument("--config", default="configs/dev.yaml")
    parser.add_argument("--universe", default=None, help="override the config universe")
    parser.add_argument("--checkpoint", default=None, help="checkpoint path (enables resume)")
    parser.add_argument("--resume", action="store_true", help="resume from --checkpoint if present")
    parser.add_argument(
        "--resource-profile",
        choices=("light", "auto", "maximum", "custom"),
        default=None,
        help="portable CPU budget (default: config resources.profile or auto)",
    )
    parser.add_argument(
        "--cpu-budget-percent",
        type=int,
        default=None,
        help="10-100; implies --resource-profile custom when supplied alone",
    )
    args = parser.parse_args(argv)

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    universe = args.universe or cfg.get("universe", "sp500-lite")
    as_of = cfg.get("as_of", "2026-06-01")
    gp_config = GPConfig.from_dict(cfg.get("gp", {}))
    resource_cfg = cfg.get("resources", {}) or {}
    profile = args.resource_profile or resource_cfg.get("profile", "auto")
    custom_percent = (
        args.cpu_budget_percent
        if args.cpu_budget_percent is not None
        else resource_cfg.get("cpu_budget_percent")
    )
    if custom_percent is not None and args.resource_profile is None:
        profile = "custom"
    policy = ResourcePolicy(
        profile=profile,
        custom_percent=custom_percent if profile == "custom" else None,
    )
    accelerated = cpp.supports_native_scoring(gp_config.ic_method)
    if not cpp.backend_enabled():
        fallback_reason = "native evaluator unavailable or disabled"
    elif not accelerated:
        fallback_reason = (
            f"{gp_config.ic_method} scoring uses the parity-safe Python evaluator"
        )
    else:
        fallback_reason = None
    resources = resolve_resources(
        policy,
        accelerated=accelerated,
        fallback_reason=fallback_reason,
    )

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
        horizon=gp_config.horizon,
    )
    print(f"universe={universe} symbols={list(panel.symbols)}")
    print(
        f"dates train={len(split.train)} valid={len(split.valid)} test={len(split.test)} (locked)\n"
    )

    ckpt = args.checkpoint
    train_panel = _train_panel(panel, split)
    print(
        f"resources={resources.profile} {resources.percent}% "
        f"workers={resources.workers}/{resources.detected_cpus} "
        f"memory={resources.run_memory_budget_bytes / 1024**2:.0f} MiB"
    )
    if resources.fallback_reason:
        print(f"resource fallback: {resources.fallback_reason}")
    with TRAINING_SCHEDULER.acquire(resources) as lease:
        if args.resume and ckpt and Path(ckpt).exists():
            gp = GP.from_checkpoint(
                ckpt,
                train_panel,
                workers=lease.max_workers,
                memory_budget_bytes=lease.memory_budget_bytes,
            )
            print(f"resumed from {ckpt} at generation {gp.generation}")
        else:
            gp = GP(
                gp_config,
                train_panel,
                workers=lease.max_workers,
                memory_budget_bytes=lease.memory_budget_bytes,
            )
        best = gp.run(checkpoint_path=ckpt)
    for row in gp.history:
        print(
            f"  gen {int(row['generation']):>3}  best={row['best_fitness']:.4f}  "
            f"mean={row['mean_fitness']:.4f}  ic={row['best_ic']:.4f}"
        )

    trials = [ind.tree for ind in gp.population]
    bt = cfg.get("backtest", {})
    costs = TransactionCostModel(
        commission_bps=float(bt.get("commission_bps", 1.0)),
        slippage_bps=float(bt.get("slippage_bps", 5.0)),
    )
    scheme = QuantileLongShort(float(bt.get("quantile", 0.2)))

    print("\nbest factor:", best.tree)
    # ``build_report`` owns the sole LockedTestSet unlock and reuses that same factor evaluation
    # for the rich holdout report. Do not inspect or compare the test split before this call.
    report = build_report(
        best.tree,
        trials,
        split,
        panel,
        searched_trials=gp.trial_count,
        horizon=gp_config.horizon,
        ic_method=gp_config.ic_method,
        min_names=gp_config.min_names,
        scheme=scheme,
        costs=costs,
    )
    holdout = report.get("oos_backtest") or {}
    holdout_metrics = holdout.get("metrics") or {}
    print("\nlocked holdout portfolio (quantile long/short, net of costs):")
    print(
        "  "
        f"net Sharpe={_metric(holdout_metrics.get('net_sharpe'))}  "
        f"gross Sharpe={_metric(holdout_metrics.get('gross_sharpe'))}  "
        f"drawdown={_metric(holdout_metrics.get('max_drawdown'))}  "
        f"turnover={_metric(holdout_metrics.get('turnover'))}"
    )
    print(
        f"\ntrials searched      = {report['n_trials']}  "
        f"({gp.trial_count} factor candidates, one pinned scheme)"
    )
    print(f"train    |rank IC|   = {report['train_ic']:.4f}")
    print(f"OOS test |rank IC|   = {report['oos_ic']:.4f}   <- default, honest metric")
    print(f"net deflated Sharpe  = {report['deflated_sharpe']:.4f}   (>0.95 significant)")
    print(f"PBO (net)            = {report['pbo']:.4f}   (>=0.5 overfit red flag)")
    verdict = (
        "PLAUSIBLE"
        if report["significant"]
        else "NOT SIGNIFICANT (likely overfit / luck)"
    )
    print(f"verdict              = {verdict}")
    print("\n" + _DISCLAIMER)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
