"""Daily Signals jobs. Calculations are frozen once; charts and portfolios read the snapshot."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import threading
import uuid
from datetime import UTC, datetime
from typing import Any

import exchange_calendars as xcals
import numpy as np
import pandas as pd
from fastapi import HTTPException

from alphalineage.api.errors import ActionError
from alphalineage.api.formula_sources import api, public, resolve
from alphalineage.api.progress import SyncProgress
from alphalineage.backtest.portfolio import PortfolioStrategySpec
from alphalineage.core.evaluate import evaluate
from alphalineage.core.gp import TrainingCancelled
from alphalineage.core.panel import Panel
from alphalineage.data import paths
from alphalineage.data.cache import ParquetCache
from alphalineage.data.identifiers import atomic_write_text, child_path
from alphalineage.explain.anatomy import window_profile
from alphalineage.library.signals import snapshot

SUBMIT_LOCK = threading.Lock()


def completed_session(as_of: str | None = None, now: datetime | None = None) -> str:
    instant = pd.Timestamp(now or datetime.now(UTC))
    if instant.tzinfo is None:
        instant = instant.tz_localize("UTC")
    cutoff = pd.Timestamp(as_of or instant.tz_convert("America/New_York").date()).normalize()
    cal = xcals.get_calendar(
        "XNYS",
        start=str((cutoff - pd.Timedelta(days=20)).date()),
        end=str((cutoff + pd.Timedelta(days=5)).date()),
    )
    schedule = cal.schedule.loc[str((cutoff - pd.Timedelta(days=15)).date()) : str(cutoff.date())]
    done = schedule.index[schedule["close"] <= instant]
    if not len(done):
        raise ActionError("calendar_unavailable", "No completed market session in this range.")
    return pd.Timestamp(done[-1]).date().isoformat()


def preparation(
    request: Any, resolved: list[dict[str, Any]], default_panel: Panel | None = None
) -> dict[str, Any]:
    a = api()
    universe_name = request.universe or resolved[0].get("universe") or a._DEFAULT_UNIVERSE
    try:
        universe = a._resolve_universe(universe_name)
    except (ValueError, HTTPException) as exc:
        if "membership" not in str(getattr(exc, "detail", exc)).lower():
            raise
        raise ActionError(
            "invalid_membership_metadata",
            "The universe has invalid membership dates.",
            action="Correct membership dates in Build & Data",
            details=str(getattr(exc, "detail", exc)),
        ) from exc
    end = completed_session(request.as_of)
    profiles = [window_profile(item["tree"]) for item in resolved]
    recursive = any(p.unbounded_lookback or p.recursive_smoothing for p in profiles)
    start = pd.Timestamp(end) - pd.DateOffset(years=3)
    approximate = False
    for source, profile in zip(resolved, profiles, strict=True):
        start = min(
            start, pd.Timestamp(end) - pd.Timedelta(days=2 * (profile.effective_lookback_bars + 40))
        )
        if profile.unbounded_lookback or profile.recursive_smoothing:
            if source.get("history_start"):
                start = min(start, pd.Timestamp(source["history_start"]))
            else:
                # A fixed origin avoids changing recursive initialization on every refresh.
                start = min(start, pd.Timestamp("2000-01-01"))
                approximate = True
    symbols = sorted(universe.members_overlapping(start, pd.Timestamp(end)))
    active = sorted(universe.members_asof(end))
    missing, stale, warmup, invalid = [], [], [], []
    cache = ParquetCache()
    for symbol in symbols:
        if not cache.has(symbol):
            missing.append(symbol)
            continue
        try:
            frame = cache.load(symbol)
            if frame.empty:
                missing.append(symbol)
            elif frame.index[-1] < pd.Timestamp(end) and symbol in active:
                stale.append(symbol)
            if not frame.empty and frame.index[0] > start:
                warmup.append(symbol)
                if recursive:
                    entry = min(m.entry for m in universe.memberships if m.symbol == symbol)
                    approximate |= frame.index[0] > max(start, entry) + pd.Timedelta(days=5)
        except (ValueError, OSError):
            invalid.append(symbol)
    # Dependency overrides are an explicit embedded panel, not a request to fetch market data.
    if default_panel is not None:
        symbols = active = list(default_panel.symbols)
        end = pd.Timestamp(default_panel.dates.max()).date().isoformat()
        missing = stale = warmup = invalid = []
    return {
        "universe": universe_name,
        "universe_fingerprint": universe.fingerprint,
        "required_fields": ["open", "high", "low", "close", "volume"],
        "calendar": "XNYS",
        "start": start.date().isoformat(),
        "expected_session": end,
        "aliases": dict(universe.aliases),
        "symbols": symbols,
        "active_symbols": active,
        "missing_symbols": missing,
        "stale_symbols": stale,
        "warmup_symbols": warmup,
        "invalid_symbols": invalid,
        "fetch_symbols": sorted(set(missing + stale + warmup)),
        "approximate": approximate,
        "coverage_floor": request.minimum_coverage,
        "offline": a._auto_sync_mode() == "off",
        "effective_lookback_bars": max(p.effective_lookback_bars for p in profiles),
    }


def _path(snapshot_id: str):
    return child_path(paths.data_dir() / "signals", snapshot_id, ".json", label="snapshot id")


def load(snapshot_id: str) -> dict[str, Any]:
    path = _path(snapshot_id)
    if not path.exists():
        raise HTTPException(
            404, "This signal snapshot is no longer available. Refresh the ranking."
        )
    return json.loads(path.read_text(encoding="utf-8"))


def _panel(plan: dict[str, Any], default_panel: Panel | None) -> Panel:
    if default_panel is not None:
        return default_panel
    a, cache = api(), ParquetCache()
    # Preserve missing members as NaN columns so coverage and exclusions use the real denominator.
    valid = []
    for symbol in plan["symbols"]:
        try:
            if cache.has(symbol):
                Panel.from_cache([symbol], cache=cache, end=plan["expected_session"])
                valid.append(symbol)
        except (ValueError, OSError):
            pass
    if not valid:
        raise ActionError(
            "no_prices",
            "No usable prices are available for this universe.",
            action="Refresh prices",
            symbols=plan["symbols"],
            retryable=True,
        )
    panel = Panel.from_cache(valid, cache=cache, end=plan["expected_session"])
    universe = a._resolve_universe(plan["universe"])
    fields = {
        name: frame.loc[plan["start"] :].reindex(columns=plan["symbols"])
        for name, frame in panel.fields.items()
    }
    membership = universe.membership_mask(fields["close"].index, plan["symbols"])
    fields = {name: frame.where(membership) for name, frame in fields.items()}
    fields["returns"] = fields["close"].pct_change(fill_method=None)
    # Keep the requested session even when every cached symbol is stale.
    dates = fields["close"].index.union(pd.DatetimeIndex([plan["expected_session"]])).sort_values()
    return Panel({name: frame.reindex(dates) for name, frame in fields.items()})


def run(
    request: Any,
    resolved: list[dict[str, Any]],
    plan: dict[str, Any],
    default_panel: Panel | None,
    progress: WorkspaceProgress,
    stop: Any,
) -> dict[str, Any]:
    a = api()
    sync: dict[str, Any] = {}
    refresh = request.data_mode == "refresh" or request.refresh_prices
    if refresh and not plan["offline"] and plan["fetch_symbols"] and default_panel is None:
        progress.phase = "syncing"
        sync = a._run_data_sync(
            a.DataSyncRequest(
                symbols=plan["fetch_symbols"],
                universe=plan["universe"],
                start=plan["start"],
                end=plan["expected_session"],
                aliases=plan["aliases"],
                mode="incremental",
            ),
            progress,
            stop,
        )
    if stop():
        raise TrainingCancelled("Signal preparation stopped.")
    progress.phase = "rechecking"
    if default_panel is None:
        plan = preparation(request, resolved)
    progress.phase = "evaluating"
    try:
        panel = _panel(plan, default_panel)
    except ActionError as exc:
        if sync.get("quota"):
            raise ActionError(
                "quota_limit",
                "The provider allowance was reached before usable prices were available.",
                action="Retry after the provider quota resets",
                symbols=plan["fetch_symbols"],
                retryable=True,
                details=str(sync["quota"]),
            ) from exc
        if sync.get("failed_count"):
            raise ActionError(
                "provider_failure",
                "Prices could not be prepared for this universe.",
                action="Retry refresh or inspect provider settings",
                symbols=plan["fetch_symbols"],
                retryable=True,
                details=str(sync.get("results", [])),
            ) from exc
        raise
    frames = []
    for source in resolved:
        if stop():
            raise TrainingCancelled("Signal evaluation stopped.")
        profile = window_profile(source["tree"])
        source_panel = panel
        if profile.unbounded_lookback or profile.recursive_smoothing:
            origin = source.get("history_start") or "2000-01-01"
            source_panel = Panel({name: frame.loc[origin:] for name, frame in panel.fields.items()})
        value = evaluate(source["tree"], source_panel)
        if not isinstance(value, pd.DataFrame):
            raise ActionError(
                "invalid_formula",
                "This formula does not produce a value per stock.",
                action="Edit formula",
            )
        frames.append(
            value.reindex(panel.dates).where(np.isfinite(value)).where(panel["close"].notna())
        )
    active = plan["active_symbols"]
    end = pd.Timestamp(plan["expected_session"])
    primary = frames[0].reindex(columns=active)
    primary_panel = Panel(
        {name: frame.reindex(columns=active) for name, frame in panel.fields.items()}
    )
    execution = request.execution or resolved[0].get("execution") or "next_open"
    result = snapshot(
        primary,
        primary_panel,
        min_names=resolved[0]["min_names"],
        execution=execution,
        approximate=plan["approximate"],
        snapshot_date=end,
    )
    floor = max(resolved[0]["min_names"], math.ceil(len(active) * request.minimum_coverage))
    actionable = result["ranked_count"] >= floor
    snapshot_id = uuid.uuid4().hex
    fingerprint = hashlib.sha256()
    for name, frame in panel.fields.items():
        fingerprint.update(name.encode())
        fingerprint.update(pd.util.hash_pandas_object(frame, index=True).values.tobytes())
    context = [public(item) for item in resolved]
    result.update(
        {
            "snapshot_id": snapshot_id,
            "source": request.source,
            "source_name": resolved[0]["name"],
            "sources": context,
            "evidence": resolved[0]["evidence"],
            "validation_passed": resolved[0].get("validation_passed"),
            "missing_context": [
                key
                for key in resolved[0]["missing_context"]
                if not (key == "execution" and request.execution)
                and not (key == "direction" and request.direction)
            ],
            "direction": request.direction
            or ("higher" if resolved[0].get("polarity") is not None else None),
            "universe": plan["universe"],
            "universe_fingerprint": plan["universe_fingerprint"],
            "data_revision": fingerprint.hexdigest(),
            "expected_session": plan["expected_session"],
            "active_count": len(active),
            "coverage": result["ranked_count"] / max(1, len(active)),
            "minimum_coverage": request.minimum_coverage,
            "actionable": actionable,
            "status": "ready"
            if actionable and not result["excluded_count"]
            else "partial"
            if actionable
            else "insufficient_coverage",
            "history_start": pd.Timestamp(panel.dates.min()).date().isoformat(),
            "history_bars": len(panel.dates),
            "effective_lookback_bars": plan["effective_lookback_bars"],
            "actual_signal_date": plan["expected_session"] if result["ranked_count"] else None,
            "fetched_at": a._now_iso() if sync else None,
            "computed_at": a._now_iso(),
            "price_basis": "split and dividend adjusted",
            "disclaimer": a.DISCLAIMER,
            "sync": {
                key: sync[key]
                for key in ("termination_reason", "failed_count", "succeeded_count", "quota")
                if key in sync
            },
            "offline": plan["offline"],
            "primary_strategy": resolved[0].get("primary_strategy", {}),
        }
    )
    attempts = {item["symbol"]: item for item in sync.get("results", [])}
    for item in result["excluded"]:
        attempt = attempts.get(item["symbol"], {})
        if item["symbol"] in plan["invalid_symbols"]:
            item.update(reason="invalid_prices", detail="Cached prices failed integrity checks.")
        elif attempt.get("status") == "failed":
            unavailable = attempt.get("error") == "provider returned no rows"
            item.update(
                reason="unavailable_history" if unavailable else "provider_failure",
                detail=attempt.get("error", "Provider request failed."),
            )
        elif item["reason"] == "warming_up":
            item.update(
                reason="insufficient_warmup",
                detail=(
                    "The available history does not initialize this formula; "
                    "provider history may start after membership entry."
                ),
            )
        elif sync.get("quota") and item["reason"] in {"no_prices", "stale_prices"}:
            item.update(reason="quota_limit", detail="The provider quota prevented this download.")
    if not actionable:
        result["diagnostic_rows"] = result["rows"]
        result["rows"] = []
    progress.phase = "saving"
    dates = [pd.Timestamp(day).date().isoformat() for day in panel.dates]

    def matrix(frame):
        return frame.reindex(columns=panel.symbols).to_numpy().tolist()

    denominators = (
        a._resolve_universe(plan["universe"])
        .membership_mask(panel.dates, plan["symbols"])
        .sum(axis=1)
        if default_panel is None
        else pd.Series(len(active), index=panel.dates)
    ).replace(0, np.nan)
    artifact = {
        "snapshot": result,
        "dates": dates,
        "symbols": list(panel.symbols),
        "prices": {
            name: matrix(panel[name]) for name in ("open", "high", "low", "close", "volume")
        },
        "values": [matrix(frame) for frame in frames],
        "percentiles": [matrix((frame.rank(axis=1, pct=True) * 100).round(2)) for frame in frames],
        "ranks": [matrix(frame.rank(axis=1, ascending=False, method="min")) for frame in frames],
        "coverage": [frame.notna().sum(axis=1).div(denominators).tolist() for frame in frames],
    }
    payload = a._json_finite(artifact)
    path = _path(snapshot_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(path, json.dumps(payload, separators=(",", ":")))
    progress.phase = "done"
    return payload["snapshot"]


class WorkspaceProgress(SyncProgress):
    phase = "checking"

    def snapshot(self):
        return {**super().snapshot(), "phase": self.phase}


def submit(request: Any, default_panel: Panel | None = None) -> dict[str, Any]:
    a = api()
    resolved = [resolve(request.source, request.bindings)] + [
        resolve(item.source, item.bindings) for item in request.comparisons
    ]
    plan = preparation(request, resolved, default_panel)
    identity = hashlib.sha256(
        json.dumps(
            {
                "request": request.model_dump(),
                "plan": plan,
                "sources": [public(source) for source in resolved],
            },
            sort_keys=True,
        ).encode()
    ).hexdigest()
    with SUBMIT_LOCK:
        for job in a._jobs.list():
            if job.metadata.get("signal_identity") == identity and job.status in {
                "queued",
                "running",
            }:
                return {"job_id": job.id, "status": job.status, "reused": True}
        progress = WorkspaceProgress(total=len(plan["fetch_symbols"]))
        cancel = threading.Event()
        job_id = a._jobs.submit(
            run,
            request,
            resolved,
            plan,
            default_panel,
            progress,
            cancel.is_set,
            progress=progress,
            cancel=cancel,
            metadata={"kind": "signal_snapshot", "signal_identity": identity},
        )
    return {"job_id": job_id, "status": "queued", "reused": False}


def series(snapshot_id: str, symbol: str) -> dict[str, Any]:
    artifact = load(snapshot_id)
    if symbol not in artifact["symbols"]:
        raise HTTPException(404, "This stock is not part of the snapshot universe.")
    column = artifact["symbols"].index(symbol)
    dates = artifact["dates"]
    bars = [
        {"time": date, **{name: rows[i][column] for name, rows in artifact["prices"].items()}}
        for i, date in enumerate(dates)
    ]
    sources = artifact["snapshot"]["sources"]
    return {
        "snapshot_id": snapshot_id,
        "symbol": symbol,
        "bars": bars,
        "data_revision": artifact["snapshot"]["data_revision"],
        "signals": [
            {
                "source": source["id"],
                "name": source["name"],
                "evidence": source["evidence"],
                "points": [
                    {
                        "time": date,
                        "value": artifact["values"][j][i][column],
                        "percentile": artifact["percentiles"][j][i][column],
                        "rank": artifact.get("ranks", artifact["percentiles"])[j][i][column],
                        "coverage": artifact["coverage"][j][i],
                    }
                    for i, date in enumerate(dates)
                ],
            }
            for j, source in enumerate(sources)
        ],
    }


def portfolio(
    snapshot_id: str, strategy: dict[str, Any] | None = None, notional: float | None = None
) -> dict[str, Any]:
    result = load(snapshot_id)["snapshot"]
    if not result["actionable"]:
        raise HTTPException(
            409, "Price coverage is too low for a portfolio preview. Refresh prices."
        )
    if result["missing_context"]:
        raise HTTPException(
            409, "Choose execution timing and ranking direction, then refresh the snapshot."
        )
    supplied = (
        strategy
        or result.get("primary_strategy")
        or {"id": "quantile-20", "scheme": "quantile_ls", "quantile": 0.2}
    )
    spec = PortfolioStrategySpec.from_dict(supplied)
    values = {
        item["symbol"]: item["value"] * (-1 if result.get("direction") == "lower" else 1)
        for item in result["rows"]
    }
    weights = spec.weighting_scheme().weights(pd.DataFrame([values])).iloc[0]
    rows = [
        {
            "symbol": symbol,
            "weight": float(weight),
            "side": "long" if weight > 0 else "short" if weight < 0 else "flat",
            "amount": None if notional is None else float(weight) * notional,
        }
        for symbol, weight in weights.items()
    ]
    return {
        "snapshot_id": snapshot_id,
        "strategy": spec.to_dict(),
        "rows": rows,
        "gross_exposure": float(weights.abs().sum()),
        "net_exposure": float(weights.sum()),
        "largest_weight": float(weights.abs().max()),
        "coverage": result["coverage"],
        "excluded_count": result["excluded_count"],
        "as_of": result["as_of"],
        "execution": result["execution"],
    }


def export(snapshot_id: str, kind: str, strategy: dict[str, Any] | None = None) -> str:
    result = load(snapshot_id)["snapshot"]
    if kind == "portfolio":
        rows = portfolio(snapshot_id, strategy)["rows"]
    elif kind == "excluded":
        rows = result["excluded"]
    else:
        if not result["actionable"]:
            raise HTTPException(409, "Coverage is too low to export an actionable ranking.")
        rows = result["rows"]
    metadata = {
        key: result.get(key)
        for key in (
            "snapshot_id",
            "source",
            "universe",
            "universe_fingerprint",
            "as_of",
            "execution",
            "direction",
            "coverage",
            "ranked_count",
            "active_count",
            "status",
            "data_revision",
            "price_basis",
            "evidence",
            "validation_passed",
        )
    }
    metadata["expression_fingerprint"] = result["sources"][0]["expression_fingerprint"]
    metadata["horizon"] = result["sources"][0].get("horizon")
    if kind == "portfolio":
        selected = (
            strategy or result.get("primary_strategy") or {"scheme": "quantile_ls", "quantile": 0.2}
        )
        metadata.update(strategy=selected.get("scheme"), quantile=selected.get("quantile"))
    output = io.StringIO(newline="")
    data = [{**row, **metadata} for row in rows]
    writer = csv.DictWriter(output, fieldnames=list(data[0]) if data else list(metadata))
    writer.writeheader()
    for row in data:
        # CSV files are often opened by spreadsheet applications.
        writer.writerow(
            {
                key: "'" + value if isinstance(value, str) and value[:1] in "=+-@" else value
                for key, value in row.items()
            }
        )
    return output.getvalue()
