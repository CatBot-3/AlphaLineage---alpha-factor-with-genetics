"""P5-T1 / P7 - the FastAPI app: submit GP searches and extend the system safely.

Lightweight by design: an in-process job runner and the cached Parquet panel - no Redis, no
Postgres. User operators are registered as typed macros (data, not code - invariant 5); custom
universes are point-in-time. The `get_panel` dependency is overridable so tests inject a
synthetic panel and never touch the network or the cache.

Not investment advice. Research output only.
"""

from __future__ import annotations

import json
import os
import re
import threading
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from alphalineage.api import sessions
from alphalineage.api.jobs import JobStore
from alphalineage.api.progress import RunProgress
from alphalineage.api.service import run_search
from alphalineage.core import cpp
from alphalineage.core.extensions import (
    USER_OPERATORS,
    InvalidOperator,
    ensure_operator,
    register_operator,
    unregister_operator,
)
from alphalineage.core.gp import GPConfig, validate_seed
from alphalineage.core.panel import Panel
from alphalineage.core.primitives import REGISTRY, Primitive
from alphalineage.core.tree import Node
from alphalineage.core.tree import from_dict as tree_from_dict
from alphalineage.core.tree import validate as validate_tree
from alphalineage.core.types import DType
from alphalineage.data import paths, schema, usage
from alphalineage.data.cache import ParquetCache
from alphalineage.data.provider import FallbackProvider, PriceProvider
from alphalineage.data.tiingo_client import TiingoProvider
from alphalineage.data.universe import Membership, Universe, sample_universe
from alphalineage.data.yfinance_provider import YFinanceProvider
from alphalineage.library.factors import FactorStore

app = FastAPI(title="AlphaLineage", version="0.1.0")
# Allow the browser `app` build (Vite dev server) to call the local backend.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)
_jobs = JobStore()
_universes: dict[str, Universe] = {}
_data_jobs = JobStore()

_DEFAULT_UNIVERSE = "sp500-lite"
_DEFAULT_AS_OF = "2026-06-01"
_WORKSPACE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,79}$")
_FORMULA_NAME_RE = re.compile(r"^[a-z][a-z0-9_]{1,63}$")


# --- models ----------------------------------------------------------------------
class OperatorSpec(BaseModel):
    name: str
    arg_types: list[str]
    out_type: str
    body: dict[str, Any]


class FormulaSpec(BaseModel):
    name: str
    display_name: str = ""
    description: str = ""
    arg_types: list[str]
    out_type: str
    body: dict[str, Any]


class MembershipSpec(BaseModel):
    symbol: str
    entry: str
    exit: str | None = None


class UniverseSpec(BaseModel):
    name: str
    memberships: list[MembershipSpec]


class RunRequest(BaseModel):
    universe: str = _DEFAULT_UNIVERSE
    config: dict[str, Any] = Field(default_factory=dict)
    operators: list[OperatorSpec] = Field(default_factory=list)


class JobResponse(BaseModel):
    job_id: str
    status: str


class WorkspaceSnapshot(BaseModel):
    id: str | None = None
    name: str = "Workspace"
    version: int = Field(ge=1)
    savedAt: str
    run: dict[str, Any] | None = None
    universes: list[UniverseSpec] = Field(default_factory=list)
    operators: list[OperatorSpec] = Field(default_factory=list)
    universeDraft: dict[str, Any] | None = None
    formulaDraft: dict[str, Any] | None = None
    operatorDraft: dict[str, Any] | None = None
    ui: dict[str, Any] = Field(default_factory=dict)


class WorkspaceSummary(BaseModel):
    id: str
    name: str
    savedAt: str
    hasRun: bool


class FactorSaveRequest(BaseModel):
    name: str
    tree: dict[str, Any]
    metrics: dict[str, Any] = Field(default_factory=dict)
    provenance: dict[str, Any] = Field(default_factory=dict)
    notes: str = ""


class FactorPatch(BaseModel):
    name: str | None = None
    notes: str | None = None


class SettingsUpdate(BaseModel):
    factors_dir: str | None = None
    tiingo_api_key: str | None = None
    evaluator: str | None = None


class DataClearRequest(BaseModel):
    category: str


class SymbolCandidate(BaseModel):
    symbol: str
    name: str = ""
    exchange: str = ""
    quote_type: str = ""
    currency: str = ""
    source: str = "yfinance"


class SymbolValidationRequest(BaseModel):
    symbol: str
    start: str | None = None
    end: str | None = None


class SymbolValidation(BaseModel):
    symbol: str
    valid: bool
    rows: int = 0
    first_date: str | None = None
    last_date: str | None = None
    provider: str | None = None
    error: str | None = None


class DataCoverage(BaseModel):
    symbol: str
    cached: bool
    rows: int
    first_date: str | None
    last_date: str | None
    requested_start: str | None
    requested_end: str | None
    needs_sync: bool


class DataSyncRequest(BaseModel):
    symbols: list[str]
    start: str
    end: str | None = None
    mode: str = "incremental"


class DataSyncResult(BaseModel):
    symbol: str
    status: str
    rows_fetched: int = 0
    rows_cached: int = 0
    first_date: str | None = None
    last_date: str | None = None
    provider: str | None = None
    error: str | None = None


_EVALUATORS = {"auto", "python", "cpp"}
_SYNC_MODES = {"incremental", "refresh"}


class SessionCreateRequest(BaseModel):
    name: str = "Session"
    universe: str = _DEFAULT_UNIVERSE
    as_of: str = _DEFAULT_AS_OF
    config: dict[str, Any] = Field(default_factory=dict)
    operators: list[OperatorSpec] = Field(default_factory=list)
    seed_factor_ids: list[str] = Field(default_factory=list)
    train: float = 0.6
    valid: float = 0.2
    embargo: int = 5


class SessionContinueRequest(BaseModel):
    generations: int = 5
    config: dict[str, Any] = Field(default_factory=dict)
    universe: str | None = None
    operators: list[OperatorSpec] = Field(default_factory=list)
    seed_factor_ids: list[str] = Field(default_factory=list)


def _primitive_info(prim: Primitive) -> dict[str, Any]:
    return {
        "name": prim.name,
        "kind": prim.kind.value,
        "arg_types": [t.value for t in prim.arg_types],
        "out_type": prim.out_type.value,
        "user": prim.macro_body is not None,
    }


def _register(spec: OperatorSpec) -> Primitive:
    try:
        return register_operator(
            spec.name, [DType(t) for t in spec.arg_types], DType(spec.out_type), spec.body
        )
    except (InvalidOperator, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def _normalize_formula_name(name: str) -> str:
    slug = re.sub(r"[^a-z0-9_]+", "_", name.strip().lower()).strip("_")
    slug = re.sub(r"_+", "_", slug)
    if not _FORMULA_NAME_RE.fullmatch(slug):
        raise HTTPException(
            status_code=400,
            detail=(
                "formula name must be lowercase snake case, start with a letter, "
                "and be 2-64 characters"
            ),
        )
    return slug


def _read_formula_specs() -> list[FormulaSpec]:
    path = paths.formulas_path()
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=500, detail="invalid formula store") from exc
    if not isinstance(payload, list):
        raise HTTPException(status_code=500, detail="invalid formula store")
    return [FormulaSpec(**item) for item in payload]


def _write_formula_specs(specs: list[FormulaSpec]) -> None:
    paths.meta_dir().mkdir(parents=True, exist_ok=True)
    payload = [_model_dump(spec) for spec in specs]
    paths.formulas_path().write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _tree_uses(node: Node, primitive_name: str) -> bool:
    return node.name == primitive_name or any(
        _tree_uses(child, primitive_name) for child in node.children
    )


def _formula_operator_spec(spec: FormulaSpec) -> OperatorSpec:
    return OperatorSpec(
        name=spec.name,
        arg_types=spec.arg_types,
        out_type=spec.out_type,
        body=spec.body,
    )


def _load_persisted_formulas() -> list[dict[str, Any]]:
    formulas: list[dict[str, Any]] = []
    for spec in _read_formula_specs():
        registered = False
        error = None
        try:
            ensure_operator(
                spec.name,
                [DType(t) for t in spec.arg_types],
                DType(spec.out_type),
                spec.body,
            )
            registered = True
        except (InvalidOperator, ValueError) as exc:
            error = str(exc)
        formulas.append({**_model_dump(spec), "registered": registered, "error": error})
    return formulas


def _price_provider() -> PriceProvider:
    providers: list[PriceProvider] = []
    if paths.tiingo_api_key():
        providers.append(TiingoProvider())
    providers.append(YFinanceProvider())
    return FallbackProvider(providers)


def _provider_source(provider: PriceProvider, symbol: str) -> str:
    if isinstance(provider, FallbackProvider):
        return provider.sources.get(symbol, provider.name)
    return provider.name


def _date_iso(value: Any) -> str:
    return pd.Timestamp(value).date().isoformat()


def _today_iso() -> str:
    return datetime.now(UTC).date().isoformat()


def _next_day_iso(value: Any) -> str:
    return (pd.Timestamp(value).date() + timedelta(days=1)).isoformat()


def _search_symbol_candidates(query: str, limit: int = 8) -> list[SymbolCandidate]:
    import yfinance as yf

    clean = query.strip()
    if not clean:
        return []
    search = yf.Search(
        clean,
        max_results=limit,
        news_count=0,
        lists_count=0,
        include_research=False,
        include_cultural_assets=False,
        recommended=0,
    )
    raw_quotes = getattr(search, "quotes", None) or []
    candidates: list[SymbolCandidate] = []
    seen: set[str] = set()
    for raw in raw_quotes:
        if not isinstance(raw, dict):
            continue
        symbol = str(raw.get("symbol") or raw.get("ticker") or "").strip().upper()
        if not symbol or symbol in seen:
            continue
        seen.add(symbol)
        candidates.append(
            SymbolCandidate(
                symbol=symbol,
                name=str(raw.get("shortname") or raw.get("longname") or raw.get("name") or ""),
                exchange=str(raw.get("exchDisp") or raw.get("exchange") or ""),
                quote_type=str(raw.get("quoteType") or raw.get("typeDisp") or ""),
                currency=str(raw.get("currency") or ""),
            )
        )
    return candidates


def _validate_symbol(
    symbol: str,
    start: str | None = None,
    end: str | None = None,
) -> SymbolValidation:
    clean = symbol.strip().upper()
    provider = _price_provider()
    try:
        frame = provider.get_prices(clean, start, end)
    except Exception as exc:  # noqa: BLE001 - validation should return readable failure payloads
        return SymbolValidation(symbol=clean, valid=False, provider=provider.name, error=str(exc))
    if frame.empty:
        return SymbolValidation(
            symbol=clean,
            valid=False,
            rows=0,
            provider=_provider_source(provider, clean),
            error="provider returned no rows",
        )
    return SymbolValidation(
        symbol=clean,
        valid=True,
        rows=len(frame),
        first_date=_date_iso(frame.index.min()),
        last_date=_date_iso(frame.index.max()),
        provider=_provider_source(provider, clean),
    )


def _coverage_for_symbol(
    symbol: str,
    start: str | None = None,
    end: str | None = None,
    cache: ParquetCache | None = None,
) -> DataCoverage:
    clean = symbol.strip().upper()
    requested_end = end or _today_iso()
    store = cache or ParquetCache()
    if not store.has(clean):
        return DataCoverage(
            symbol=clean,
            cached=False,
            rows=0,
            first_date=None,
            last_date=None,
            requested_start=start,
            requested_end=requested_end,
            needs_sync=True,
        )
    frame = store.load(clean)
    if frame.empty:
        first_date = last_date = None
        needs_sync = True
    else:
        first_date = _date_iso(frame.index.min())
        last_date = _date_iso(frame.index.max())
        needs_sync = bool(
            (start and pd.Timestamp(first_date) > pd.Timestamp(start)) or last_date < requested_end
        )
    return DataCoverage(
        symbol=clean,
        cached=True,
        rows=len(frame),
        first_date=first_date,
        last_date=last_date,
        requested_start=start,
        requested_end=requested_end,
        needs_sync=needs_sync,
    )


def _merge_price_frames(frames: list[pd.DataFrame]) -> pd.DataFrame:
    merged = pd.concat(frames).sort_index()
    merged = merged[~merged.index.duplicated(keep="last")]
    return schema.validate(merged[schema.PRICE_COLUMNS].astype("float64"))


def _sync_one_symbol(
    symbol: str,
    *,
    start: str,
    end: str | None,
    mode: str,
    provider: PriceProvider,
    cache: ParquetCache,
) -> DataSyncResult:
    clean = symbol.strip().upper()
    requested_end = end or _today_iso()
    try:
        existing = cache.load(clean) if cache.has(clean) else None
        fetch_ranges: list[tuple[str, str | None]] = []
        if mode == "refresh" or existing is None or existing.empty:
            fetch_ranges.append((start, end))
            existing = None if mode == "refresh" else existing
        else:
            first = _date_iso(existing.index.min())
            last = _date_iso(existing.index.max())
            if pd.Timestamp(first) > pd.Timestamp(start):
                fetch_ranges.append((start, first))
            if last < requested_end:
                fetch_ranges.append((_next_day_iso(last), end))

        if not fetch_ranges and existing is not None:
            return DataSyncResult(
                symbol=clean,
                status="skipped",
                rows_cached=len(existing),
                first_date=_date_iso(existing.index.min()),
                last_date=_date_iso(existing.index.max()),
                provider=_provider_source(provider, clean),
            )

        fetched: list[pd.DataFrame] = []
        for range_start, range_end in fetch_ranges:
            frame = provider.get_prices(clean, range_start, range_end)
            if not frame.empty:
                fetched.append(frame)
        if not fetched and existing is None:
            return DataSyncResult(
                symbol=clean,
                status="failed",
                provider=_provider_source(provider, clean),
                error="provider returned no rows",
            )

        frames = [*fetched] if existing is None else [existing, *fetched]
        merged = _merge_price_frames(frames)
        cache.store(clean, merged)
        stored = cache.load(clean)
        return DataSyncResult(
            symbol=clean,
            status="fetched" if fetched else "skipped",
            rows_fetched=sum(len(frame) for frame in fetched),
            rows_cached=len(stored),
            first_date=_date_iso(stored.index.min()),
            last_date=_date_iso(stored.index.max()),
            provider=_provider_source(provider, clean),
        )
    except Exception as exc:  # noqa: BLE001 - one failed symbol should not abort the batch
        return DataSyncResult(symbol=clean, status="failed", provider=provider.name, error=str(exc))


def _run_data_sync(req: DataSyncRequest) -> dict[str, Any]:
    provider = _price_provider()
    cache = ParquetCache()
    results = [
        _sync_one_symbol(
            symbol,
            start=req.start,
            end=req.end,
            mode=req.mode,
            provider=provider,
            cache=cache,
        )
        for symbol in req.symbols
        if symbol.strip()
    ]
    return {
        "mode": req.mode,
        "start": req.start,
        "end": req.end,
        "results": [_model_dump(result) for result in results],
    }


def _now_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def _model_dump(model: BaseModel) -> dict[str, Any]:
    dump = getattr(model, "model_dump", None)
    if dump is not None:
        return dump(mode="json")
    return model.dict()


def _new_workspace_id(name: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "-", name.strip().lower()).strip("-._")
    return f"{slug or 'workspace'}-{uuid.uuid4().hex[:8]}"


def _workspace_path(workspace_id: str) -> Path:
    if not _WORKSPACE_ID_RE.fullmatch(workspace_id):
        raise HTTPException(status_code=400, detail="invalid workspace id")
    return paths.workspaces_dir() / f"{workspace_id}.json"


def _workspace_snapshot_from_file(path: Path) -> WorkspaceSnapshot:
    try:
        return WorkspaceSnapshot(**json.loads(path.read_text(encoding="utf-8")))
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
        raise HTTPException(status_code=500, detail=f"invalid workspace file: {path.name}") from exc


def _save_workspace_snapshot(snapshot: WorkspaceSnapshot) -> WorkspaceSnapshot:
    workspace_id = snapshot.id or _new_workspace_id(snapshot.name)
    if not _WORKSPACE_ID_RE.fullmatch(workspace_id):
        raise HTTPException(status_code=400, detail="invalid workspace id")
    payload = _model_dump(snapshot)
    payload["id"] = workspace_id
    target = _workspace_path(workspace_id)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return WorkspaceSnapshot(**payload)


def _universe_from_spec(spec: UniverseSpec) -> Universe:
    members = [
        Membership(
            m.symbol,
            pd.Timestamp(m.entry),
            pd.Timestamp(m.exit) if m.exit else None,
        )
        for m in spec.memberships
    ]
    return Universe(spec.name, members)


def _universe_to_spec(universe: Universe) -> dict[str, Any]:
    return {
        "name": universe.name,
        "memberships": [
            {
                "symbol": m.symbol,
                "entry": m.entry.date().isoformat(),
                "exit": m.exit.date().isoformat() if m.exit is not None else None,
            }
            for m in universe.memberships
        ],
    }


def _persist_universe(spec: UniverseSpec) -> Universe:
    universe = _universe_from_spec(spec)
    _universes[spec.name] = universe
    universe.save()
    return universe


def _load_persisted_universes() -> None:
    universe_dir = paths.universe_dir()
    if not universe_dir.exists():
        return
    for source in universe_dir.glob("*.parquet"):
        if source.stem in _universes:
            continue
        try:
            _universes[source.stem] = Universe.load(source.stem, source)
        except Exception:
            continue


def _save_run_workspace(job_id: str, result: Any) -> None:
    if not isinstance(result, dict):
        return
    _save_workspace_snapshot(
        WorkspaceSnapshot(
            id=f"run-{job_id}",
            name=f"Run {job_id[:8]}",
            version=1,
            savedAt=_now_iso(),
            run=result,
            ui={"selectedTab": "dashboard", "source": "backend-run"},
        )
    )


def get_panel() -> Panel:
    """Build the working panel from the cached universe (overridden in tests)."""
    symbols = sample_universe(_DEFAULT_UNIVERSE).members_asof(_DEFAULT_AS_OF)
    return Panel.from_cache(symbols)


def _factor_store() -> FactorStore:
    """A factor store rooted at the currently configured directory (resolved fresh)."""
    return FactorStore(paths.factors_dir())


def _panel_for_universe(universe: str, as_of: str, default_panel: Panel) -> Panel:
    """The working panel for ``universe``: a custom point-in-time one, else the default."""
    _load_persisted_universes()
    if universe in _universes:
        symbols = _universes[universe].members_asof(as_of)
        return Panel.from_cache(symbols)
    return default_panel


def _load_seed_factors(
    factor_ids: list[str],
) -> tuple[list[Any], int, int]:
    """Resolve saved factors into seed trees, re-registering their operators (P4).

    Returns ``(seed_trees, trial_baseline, test_reads_baseline)`` where the baselines sum the
    provenance of distinct source sessions, so a seeded session inherits their honesty counts.
    """
    store = _factor_store()
    seeds: list[Any] = []
    trial_baseline = 0
    test_reads_baseline = 0
    seen: set[str] = set()
    for factor_id in factor_ids:
        factor = store.get(factor_id)
        if factor is None:
            raise HTTPException(status_code=400, detail=f"unknown factor {factor_id!r}")
        for spec in factor.required_operators:  # data, not code (invariant 5)
            try:
                ensure_operator(
                    spec["name"],
                    [DType(t) for t in spec["arg_types"]],
                    DType(spec["out_type"]),
                    spec["body"],
                )
            except (InvalidOperator, ValueError) as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
        seeds.append(factor.tree)
        provenance = factor.provenance or {}
        key = str(provenance.get("session_id") or factor_id)
        if key not in seen:
            seen.add(key)
            trial_baseline += int(provenance.get("cumulative_trials", 0) or 0)
            test_reads_baseline += int(provenance.get("test_reads", 0) or 0)
    return seeds, trial_baseline, test_reads_baseline


# --- endpoints -------------------------------------------------------------------
@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/primitives")
def list_primitives() -> list[dict[str, Any]]:
    """The primitive palette (built-ins + user operators) for the operator composer."""
    _load_persisted_formulas()
    return [_primitive_info(p) for p in REGISTRY.values()]


@app.post("/operators")
def add_operator(spec: OperatorSpec) -> dict[str, Any]:
    return _primitive_info(_register(spec))


@app.get("/operators")
def list_operators() -> list[dict[str, Any]]:
    return [_primitive_info(p) for p in USER_OPERATORS.values()]


@app.delete("/operators/{name}")
def remove_operator(name: str) -> dict[str, str]:
    unregister_operator(name)
    return {"removed": name}


@app.get("/formulas")
def list_formulas() -> list[dict[str, Any]]:
    return _load_persisted_formulas()


@app.post("/formulas")
def add_formula(spec: FormulaSpec) -> dict[str, Any]:
    normalized = _normalize_formula_name(spec.name)
    spec = FormulaSpec(
        **{
            **_model_dump(spec),
            "name": normalized,
            "display_name": spec.display_name or normalized.replace("_", " "),
        }
    )
    saved = _read_formula_specs()
    if any(item.name == spec.name for item in saved):
        raise HTTPException(status_code=400, detail="formula name already exists")
    _register(_formula_operator_spec(spec))
    saved.append(spec)
    _write_formula_specs(saved)
    return {**_model_dump(spec), "registered": True, "error": None}


@app.delete("/formulas/{name}")
def delete_formula(name: str) -> dict[str, str]:
    normalized = _normalize_formula_name(name)
    saved = _read_formula_specs()
    target = next((spec for spec in saved if spec.name == normalized), None)
    if target is None:
        raise HTTPException(status_code=404, detail="unknown formula")
    for spec in saved:
        if spec.name == normalized:
            continue
        if _tree_uses(tree_from_dict(spec.body), normalized):
            raise HTTPException(status_code=400, detail=f"formula is used by {spec.name!r}")
    unregister_operator(normalized)
    _write_formula_specs([spec for spec in saved if spec.name != normalized])
    return {"removed": normalized}


@app.get("/universes")
def list_universes() -> list[dict[str, Any]]:
    _load_persisted_universes()
    sample = sample_universe(_DEFAULT_UNIVERSE)
    custom = [
        {**_universe_to_spec(universe), "symbols": universe.all_symbols(), "source": "custom"}
        for universe in sorted(_universes.values(), key=lambda u: u.name)
    ]
    return [
        {**_universe_to_spec(sample), "symbols": sample.all_symbols(), "source": "sample"},
        *custom,
    ]


@app.post("/universes")
def add_universe(spec: UniverseSpec) -> dict[str, Any]:
    if spec.name == _DEFAULT_UNIVERSE:
        raise HTTPException(status_code=400, detail="sample universes cannot be replaced")
    universe = _persist_universe(spec)
    return {"name": spec.name, "symbols": universe.all_symbols()}


@app.get("/universes/{name}")
def get_universe(name: str) -> dict[str, Any]:
    _load_persisted_universes()
    if name == _DEFAULT_UNIVERSE:
        universe = sample_universe(_DEFAULT_UNIVERSE)
        return {
            **_universe_to_spec(universe),
            "symbols": universe.all_symbols(),
            "source": "sample",
        }
    universe = _universes.get(name)
    if universe is None:
        raise HTTPException(status_code=404, detail="unknown universe")
    return {**_universe_to_spec(universe), "symbols": universe.all_symbols(), "source": "custom"}


@app.put("/universes/{name}")
def update_universe(name: str, spec: UniverseSpec) -> dict[str, Any]:
    if name == _DEFAULT_UNIVERSE:
        raise HTTPException(status_code=400, detail="sample universes cannot be updated")
    if spec.name != name:
        raise HTTPException(status_code=400, detail="universe name must match the path")
    universe = _persist_universe(spec)
    return {"name": spec.name, "symbols": universe.all_symbols()}


@app.delete("/universes/{name}")
def delete_universe(name: str) -> dict[str, str]:
    if name == _DEFAULT_UNIVERSE:
        raise HTTPException(status_code=400, detail="sample universes cannot be deleted")
    _load_persisted_universes()
    target = paths.universe_dir() / f"{name}.parquet"
    if name not in _universes and not target.exists():
        raise HTTPException(status_code=404, detail="unknown universe")
    _universes.pop(name, None)
    if target.exists():
        target.unlink()
    return {"removed": name}


@app.get("/workspaces", response_model=list[WorkspaceSummary])
def list_workspaces() -> list[WorkspaceSummary]:
    workspace_dir = paths.workspaces_dir()
    if not workspace_dir.exists():
        return []
    workspace_paths = list(workspace_dir.glob("*.json"))
    summaries = [
        WorkspaceSummary(
            id=snapshot.id or path.stem,
            name=snapshot.name,
            savedAt=snapshot.savedAt,
            hasRun=snapshot.run is not None,
        )
        for path, snapshot in (
            (path, _workspace_snapshot_from_file(path)) for path in workspace_paths
        )
    ]
    return sorted(summaries, key=lambda item: item.savedAt, reverse=True)


@app.post("/workspaces", response_model=WorkspaceSnapshot)
def save_workspace(snapshot: WorkspaceSnapshot) -> WorkspaceSnapshot:
    for spec in snapshot.universes:
        _persist_universe(spec)
    return _save_workspace_snapshot(snapshot)


@app.get("/workspaces/{workspace_id}", response_model=WorkspaceSnapshot)
def get_workspace(workspace_id: str) -> WorkspaceSnapshot:
    target = _workspace_path(workspace_id)
    if not target.exists():
        raise HTTPException(status_code=404, detail="unknown workspace")
    return _workspace_snapshot_from_file(target)


@app.delete("/workspaces/{workspace_id}")
def delete_workspace(workspace_id: str) -> dict[str, str]:
    target = _workspace_path(workspace_id)
    if not target.exists():
        raise HTTPException(status_code=404, detail="unknown workspace")
    target.unlink()
    return {"removed": workspace_id}


@app.post("/runs", response_model=JobResponse)
def submit_run(req: RunRequest, panel: Panel = Depends(get_panel)) -> JobResponse:  # noqa: B008
    _load_persisted_formulas()
    for spec in req.operators:  # register user operators before the search sees them
        _register(spec)
    panel = _panel_for_universe(req.universe, _DEFAULT_AS_OF, panel)
    config = GPConfig.from_dict(req.config)
    progress = RunProgress(target_generations=config.generations)
    cancel = threading.Event()

    def _task() -> dict[str, Any]:
        return run_search(config, panel, progress=progress, stop=cancel.is_set)

    job_id = _jobs.submit(_task, progress=progress, cancel=cancel, on_success=_save_run_workspace)
    return JobResponse(job_id=job_id, status="queued")


@app.get("/runs/{job_id}")
def get_run(job_id: str) -> dict[str, Any]:
    job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="unknown job")
    return {
        "job_id": job.id,
        "status": job.status,
        "result": job.result,
        "error": job.error,
        "progress": job.progress.snapshot() if job.progress is not None else None,
    }


@app.post("/runs/{job_id}/stop")
def stop_run(job_id: str) -> dict[str, bool]:
    """Ask a running search to halt after its current generation; it completes normally."""
    if not _jobs.cancel(job_id):
        raise HTTPException(status_code=404, detail="unknown job")
    return {"stopping": True}


@app.get("/runs/{job_id}/lineage")
def get_lineage(job_id: str) -> dict[str, Any]:
    job = _jobs.get(job_id)
    if job is None or not isinstance(job.result, dict):
        raise HTTPException(status_code=404, detail="no lineage yet")
    lineage: dict[str, Any] = job.result.get("lineage", {})
    return lineage


# --- factor library --------------------------------------------------------------
@app.get("/factors")
def list_factors() -> list[dict[str, Any]]:
    return [factor.to_dict() for factor in _factor_store().list()]


@app.post("/factors")
def save_factor(req: FactorSaveRequest) -> dict[str, Any]:
    try:
        tree = validate_tree(tree_from_dict(req.tree))  # only real, registered primitives
    except Exception as exc:  # noqa: BLE001 - any malformed/unknown tree is a client error
        raise HTTPException(status_code=400, detail=f"invalid factor tree: {exc}") from exc
    factor = _factor_store().save(
        name=req.name,
        tree=tree,
        metrics=req.metrics,
        provenance=req.provenance,
        notes=req.notes,
        saved_at=_now_iso(),
    )
    return factor.to_dict()


@app.get("/factors/{factor_id}")
def get_factor(factor_id: str) -> dict[str, Any]:
    factor = _factor_store().get(factor_id)
    if factor is None:
        raise HTTPException(status_code=404, detail="unknown factor")
    return factor.to_dict()


@app.patch("/factors/{factor_id}")
def patch_factor(factor_id: str, patch: FactorPatch) -> dict[str, Any]:
    factor = _factor_store().update(factor_id, name=patch.name, notes=patch.notes)
    if factor is None:
        raise HTTPException(status_code=404, detail="unknown factor")
    return factor.to_dict()


@app.delete("/factors/{factor_id}")
def delete_factor(factor_id: str) -> dict[str, str]:
    if not _factor_store().delete(factor_id):
        raise HTTPException(status_code=404, detail="unknown factor")
    return {"removed": factor_id}


# --- settings --------------------------------------------------------------------
@app.get("/settings")
def get_settings() -> dict[str, Any]:
    stored = paths.read_settings()
    return {
        "factors_dir": str(paths.factors_dir()),
        "tiingo_api_key_set": bool(paths.tiingo_api_key()),  # never echo the secret itself
        "evaluator": stored.get("evaluator", "auto"),
        "cpp_available": cpp.available(),
    }


@app.put("/settings")
def update_settings(update: SettingsUpdate) -> dict[str, Any]:
    settings = paths.read_settings()
    if update.factors_dir is not None:
        target = Path(update.factors_dir).expanduser()
        try:
            target.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise HTTPException(status_code=400, detail=f"cannot use factors_dir: {exc}") from exc
        settings["factors_dir"] = str(target)
    if update.tiingo_api_key is not None:
        key = update.tiingo_api_key.strip()
        if key:
            settings["tiingo_api_key"] = key
        else:
            settings.pop("tiingo_api_key", None)  # empty string clears the stored key
    if update.evaluator is not None:
        evaluator = update.evaluator.lower()
        if evaluator not in _EVALUATORS:
            raise HTTPException(
                status_code=400, detail=f"evaluator must be one of {sorted(_EVALUATORS)}"
            )
        settings["evaluator"] = evaluator
        cpp.set_backend(evaluator)
    paths.write_settings(settings)
    return get_settings()


# --- local data usage + cleanup --------------------------------------------------
@app.get("/symbols/search", response_model=list[SymbolCandidate])
def search_symbols(query: str, limit: int = 8) -> list[SymbolCandidate]:
    try:
        return _search_symbol_candidates(query, limit=limit)
    except Exception as exc:  # noqa: BLE001 - surface provider/search failures clearly
        raise HTTPException(status_code=502, detail=f"symbol search failed: {exc}") from exc


@app.post("/symbols/validate", response_model=SymbolValidation)
def validate_symbol(req: SymbolValidationRequest) -> SymbolValidation:
    return _validate_symbol(req.symbol, req.start, req.end)


@app.get("/data/usage")
def data_usage() -> list[dict[str, Any]]:
    return usage.usage()


@app.get("/data/coverage", response_model=list[DataCoverage])
def data_coverage(
    symbols: str,
    start: str | None = None,
    end: str | None = None,
) -> list[DataCoverage]:
    parsed = [symbol.strip().upper() for symbol in symbols.split(",") if symbol.strip()]
    return [_coverage_for_symbol(symbol, start, end) for symbol in parsed]


@app.post("/data/sync")
def data_sync(req: DataSyncRequest) -> dict[str, str]:
    mode = req.mode.lower()
    if mode not in _SYNC_MODES:
        raise HTTPException(status_code=400, detail=f"mode must be one of {sorted(_SYNC_MODES)}")
    symbols = sorted({symbol.strip().upper() for symbol in req.symbols if symbol.strip()})
    if not symbols:
        raise HTTPException(status_code=400, detail="at least one symbol is required")
    request = DataSyncRequest(symbols=symbols, start=req.start, end=req.end, mode=mode)
    job_id = _data_jobs.submit(lambda: _run_data_sync(request))
    return {"job_id": job_id, "status": "queued"}


@app.get("/data/sync/{job_id}")
def data_sync_status(job_id: str) -> dict[str, Any]:
    job = _data_jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="unknown sync job")
    return {
        "job_id": job.id,
        "status": job.status,
        "result": job.result,
        "error": job.error,
    }


@app.post("/data/clear")
def data_clear(req: DataClearRequest) -> dict[str, Any]:
    try:
        return usage.clear(req.category)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


# --- iterative training sessions -------------------------------------------------
def _validate_seeds(seeds: list[Any], config: GPConfig) -> None:
    for seed in seeds:
        try:
            validate_seed(seed, max_depth=config.max_depth, max_nodes=config.max_nodes)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc


def _session_job_view(session: dict[str, Any]) -> dict[str, Any] | None:
    job = _jobs.get(session.get("last_job_id") or "")
    if job is None:
        return None
    return {
        "id": job.id,
        "status": job.status,
        "progress": job.progress.snapshot() if job.progress is not None else None,
    }


@app.post("/sessions")
def create_session(
    req: SessionCreateRequest,
    panel: Panel = Depends(get_panel),  # noqa: B008
) -> dict[str, str]:
    _load_persisted_formulas()
    for spec in req.operators:
        _register(spec)
    panel = _panel_for_universe(req.universe, req.as_of, panel)
    seeds, trial_baseline, test_reads_baseline = _load_seed_factors(req.seed_factor_ids)
    config = GPConfig.from_dict(req.config)
    _validate_seeds(seeds, config)
    try:
        boundaries = sessions.derive_boundaries(
            panel.dates, train=req.train, valid=req.valid, embargo=req.embargo
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    session = sessions.new_session(
        name=req.name,
        universe=req.universe,
        as_of=req.as_of,
        config=config,
        operators=[_model_dump(spec) for spec in req.operators],
        seed_factor_ids=req.seed_factor_ids,
        boundaries=boundaries,
        trial_baseline=trial_baseline,
        test_reads_baseline=test_reads_baseline,
        created_at=_now_iso(),
    )
    session_id = session["id"]
    progress = RunProgress(target_generations=config.generations)
    cancel = threading.Event()
    job_id = uuid.uuid4().hex

    def _task() -> dict[str, Any]:
        return sessions.run_segment(
            session_id,
            job_id=job_id,
            panel=panel,
            config=config,
            generations=config.generations,
            seeds=seeds,
            progress=progress,
            stop=cancel.is_set,
        )

    _jobs.submit(_task, job_id=job_id, progress=progress, cancel=cancel)
    return {"session_id": session_id, "job_id": job_id}


@app.post("/sessions/{session_id}/continue")
def continue_session(
    session_id: str,
    req: SessionContinueRequest,
    panel: Panel = Depends(get_panel),  # noqa: B008
) -> dict[str, str]:
    _load_persisted_formulas()
    if not sessions.exists(session_id):
        raise HTTPException(status_code=404, detail="unknown session")
    session = sessions.load_session(session_id)

    job = _jobs.get(session.get("last_job_id") or "")
    if job is not None and job.status in ("queued", "running"):
        raise HTTPException(status_code=409, detail="a segment is already running")

    for spec in req.operators:  # newly added operators for this segment
        try:
            ensure_operator(
                spec.name, [DType(t) for t in spec.arg_types], DType(spec.out_type), spec.body
            )
        except (InvalidOperator, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
    # re-register the session's own operators (process may have restarted / been cleared)
    for stored in session.get("operators", []):
        try:
            ensure_operator(
                stored["name"],
                [DType(t) for t in stored["arg_types"]],
                DType(stored["out_type"]),
                stored["body"],
            )
        except (InvalidOperator, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    universe = req.universe or session["universe"]
    new_panel = _panel_for_universe(universe, session["as_of"], panel)
    universe_changed = universe != session["universe"]

    stored_config = GPConfig.from_dict(session["config"])
    merged = {**session["config"], **req.config}
    config = GPConfig.from_dict(merged)
    scoring_changed = any(
        getattr(stored_config, field) != getattr(config, field)
        for field in ("parsimony", "ic_method", "min_names", "horizon")
    )
    rescore = universe_changed or scoring_changed

    extra_seeds, _, _ = _load_seed_factors(req.seed_factor_ids)
    _validate_seeds(extra_seeds, config)

    if universe_changed:
        session["universe"] = universe
        session["config"] = merged
        sessions.save_session(session)
    elif merged != session["config"]:
        session["config"] = merged
        sessions.save_session(session)

    progress = RunProgress(target_generations=config.generations)
    cancel = threading.Event()
    job_id = uuid.uuid4().hex

    def _task() -> dict[str, Any]:
        return sessions.run_segment(
            session_id,
            job_id=job_id,
            panel=new_panel,
            config=config,
            generations=req.generations,
            extra_seeds=extra_seeds,
            rescore=rescore,
            progress=progress,
            stop=cancel.is_set,
        )

    _jobs.submit(_task, job_id=job_id, progress=progress, cancel=cancel)
    return {"session_id": session_id, "job_id": job_id}


@app.get("/sessions")
def list_sessions() -> list[dict[str, Any]]:
    root = paths.sessions_dir()
    if not root.exists():
        return []
    summaries = []
    for directory in root.iterdir():
        if not (directory / "session.json").exists():
            continue
        session = sessions.load_session(directory.name)
        summaries.append(
            {
                "id": session["id"],
                "name": session["name"],
                "created_at": session.get("created_at", ""),
                "universe": session["universe"],
                "segments": len(session["segments"]),
                "cumulative_trials": session["cumulative_trials"],
                "test_reads": session["test_reads"],
            }
        )
    return sorted(summaries, key=lambda item: item["created_at"], reverse=True)


@app.get("/sessions/{session_id}")
def get_session(session_id: str) -> dict[str, Any]:
    if not sessions.exists(session_id):
        raise HTTPException(status_code=404, detail="unknown session")
    session = sessions.load_session(session_id)
    result_path = sessions.session_dir(session_id) / "result.json"
    result = None
    if result_path.exists():
        result = json.loads(result_path.read_text(encoding="utf-8"))
        result = {k: v for k, v in result.items() if k != "lineage"}  # lineage via its endpoint
    return {**session, "job": _session_job_view(session), "result": result}


@app.get("/sessions/{session_id}/lineage")
def get_session_lineage(session_id: str) -> dict[str, Any]:
    lineage_path = sessions.session_dir(session_id) / "lineage.json"
    if not lineage_path.exists():
        raise HTTPException(status_code=404, detail="no lineage yet")
    data: dict[str, Any] = json.loads(lineage_path.read_text(encoding="utf-8"))
    return data


@app.post("/sessions/{session_id}/stop")
def stop_session(session_id: str) -> dict[str, bool]:
    if not sessions.exists(session_id):
        raise HTTPException(status_code=404, detail="unknown session")
    session = sessions.load_session(session_id)
    job_id = session.get("last_job_id")
    if job_id and _jobs.cancel(job_id):
        return {"stopping": True}
    return {"stopping": False}


# --- static frontend (single-image Docker serving) -------------------------------
def _schedule_shutdown() -> None:
    """Exit the process shortly, after the HTTP response has flushed. Patched out in tests."""
    threading.Timer(0.4, lambda: os._exit(0)).start()


@app.post("/shutdown")
def shutdown() -> dict[str, bool]:
    """Stop the app (single-process launcher Quit). Gated so it is inert unless enabled.

    Session state is already persisted to disk after every segment, so an immediate exit is safe;
    the frontend warns about a running search before calling this.
    """
    if os.environ.get("ALPHALINEAGE_ALLOW_SHUTDOWN") != "1":
        raise HTTPException(status_code=403, detail="shutdown is disabled")
    _schedule_shutdown()
    return {"shutting_down": True}


def mount_static(directory: str) -> None:
    """Serve the built frontend from ``directory`` at ``/``.

    Registered after every API route so the API still wins for its paths and everything else
    (the SPA's assets, ``index.html``) is served statically - which lets one container host both
    the backend and the UI on the same origin (no CORS, no second server).
    """
    app.mount("/", StaticFiles(directory=directory, html=True), name="static")


_STATIC_DIR = os.environ.get("ALPHALINEAGE_STATIC_DIR")
if _STATIC_DIR and Path(_STATIC_DIR).is_dir():
    mount_static(_STATIC_DIR)


def _apply_persisted_settings() -> None:
    """Apply runtime-relevant settings (the evaluator choice) once at startup."""
    try:
        _load_persisted_formulas()
        evaluator = paths.read_settings().get("evaluator")
        if evaluator:
            cpp.set_backend(evaluator)
    except Exception:  # noqa: BLE001 - settings are best-effort at startup
        pass


_apply_persisted_settings()
