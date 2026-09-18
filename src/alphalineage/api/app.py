"""P5-T1 / P7 - the FastAPI app: submit GP searches and extend the system safely.

Lightweight by design: an in-process job runner and the cached Parquet panel - no Redis, no
Postgres. User operators are registered as typed macros (data, not code - invariant 5); custom
universes are point-in-time. The `get_panel` dependency is overridable so tests inject a
synthetic panel and never touch the network or the cache.

Not investment advice. Research output only.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import re
import shutil
import threading
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
from dotenv import find_dotenv, load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from alphalineage.agent import service as agent_service
from alphalineage.agent import tools as agent_tools
from alphalineage.agent.conversation import ConversationStore
from alphalineage.agent.guards import (
    DEFAULT_MAX_EVALUATIONS,
    DEFAULT_MAX_SECONDS,
    DEFAULT_MAX_TOOL_CALLS,
    MAX_EVALUATIONS_CEILING,
    MAX_SECONDS_CEILING,
    MAX_TOOL_CALLS_CEILING,
    PROTECTED_KEYS,
    TUNABLE_KEYS,
    GuardViolation,
)
from alphalineage.agent.guards import Budget as AgentBudget
from alphalineage.api import sessions
from alphalineage.api.jobs import JobStore
from alphalineage.api.progress import RunProgress, SyncProgress
from alphalineage.api.resources import (
    TRAINING_SCHEDULER,
    ResolvedResources,
    ResourcePolicy,
    TrainingLeaseCancelled,
    resolve_resources,
    training_capabilities,
)
from alphalineage.api.service import run_search
from alphalineage.backtest.costs import TransactionCostModel
from alphalineage.backtest.portfolio import (
    PORTFOLIO_SCHEMA_VERSION,
    PortfolioStrategySpec,
)
from alphalineage.backtest.reporting import backtest_report
from alphalineage.core import categories as core_categories
from alphalineage.core import cpp
from alphalineage.core.evaluate import evaluate
from alphalineage.core.extensions import (
    USER_OPERATORS,
    InvalidOperator,
    ensure_operator,
    expand_all,
    infer_macro_type,
    register_operator,
    unregister_operator,
)
from alphalineage.core.fitness import forward_returns, label_span
from alphalineage.core.gp import (
    MAX_GENERATIONS,
    MAX_HORIZON,
    MAX_SEARCH_EVALUATIONS,
    GPConfig,
    TrainingCancelled,
    validate_seed,
)
from alphalineage.core.panel import Panel
from alphalineage.core.primitive_docs import primitive_doc
from alphalineage.core.primitives import OPERATORS, REGISTRY, Primitive
from alphalineage.core.tree import Node
from alphalineage.core.tree import from_dict as tree_from_dict
from alphalineage.core.tree import from_json as tree_from_json
from alphalineage.core.tree import to_dict as tree_to_dict
from alphalineage.core.tree import to_json as tree_to_json
from alphalineage.core.tree import validate as validate_tree
from alphalineage.core.types import DType, is_subtype

# ``schema`` remains re-exported from this module for older integrations/tests.
from alphalineage.data import paths, schema, universe_folders, usage  # noqa: F401
from alphalineage.data.adjust import split_adjusted_close
from alphalineage.data.cache import ParquetCache, merge_price_frames
from alphalineage.data.identifiers import (
    atomic_write_text,
    child_path,
)
from alphalineage.data.identifiers import (
    validate_symbol as validate_market_symbol,
)
from alphalineage.data.provider import FallbackProvider, PriceProvider, QuotaExceededError
from alphalineage.data.tiingo_client import TiingoProvider
from alphalineage.data.universe import (
    Membership,
    Universe,
    bundled_snapshot_name,
    bundled_snapshot_specs,
    bundled_universe,
    normalize_market_symbol,
    sample_universe,
    universe_integrity,
    universe_presets,
)
from alphalineage.data.yfinance_provider import YFinanceProvider
from alphalineage.explain import credentials as explain_credentials
from alphalineage.explain.providers import LLMError
from alphalineage.library.factors import DISCLAIMER, FactorStore
from alphalineage.library.indicator_catalog import (
    CATALOG_NAMES,
    CATALOG_ORIGIN,
    CATALOG_REVISION,
    INDICATOR_CATALOG,
    LEGACY_CATALOG_REPLACEMENTS,
)
from alphalineage.library.overlap import (
    OVERLAP_VERSION,
    ReferenceFactor,
    overlap_report,
    references_from_formulas,
)
from alphalineage.validation.splits import time_split

# Local launches historically read ``.env`` in some helper scripts but not in the
# application process itself.  Load it before settings/providers are resolved while
# preserving the standard precedence of an explicitly supplied process environment.
_DOTENV_PATH = find_dotenv(usecwd=True)
if _DOTENV_PATH and os.environ.get("ALPHALINEAGE_SKIP_DOTENV") != "1":
    load_dotenv(_DOTENV_PATH, override=False)

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
_data_sync_submit_lock = threading.Lock()
_universe_lifecycle_lock = threading.RLock()
_formula_test_jobs = JobStore()
# Formula registration is process-global.  Serialize deletion with one-shot run submission so a
# run is visible as a revision holder before its pinned primitives can be unregistered.
_formula_lifecycle_lock = threading.RLock()

_DEFAULT_UNIVERSE = "sp500-lite"
_DEFAULT_AS_OF = datetime.now(UTC).date().isoformat()
_WORKSPACE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,79}$")
_FORMULA_NAME_RE = re.compile(r"^[a-z][a-z0-9_]{1,63}$")
_FORMULA_INPUT_RE = re.compile(r"^[a-z][a-z0-9_]{0,31}$")
_BENCHMARK_CATALOG: tuple[dict[str, str], ...] = (
    {
        "id": "sp500",
        "label": "S&P 500",
        "symbol": "^GSPC",
        "color": "#8f2d22",
        "methodology": (
            "S&P 500 split-adjusted close-to-close price return "
            "(cash dividends are not reinvested)."
        ),
    },
    {
        "id": "djia",
        "label": "Dow Jones Industrial Average",
        "symbol": "^DJI",
        "color": "#8a6514",
        "methodology": (
            "Dow Jones Industrial Average split-adjusted close-to-close price return "
            "(cash dividends are not reinvested)."
        ),
    },
    {
        "id": "nasdaq100",
        "label": "Nasdaq-100",
        "symbol": "^NDX",
        "color": "#6d3fa0",
        "methodology": (
            "Nasdaq-100 split-adjusted close-to-close price return "
            "(cash dividends are not reinvested)."
        ),
    },
)


# --- models ----------------------------------------------------------------------
class OperatorSpec(BaseModel):
    name: str
    arg_types: list[str]
    out_type: str
    body: dict[str, Any]
    policy: dict[str, Any] | None = None


class FormulaTuningSpec(BaseModel):
    """Deterministic local-search policy for a numeric formula parameter."""

    enabled: bool = True
    min: float | int
    max: float | int
    step: float | int
    radius: int = Field(default=1, ge=1, le=16)


class FormulaInputSpec(BaseModel):
    name: str
    type: str
    description: str = ""
    default: float | int | None = None
    role: Literal["data", "parameter"] | None = None
    tuning: FormulaTuningSpec | None = None


class FormulaConstraintSpec(BaseModel):
    left: str
    operator: Literal["lt", "le", "gt", "ge", "ne"]
    right: str


class FormulaSpec(BaseModel):
    name: str
    display_name: str = ""
    description: str = ""
    arg_types: list[str] = Field(default_factory=list)
    inputs: list[FormulaInputSpec] = Field(default_factory=list)
    out_type: str
    body: dict[str, Any]
    category: str = ""
    revision: int = 1
    runtime_name: str = ""
    created_at: str = ""
    updated_at: str = ""
    origin: str = "user_formula"
    editable: bool = True
    family: str = ""
    aliases: list[str] = Field(default_factory=list)
    catalog_revision: int | None = None
    constraints: list[FormulaConstraintSpec] = Field(default_factory=list)
    status: Literal["active", "retired"] = "active"
    replacement: str | None = None
    family_order: int = Field(default=0, ge=0)


class CategoryUpdate(BaseModel):
    order: list[str] | None = None
    overrides: dict[str, str] | None = None


class PrimitiveCategoryUpdate(BaseModel):
    category: str


class MembershipSpec(BaseModel):
    symbol: str
    entry: str
    exit: str | None = None


class UniverseSpec(BaseModel):
    name: str
    memberships: list[MembershipSpec]


class UniverseFolderCreate(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    parent: str | None = Field(default=None, max_length=120)


class UniverseFolderUpdate(BaseModel):
    """Rename and/or move a folder. Omit a field to keep it; ``parent: null`` moves to the top."""

    name: str | None = Field(default=None, min_length=1, max_length=200)
    parent: str | None = Field(default=None, max_length=120)


class UniversePlacementUpdate(BaseModel):
    universes: list[str] = Field(min_length=1, max_length=1_000)
    folder: str | None = Field(default=None, max_length=120)


class TrainingResourcesRequest(BaseModel):
    """Visible, device-relative CPU policy; execution settings never alter GP semantics."""

    profile: Literal["light", "auto", "maximum", "custom"] = "auto"
    cpu_budget_percent: int | None = Field(default=None, ge=10, le=100)


class RunRequest(BaseModel):
    universe: str = Field(default=_DEFAULT_UNIVERSE, min_length=1, max_length=80)
    as_of: str = Field(default=_DEFAULT_AS_OF, min_length=1, max_length=64)
    config: dict[str, Any] = Field(default_factory=dict, max_length=64)
    operators: list[OperatorSpec] = Field(default_factory=list, max_length=100)
    resources: TrainingResourcesRequest = Field(default_factory=TrainingResourcesRequest)


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


class FormulaTestSource(BaseModel):
    kind: Literal["draft", "saved"]
    body: dict[str, Any] | None = None
    inputs: list[FormulaInputSpec] = Field(default_factory=list, max_length=64)
    out_type: str | None = None
    runtime_name: str | None = None


class FormulaBinding(BaseModel):
    kind: Literal["field", "formula", "result", "literal"]
    field: str | None = None
    runtime_name: str | None = None
    result_id: str | None = None
    value: float | int | None = None


class PortfolioStrategyRequest(BaseModel):
    id: str = Field(min_length=1, max_length=80)
    scheme: Literal["quantile_ls", "rank_proportional"]
    quantile: float | None = Field(default=None, gt=0.0, lt=0.5)

    def to_spec(self) -> PortfolioStrategySpec:
        return PortfolioStrategySpec(self.id, self.scheme, self.quantile)


class FormulaTestRequest(BaseModel):
    source: FormulaTestSource
    bindings: dict[str, FormulaBinding] = Field(default_factory=dict, max_length=64)
    universe: str = Field(default=_DEFAULT_UNIVERSE, min_length=1, max_length=80)
    start: str | None = Field(default=None, max_length=64)
    end: str | None = Field(default=None, max_length=64)
    horizon: int = Field(default=1, ge=1, le=MAX_HORIZON)
    # Same meaning as the session setting; the formula test is exploratory, so the legacy
    # same-close default is kept for existing saved results and API callers.
    execution: Literal["close", "next_open", "next_close"] = "close"
    weighting_scheme: Literal["quantile_ls", "rank_proportional"] = "quantile_ls"
    quantile: float = Field(default=0.2, ge=0.01, le=0.49)
    commission_bps: float = Field(default=1.0, ge=0.0, le=10_000.0)
    slippage_bps: float = Field(default=5.0, ge=0.0, le=10_000.0)
    strategies: list[PortfolioStrategyRequest] | None = Field(
        default=None, min_length=1, max_length=4
    )
    primary_strategy_id: str | None = Field(default=None, min_length=1, max_length=80)


class FormulaTestKeepRequest(BaseModel):
    name: str = Field(default="Formula backtest", min_length=1, max_length=120)
    notes: str = Field(default="", max_length=10_000)


class SettingsUpdate(BaseModel):
    factors_dir: str | None = None
    tiingo_api_key: str | None = None
    evaluator: str | None = None
    # P9 explanation layer. The key follows the Tiingo contract exactly: an empty string clears
    # it, and it is never echoed back by GET /settings.
    llm_provider: str | None = None
    llm_model: str | None = None
    llm_base_url: str | None = None
    llm_api_key: str | None = None
    #: Which provider the key belongs to, when setting a key for a provider other than the
    #: currently selected one.
    llm_api_key_provider: str | None = None


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
    force: bool = False


class SymbolValidation(BaseModel):
    symbol: str
    valid: bool
    rows: int = 0
    first_date: str | None = None
    last_date: str | None = None
    provider: str | None = None
    error: str | None = None
    cached: bool = False


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
    symbols: list[str] = Field(default_factory=list, max_length=10_000)
    universe: str | None = Field(default=None, min_length=1, max_length=80)
    start: str
    end: str | None = None
    mode: str = "incremental"
    # Canonical cache symbol -> provider symbol. Universe submissions populate this from the
    # pinned definition; direct symbol submissions normally leave it empty.
    aliases: dict[str, str] = Field(default_factory=dict)


class DataSyncResult(BaseModel):
    symbol: str
    provider_symbol: str | None = None
    status: str
    rows_fetched: int = 0
    rows_cached: int = 0
    first_date: str | None = None
    last_date: str | None = None
    provider: str | None = None
    error: str | None = None
    quota_scope: str | None = None


class MembershipSyncRequest(BaseModel):
    symbols: list[str]
    expected_start: str


class MembershipSyncResult(BaseModel):
    symbol: str
    status: str  # resolved | unverified_stale | failed
    # Deprecated membership-shaped fields remain in the response for older clients. Price
    # availability is diagnostic evidence only, so these fields are intentionally inert.
    entry: str | None = None
    exit: str | None = None
    delisted: bool = False
    review_needed: bool = False
    first_date: str | None = None
    # ``list_date`` is the deprecated alias for ``first_date``.
    list_date: str | None = None
    last_date: str | None = None
    note: str | None = None
    error: str | None = None


_EVALUATORS = {"auto", "python", "cpp"}
_SYNC_MODES = {"incremental", "refresh"}


def _market_symbols(values: list[str]) -> list[str]:
    """Normalize an untrusted symbol list or expose a stable client error."""
    try:
        return sorted({normalize_market_symbol(value) for value in values if value.strip()})
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


class SessionCreateRequest(BaseModel):
    name: str = Field(default="Session", min_length=1, max_length=80)
    universe: str = Field(default=_DEFAULT_UNIVERSE, min_length=1, max_length=80)
    as_of: str = Field(default=_DEFAULT_AS_OF, min_length=1, max_length=64)
    config: dict[str, Any] = Field(default_factory=dict, max_length=64)
    operators: list[OperatorSpec] = Field(default_factory=list, max_length=100)
    seed_factor_ids: list[str] = Field(default_factory=list, max_length=100)
    train: float = Field(default=0.6, gt=0.0, lt=1.0)
    valid: float = Field(default=0.2, gt=0.0, lt=1.0)
    embargo: int = Field(default=5, ge=1, le=MAX_HORIZON)
    resources: TrainingResourcesRequest = Field(default_factory=TrainingResourcesRequest)


class SessionContinueRequest(BaseModel):
    generations: int = Field(default=5, ge=1, le=MAX_GENERATIONS)
    config: dict[str, Any] = Field(default_factory=dict, max_length=64)
    universe: str | None = Field(default=None, min_length=1, max_length=80)
    operators: list[OperatorSpec] = Field(default_factory=list, max_length=100)
    seed_factor_ids: list[str] = Field(default_factory=list, max_length=100)
    resources: TrainingResourcesRequest | None = None


class SessionFinalizeRequest(BaseModel):
    confirm_repeat: bool = False
    strategy_plan_id: str | None = Field(default=None, min_length=1, max_length=80)


class SessionStrategyComparisonRequest(BaseModel):
    strategies: list[PortfolioStrategyRequest] = Field(min_length=1, max_length=4)
    confirm_repeat: bool = False


class SessionFinalizationPlanRequest(BaseModel):
    comparison_id: str = Field(min_length=1, max_length=80)
    primary_strategy_id: str = Field(min_length=1, max_length=80)


def _acceleration_status() -> tuple[bool, str | None]:
    selected = cpp.selected_backend()
    if selected == "python":
        return False, "Python evaluator selected; parallel acceleration is disabled."
    if not cpp.available():
        return False, "C++ accelerator is unavailable; training uses one Python worker."
    return True, None


def _requested_resources(request: TrainingResourcesRequest) -> ResourcePolicy:
    try:
        return ResourcePolicy(request.profile, request.cpu_budget_percent)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def _resolve_training_resources(
    request: TrainingResourcesRequest, *, ic_method: str = "spearman"
) -> ResolvedResources:
    accelerated, reason = _acceleration_status()
    if accelerated and not cpp.supports_native_scoring(ic_method):
        accelerated = False
        reason = (
            f"{ic_method.title()} scoring uses the parity-safe Python evaluator; "
            "training is limited to one worker."
        )
    return resolve_resources(
        _requested_resources(request),
        accelerated=accelerated,
        fallback_reason=reason,
    )


def _formula_categories() -> dict[str, str]:
    """Category for each persisted formula runtime revision."""
    return {
        spec.runtime_name: spec.category
        for spec in _all_formula_specs()
        if spec.category and spec.runtime_name
    }


_RESERVED_LEAF_CATEGORIES = {core_categories.DATA, core_categories.CONSTANT}


def _category_override_allowed(name: str, category: object) -> bool:
    """Only calculation operators may be recategorized, never into leaf-only groups."""
    primitive = REGISTRY.get(name)
    normalized_category = category.strip() if isinstance(category, str) else ""
    return bool(
        primitive is not None
        and primitive.kind.value == "operator"
        and normalized_category
        and normalized_category.casefold() not in _RESERVED_LEAF_CATEGORIES
    )


def _formula_category(category: str) -> str:
    """Keep the data/constant headings leaf-only, including legacy case variants."""
    normalized = category.strip()
    if not normalized or normalized.casefold() in _RESERVED_LEAF_CATEGORIES:
        return core_categories.CUSTOM
    return normalized


def _resolve_category(name: str, *, is_user: bool, formula_categories: dict[str, str]) -> str:
    """Category for a primitive: user override > formula's own > built-in default > custom."""
    overrides = paths.read_categories().get("overrides", {})
    if (
        isinstance(overrides, dict)
        and name in overrides
        and _category_override_allowed(name, overrides[name])
    ):
        return str(overrides[name])
    if name in formula_categories:
        return formula_categories[name]
    if is_user:
        return core_categories.CUSTOM
    return core_categories.builtin_category(name)


def _allowed_operators(
    config: GPConfig,
    formula_runtime_names: set[str] | None = None,
    explicit_operator_names: set[str] | None = None,
) -> set[str] | None:
    """The operator names the GP may use, from the config's ``enabled_categories``.

    ``None`` (the default) leaves the GP on its default pool (condition category excluded), so
    the classic numeric search space is unchanged unless a run opts categories in.
    """
    enabled_set = set(config.enabled_categories or core_categories.DEFAULT_ENABLED_CATEGORIES)
    cats = _formula_categories()
    current = (
        set(formula_runtime_names)
        if formula_runtime_names is not None
        else {spec.runtime_name for spec in _read_formula_specs()}
    )
    current.update(explicit_operator_names or set())
    by_runtime = {spec.runtime_name: spec for spec in _all_formula_specs()}
    active_specs = [
        by_runtime[runtime_name]
        for runtime_name in current
        if runtime_name in by_runtime and by_runtime[runtime_name].status == "active"
    ]
    if config.enabled_formula_names is not None:
        available_names = {spec.name for spec in active_specs}
        unknown = sorted(set(config.enabled_formula_names) - available_names)
        if unknown:
            raise HTTPException(
                status_code=400,
                detail=f"unknown or retired enabled formula(s): {', '.join(unknown)}",
            )
        selected_names = set(config.enabled_formula_names)
        active_specs = [spec for spec in active_specs if spec.name in selected_names]
    active_formula_runtimes = {spec.runtime_name for spec in active_specs}
    return {
        prim.name
        for prim in OPERATORS.values()
        if (prim.macro_body is None or prim.name in current)
        if prim.macro_body is None
        or prim.name in active_formula_runtimes
        or prim.name in (explicit_operator_names or set())
        if _resolve_category(
            prim.name, is_user=prim.macro_body is not None, formula_categories=cats
        )
        in enabled_set
    }


def _primitive_info(
    prim: Primitive, formula_categories: dict[str, str] | None = None
) -> dict[str, Any]:
    is_user = prim.macro_body is not None
    cats = formula_categories if formula_categories is not None else _formula_categories()
    formula = _formula_for_runtime(prim.name) if is_user else None
    doc = (
        {
            "display_name": formula.display_name or formula.name.replace("_", " ").title(),
            "description": formula.description or "User-defined typed formula.",
            "inputs": [
                {
                    "name": item.name,
                    "description": item.description,
                    "default": item.default,
                    "role": item.role,
                    "tuning": _model_dump(item.tuning) if item.tuning is not None else None,
                }
                for item in formula.inputs
            ],
        }
        if formula is not None
        else primitive_doc(prim.name, prim.arity)
    )
    inputs = []
    for index, arg_type in enumerate(prim.arg_types):
        meta = doc["inputs"][index] if index < len(doc["inputs"]) else {}
        item = {
            "name": str(meta.get("name") or f"input_{index + 1}"),
            "type": arg_type.value,
            "description": str(meta.get("description") or "Function input."),
        }
        if meta.get("default") is not None:
            item["default"] = meta["default"]
        if meta.get("role") is not None:
            item["role"] = meta["role"]
        if meta.get("tuning") is not None:
            item["tuning"] = meta["tuning"]
        inputs.append(item)
    if prim.kind.value == "operand":
        origin = "data"
    elif prim.kind.value == "ephemeral":
        origin = "value"
    elif is_user:
        origin = formula.origin if formula is not None else "user_formula"
    else:
        origin = "builtin"
    return {
        "name": prim.name,
        "logical_name": formula.name if formula is not None else prim.name,
        "display_name": doc["display_name"],
        "description": doc["description"],
        "kind": prim.kind.value,
        "arg_types": [t.value for t in prim.arg_types],
        "inputs": inputs,
        "out_type": prim.out_type.value,
        "user": is_user,
        "origin": origin,
        "editable": formula.editable if formula is not None else is_user,
        "category": _resolve_category(prim.name, is_user=is_user, formula_categories=cats),
        "revision": formula.revision if formula is not None else None,
        "runtime_name": prim.name,
        "family": formula.family if formula is not None else "",
        "aliases": formula.aliases if formula is not None else [],
        "catalog_revision": formula.catalog_revision if formula is not None else None,
        "constraints": (
            [_model_dump(item) for item in formula.constraints] if formula is not None else []
        ),
        "status": formula.status if formula is not None else "active",
        "replacement": formula.replacement if formula is not None else None,
        "family_order": formula.family_order if formula is not None else 0,
    }


def _register(spec: OperatorSpec) -> Primitive:
    try:
        return register_operator(
            spec.name,
            [DType(t) for t in spec.arg_types],
            DType(spec.out_type),
            spec.body,
            policy=spec.policy,
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


def _formula_inputs(
    arg_types: list[str], supplied: list[FormulaInputSpec]
) -> list[FormulaInputSpec]:
    if supplied and len(supplied) != len(arg_types):
        raise HTTPException(status_code=400, detail="formula inputs must match argument types")
    inputs = supplied or [
        FormulaInputSpec(name=f"input_{index + 1}", type=arg_type)
        for index, arg_type in enumerate(arg_types)
    ]
    seen: set[str] = set()
    normalized: list[FormulaInputSpec] = []
    for index, item in enumerate(inputs):
        name = re.sub(r"[^a-z0-9_]+", "_", item.name.strip().lower()).strip("_")
        if not _FORMULA_INPUT_RE.fullmatch(name):
            raise HTTPException(status_code=400, detail=f"invalid formula input name {item.name!r}")
        if name in seen:
            raise HTTPException(status_code=400, detail=f"duplicate formula input {name!r}")
        if item.type != arg_types[index]:
            raise HTTPException(
                status_code=400,
                detail=f"input {name!r} type does not match argument {index + 1}",
            )
        default = item.default
        role = item.role or (
            "parameter"
            if item.type in {DType.WINDOW.value, DType.SCALAR.value}
            else "data"
        )
        if role == "data" and item.type in {DType.WINDOW.value, DType.SCALAR.value}:
            raise HTTPException(
                status_code=400,
                detail=f"numeric input {name!r} must use the parameter role",
            )
        if role == "parameter" and item.type not in {
            DType.WINDOW.value,
            DType.SCALAR.value,
        }:
            raise HTTPException(
                status_code=400,
                detail=f"connectable input {name!r} must use the data role",
            )
        if default is not None:
            if item.type == DType.WINDOW.value:
                if isinstance(default, bool) or not isinstance(default, int):
                    raise HTTPException(
                        status_code=400,
                        detail=f"default for window input {name!r} must be an integer",
                    )
                try:
                    validate_tree(Node("window", value=default))
                except Exception as exc:  # noqa: BLE001 - normalize as a client contract error
                    raise HTTPException(status_code=400, detail=str(exc)) from exc
            elif item.type == DType.SCALAR.value:
                if (
                    isinstance(default, bool)
                    or not isinstance(default, (int, float))
                    or not math.isfinite(float(default))
                ):
                    raise HTTPException(
                        status_code=400,
                        detail=f"default for scalar input {name!r} must be finite",
                    )
            else:
                raise HTTPException(
                    status_code=400,
                    detail=f"only scalar and window inputs may define defaults ({name!r})",
                )
        tuning = item.tuning
        if tuning is not None:
            if item.type not in {DType.WINDOW.value, DType.SCALAR.value}:
                raise HTTPException(
                    status_code=400,
                    detail=f"only numeric input {name!r} may define a tuning policy",
                )
            numeric = (tuning.min, tuning.max, tuning.step)
            if any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                for value in numeric
            ):
                raise HTTPException(
                    status_code=400,
                    detail=f"tuning bounds for {name!r} must be finite numbers",
                )
            if tuning.min > tuning.max or tuning.step <= 0:
                raise HTTPException(
                    status_code=400,
                    detail=f"invalid tuning range for {name!r}",
                )
            if item.type == DType.WINDOW.value and any(
                isinstance(value, bool) or not isinstance(value, int) for value in numeric
            ):
                raise HTTPException(
                    status_code=400,
                    detail=f"window tuning values for {name!r} must be integers",
                )
            if item.type == DType.WINDOW.value:
                try:
                    validate_tree(Node("window", value=tuning.min))
                    validate_tree(Node("window", value=tuning.max))
                except Exception as exc:  # noqa: BLE001 - normalize as a client contract error
                    raise HTTPException(status_code=400, detail=str(exc)) from exc
            if default is None:
                raise HTTPException(
                    status_code=400,
                    detail=f"tunable input {name!r} requires a default",
                )
            if not tuning.min <= default <= tuning.max:
                raise HTTPException(
                    status_code=400,
                    detail=f"default for {name!r} must be within its tuning range",
                )
        seen.add(name)
        normalized.append(
            FormulaInputSpec(
                name=name,
                type=item.type,
                description=item.description.strip(),
                default=default,
                role=role,
                tuning=tuning,
            )
        )
    return normalized


def _formula_constraints(
    constraints: list[FormulaConstraintSpec], inputs: list[FormulaInputSpec]
) -> list[FormulaConstraintSpec]:
    by_name = {item.name: item for item in inputs}
    normalized: list[FormulaConstraintSpec] = []
    comparisons = {
        "lt": lambda left, right: left < right,
        "le": lambda left, right: left <= right,
        "gt": lambda left, right: left > right,
        "ge": lambda left, right: left >= right,
        "ne": lambda left, right: left != right,
    }
    for item in constraints:
        left = item.left.strip().lower()
        right = item.right.strip().lower()
        if left not in by_name or right not in by_name:
            raise HTTPException(
                status_code=400,
                detail=f"formula constraint references unknown inputs {left!r}, {right!r}",
            )
        if by_name[left].type not in {DType.WINDOW.value, DType.SCALAR.value} or by_name[
            right
        ].type not in {DType.WINDOW.value, DType.SCALAR.value}:
            raise HTTPException(
                status_code=400,
                detail="formula constraints may compare only numeric inputs",
            )
        normalized.append(
            FormulaConstraintSpec(left=left, operator=item.operator, right=right)
        )
        left_default, right_default = by_name[left].default, by_name[right].default
        if (
            left_default is not None
            and right_default is not None
            and not comparisons[item.operator](left_default, right_default)
        ):
            raise HTTPException(
                status_code=400,
                detail=f"formula defaults violate constraint {left} {item.operator} {right}",
            )
    return normalized


def _normalize_formula_spec(
    spec: FormulaSpec,
    *,
    revision: int | None = None,
    runtime_name: str | None = None,
    created_at: str | None = None,
) -> FormulaSpec:
    name = _normalize_formula_name(spec.name)
    arg_types = list(spec.arg_types or [item.type for item in spec.inputs])
    for arg_type in arg_types:
        try:
            DType(arg_type)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=f"unknown input type {arg_type!r}") from exc
    try:
        DType(spec.out_type)
    except ValueError as exc:
        raise HTTPException(
            status_code=400, detail=f"unknown output type {spec.out_type!r}"
        ) from exc
    rev = revision if revision is not None else max(1, int(spec.revision or 1))
    now = _now_iso()
    inputs = _formula_inputs(arg_types, list(spec.inputs))
    replacement = spec.replacement.strip() if spec.replacement else None
    if replacement is not None:
        replacement = _normalize_formula_name(replacement)
        if replacement == name:
            raise HTTPException(status_code=400, detail="formula cannot replace itself")
    return FormulaSpec(
        name=name,
        display_name=spec.display_name.strip() or name.replace("_", " ").title(),
        description=spec.description.strip(),
        arg_types=arg_types,
        inputs=inputs,
        out_type=spec.out_type,
        body=spec.body,
        category=_formula_category(spec.category),
        revision=rev,
        runtime_name=runtime_name or spec.runtime_name or (name if rev == 1 else f"{name}__r{rev}"),
        created_at=created_at or spec.created_at or now,
        updated_at=now,
        origin=spec.origin.strip() or "user_formula",
        editable=bool(spec.editable),
        family=spec.family.strip(),
        aliases=list(dict.fromkeys(alias.strip() for alias in spec.aliases if alias.strip())),
        catalog_revision=spec.catalog_revision,
        constraints=_formula_constraints(list(spec.constraints), inputs),
        status=spec.status,
        replacement=replacement,
        family_order=spec.family_order,
    )


def _formula_policy(spec: FormulaSpec) -> dict[str, Any]:
    """JSON-safe parameter contract pinned with every registered formula revision."""
    return {
        "inputs": [_model_dump(item) for item in spec.inputs],
        "constraints": [_model_dump(item) for item in spec.constraints],
        "status": spec.status,
        "catalog_revision": spec.catalog_revision,
    }


def _legacy_formula_store(items: list[Any]) -> dict[str, Any]:
    families: list[dict[str, Any]] = []
    for item in items:
        spec = _normalize_formula_spec(FormulaSpec(**item), revision=1)
        families.append({"name": spec.name, "latest_revision": 1, "revisions": [_model_dump(spec)]})
    return {"schema_version": 2, "families": families}


def _catalog_signature(spec: FormulaSpec) -> dict[str, Any]:
    """Stable managed-content signature; storage timestamps/runtime revision are excluded."""
    payload = _model_dump(spec)
    for key in ("revision", "runtime_name", "created_at", "updated_at"):
        payload.pop(key, None)
    return payload


def _pin_catalog_body(
    body: dict[str, Any], runtime_names: dict[str, str]
) -> dict[str, Any]:
    name = str(body["name"])
    result: dict[str, Any] = {"name": runtime_names.get(name, name)}
    if "value" in body:
        result["value"] = body["value"]
    if body.get("children"):
        result["children"] = [
            _pin_catalog_body(child, runtime_names) for child in body["children"]
        ]
    return result


def _merge_indicator_catalog(store: dict[str, Any]) -> bool:
    """Idempotently add/update the packaged catalog while preserving every published revision."""
    changed = False
    pinned: dict[str, str] = {}
    families = {str(item.get("name")): item for item in store["families"]}
    for definition in INDICATOR_CATALOG:
        name = str(definition["name"])
        dependencies = _tree_names(definition["body"]) & CATALOG_NAMES
        # A legacy user formula occupying a reserved ta_* name is never overwritten. Catalog
        # dependents are skipped instead of silently binding to user-controlled calculations.
        if not dependencies <= pinned.keys():
            continue
        family = families.get(name)
        if family is not None:
            latest = _family_latest(family)
            if latest.origin != CATALOG_ORIGIN:
                continue
            revision = latest.revision
            runtime_name = latest.runtime_name
            created_at = latest.created_at
        else:
            latest = None
            revision = 1
            runtime_name = name
            created_at = None

        candidate = _normalize_formula_spec(
            FormulaSpec(
                **{
                    **definition,
                    "body": _pin_catalog_body(definition["body"], pinned),
                }
            ),
            revision=revision,
            runtime_name=runtime_name,
            created_at=created_at,
        )
        if latest is None:
            family = {
                "name": name,
                "latest_revision": 1,
                "revisions": [_model_dump(candidate)],
            }
            store["families"].append(family)
            families[name] = family
            changed = True
        elif _catalog_signature(candidate) != _catalog_signature(latest):
            calculation_changed = (
                candidate.body != latest.body
                or candidate.arg_types != latest.arg_types
                or candidate.out_type != latest.out_type
                or candidate.catalog_revision != latest.catalog_revision
            )
            if calculation_changed:
                next_revision = latest.revision + 1
                candidate = _normalize_formula_spec(
                    candidate,
                    revision=next_revision,
                    runtime_name=f"{name}__r{next_revision}",
                    created_at=latest.created_at,
                )
                family["revisions"].append(_model_dump(candidate))
                family["latest_revision"] = next_revision
            else:
                candidate = _normalize_formula_spec(
                    candidate,
                    revision=latest.revision,
                    runtime_name=latest.runtime_name,
                    created_at=latest.created_at,
                )
                family["revisions"] = [
                    _model_dump(candidate)
                    if int(item.get("revision", 1)) == latest.revision
                    else item
                    for item in family["revisions"]
                ]
            changed = True
        pinned[name] = candidate.runtime_name
    # Retire managed families removed or renamed by the packaged catalog. Publishing a new
    # metadata revision (rather than rewriting the old one) preserves the active runtime/status
    # pinned by historical sessions and saved formulas.
    for name, family in families.items():
        if name in CATALOG_NAMES:
            continue
        latest = _family_latest(family)
        if latest.origin != CATALOG_ORIGIN or latest.status == "retired":
            continue
        next_revision = latest.revision + 1
        replacement = LEGACY_CATALOG_REPLACEMENTS.get(name)
        retired = _normalize_formula_spec(
            FormulaSpec(
                **{
                    **_model_dump(latest),
                    "display_name": f"{latest.display_name} (retired)",
                    "description": (
                        f"{latest.description} Legacy managed formula retained for pinned "
                        "dependencies."
                    ).strip(),
                    "catalog_revision": CATALOG_REVISION,
                    "status": "retired",
                    "replacement": replacement,
                }
            ),
            revision=next_revision,
            runtime_name=f"{name}__r{next_revision}",
            created_at=latest.created_at,
        )
        family["revisions"].append(_model_dump(retired))
        family["latest_revision"] = next_revision
        changed = True
    return changed


def _read_formula_store() -> dict[str, Any]:
    path = paths.formulas_path()
    if not path.exists():
        store = {"schema_version": 2, "families": []}
    else:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise HTTPException(status_code=500, detail="invalid formula store") from exc
        if isinstance(payload, list):
            store = _legacy_formula_store(payload)
        else:
            if not isinstance(payload, dict) or not isinstance(payload.get("families"), list):
                raise HTTPException(status_code=500, detail="invalid formula store")
            families: list[dict[str, Any]] = []
            for raw in payload["families"]:
                revisions = [
                    _model_dump(_normalize_formula_spec(FormulaSpec(**item)))
                    for item in raw.get("revisions", [])
                ]
                if not revisions:
                    continue
                latest = int(
                    raw.get("latest_revision") or max(item["revision"] for item in revisions)
                )
                families.append(
                    {
                        "name": str(raw.get("name") or revisions[-1]["name"]),
                        "latest_revision": latest,
                        "revisions": revisions,
                    }
                )
            store = {"schema_version": 2, "families": families}
    if _merge_indicator_catalog(store):
        _write_formula_store(store)
    return store


def _write_formula_store(store: dict[str, Any]) -> None:
    paths.meta_dir().mkdir(parents=True, exist_ok=True)
    atomic_write_text(
        paths.formulas_path(),
        json.dumps(store, indent=2, sort_keys=True),
    )


def _write_formula_specs(specs: list[FormulaSpec]) -> None:
    """Compatibility helper used by older callers and tests."""
    _write_formula_store(
        {
            "schema_version": 2,
            "families": [
                {
                    "name": spec.name,
                    "latest_revision": spec.revision,
                    "revisions": [_model_dump(_normalize_formula_spec(spec))],
                }
                for spec in specs
            ],
        }
    )


def _family_latest(family: dict[str, Any]) -> FormulaSpec:
    latest = int(family["latest_revision"])
    item = next(
        (entry for entry in family["revisions"] if int(entry.get("revision", 1)) == latest),
        family["revisions"][-1],
    )
    return _normalize_formula_spec(FormulaSpec(**item))


def _read_formula_specs() -> list[FormulaSpec]:
    return [_family_latest(family) for family in _read_formula_store()["families"]]


def _all_formula_specs(store: dict[str, Any] | None = None) -> list[FormulaSpec]:
    current = store or _read_formula_store()
    return [
        _normalize_formula_spec(FormulaSpec(**item))
        for family in current["families"]
        for item in family["revisions"]
    ]


def _formula_for_runtime(runtime_name: str) -> FormulaSpec | None:
    return next(
        (spec for spec in _all_formula_specs() if spec.runtime_name == runtime_name),
        None,
    )


def _tree_uses(node: Node, primitive_name: str) -> bool:
    return node.name == primitive_name or any(
        _tree_uses(child, primitive_name) for child in node.children
    )


def _formula_operator_spec(spec: FormulaSpec) -> OperatorSpec:
    return OperatorSpec(
        name=spec.runtime_name,
        arg_types=spec.arg_types,
        out_type=spec.out_type,
        body=spec.body,
        policy=_formula_policy(spec),
    )


def _formula_sort(
    specs: list[FormulaSpec],
) -> tuple[list[FormulaSpec], dict[str, str]]:
    """Topologically order runtime revisions and report persisted cycles/dependents."""
    by_name = {spec.runtime_name: spec for spec in specs}
    dependencies = {
        spec.runtime_name: sorted(_tree_names(spec.body) & by_name.keys()) for spec in specs
    }
    state: dict[str, int] = {}
    stack: list[str] = []
    errors: dict[str, str] = {}

    def scan(runtime_name: str) -> None:
        current = state.get(runtime_name, 0)
        if current == 2:
            return
        if current == 1:
            start = stack.index(runtime_name)
            cycle = [*stack[start:], runtime_name]
            message = f"formula dependency cycle: {' -> '.join(cycle)}"
            for item in cycle[:-1]:
                errors[item] = message
            return
        state[runtime_name] = 1
        stack.append(runtime_name)
        for dependency in dependencies[runtime_name]:
            scan(dependency)
        stack.pop()
        state[runtime_name] = 2

    for spec in specs:
        scan(spec.runtime_name)

    # Acyclic callers of a corrupt cycle cannot be safely registered either.  Propagate a
    # concise dependency error so every affected formula remains visible but unusable.
    changed = True
    while changed:
        changed = False
        for runtime_name, items in dependencies.items():
            if runtime_name in errors:
                continue
            broken = next((item for item in items if item in errors), None)
            if broken is not None:
                errors[runtime_name] = f"formula dependency {broken!r} is invalid: {errors[broken]}"
                changed = True

    ordered: list[FormulaSpec] = []
    placed: set[str] = set()

    def place(runtime_name: str) -> None:
        if runtime_name in placed or runtime_name in errors:
            return
        for dependency in dependencies[runtime_name]:
            place(dependency)
        placed.add(runtime_name)
        ordered.append(by_name[runtime_name])

    for spec in specs:
        place(spec.runtime_name)
    return ordered, errors


def _topo_sort_formulas(specs: list[FormulaSpec]) -> list[FormulaSpec]:
    ordered, errors = _formula_sort(specs)
    if errors:
        first = next(iter(errors.values()))
        raise InvalidOperator(first)
    return ordered


def _assert_latest_formula_dag(store: dict[str, Any]) -> None:
    """Reject logical family cycles, even when their runtime revisions remain individually DAGs."""
    latest = [_family_latest(family) for family in store["families"]]
    runtime_to_family = {
        spec.runtime_name: spec.name for spec in _all_formula_specs(store)
    }
    graph = {
        spec.name: sorted(
            {
                runtime_to_family[name]
                for name in _tree_names(spec.body)
                if name in runtime_to_family
            }
        )
        for spec in latest
    }
    state: dict[str, int] = {}
    stack: list[str] = []

    def visit(name: str) -> None:
        current = state.get(name, 0)
        if current == 2:
            return
        if current == 1:
            start = stack.index(name)
            cycle = [*stack[start:], name]
            raise InvalidOperator(f"formula dependency cycle: {' -> '.join(cycle)}")
        state[name] = 1
        stack.append(name)
        for dependency in graph.get(name, []):
            visit(dependency)
        stack.pop()
        state[name] = 2

    for name in graph:
        visit(name)


def _load_persisted_formulas() -> list[dict[str, Any]]:
    specs = _all_formula_specs()
    expected_runtime_names = {spec.runtime_name for spec in specs}
    # Runtime registries are process-global while tests and embedded callers may switch the data
    # directory. Reserved managed names absent from the active store must not leak across stores.
    for runtime_name in list(USER_OPERATORS):
        if runtime_name.startswith("ta_") and runtime_name not in expected_runtime_names:
            unregister_operator(runtime_name)
    status: dict[str, tuple[bool, str | None]] = {}
    ordered, dependency_errors = _formula_sort(specs)
    for runtime_name, dependency_error in dependency_errors.items():
        unregister_operator(runtime_name)
        status[runtime_name] = (False, dependency_error)
    for spec in ordered:
        try:
            existing = USER_OPERATORS.get(spec.runtime_name)
            expected_body = tree_from_dict(spec.body)
            if existing is not None and (
                existing.arg_types != tuple(DType(t) for t in spec.arg_types)
                or existing.out_type != DType(spec.out_type)
                or existing.macro_body != expected_body
                or existing.macro_policy != _formula_policy(spec)
            ):
                unregister_operator(spec.runtime_name)
            ensure_operator(
                spec.runtime_name,
                [DType(t) for t in spec.arg_types],
                DType(spec.out_type),
                spec.body,
                policy=_formula_policy(spec),
            )
            status[spec.runtime_name] = (True, None)
        except (InvalidOperator, ValueError) as exc:
            status[spec.runtime_name] = (False, str(exc))
    latest = _read_formula_specs()
    formulas: list[dict[str, Any]] = []
    for spec in latest:
        registered, registration_error = status.get(spec.runtime_name, (False, None))
        formulas.append(
            {
                **_model_dump(spec),
                "registered": registered,
                "error": registration_error,
            }
        )
    return formulas


def _formula_family(store: dict[str, Any], name: str) -> dict[str, Any] | None:
    return next((family for family in store["families"] if family["name"] == name), None)


def _replace_tree_names(tree: dict[str, Any], replacements: dict[str, str]) -> dict[str, Any]:
    result: dict[str, Any] = {"name": replacements.get(str(tree["name"]), str(tree["name"]))}
    if "value" in tree:
        result["value"] = tree["value"]
    if tree.get("children"):
        result["children"] = [
            _replace_tree_names(child, replacements) for child in tree["children"]
        ]
    return result


def _tree_names(tree: dict[str, Any]) -> set[str]:
    names = {str(tree.get("name", ""))}
    for child in tree.get("children", []):
        names.update(_tree_names(child))
    return names


def _formula_impact(name: str, proposed: FormulaSpec | None = None) -> dict[str, Any]:
    store = _read_formula_store()
    family = _formula_family(store, name)
    if family is None:
        raise HTTPException(status_code=404, detail="unknown formula")
    current = _family_latest(family)
    target_runtime_names = {
        str(item.get("runtime_name") or item.get("name")) for item in family["revisions"]
    }
    direct: list[str] = []
    reverse: dict[str, set[str]] = {}
    all_specs = _all_formula_specs(store)
    runtime_to_family = {spec.runtime_name: spec.name for spec in all_specs}
    for spec in all_specs:
        used = _tree_names(spec.body)
        for runtime_name in used:
            dependency_family = runtime_to_family.get(runtime_name)
            if dependency_family is not None and dependency_family != spec.name:
                reverse.setdefault(dependency_family, set()).add(spec.name)
        if spec.name != name and used & target_runtime_names:
            direct.append(spec.name)

    transitive: set[str] = set(direct)
    pending = list(direct)
    while pending:
        dependency = pending.pop()
        for caller in reverse.get(dependency, set()):
            if caller not in transitive and caller != name:
                transitive.add(caller)
                pending.append(caller)

    factor_refs: list[str] = []
    try:
        for factor in _factor_store().list():
            required = {item.get("name") for item in factor.required_operators}
            if target_runtime_names & required or target_runtime_names & {
                node.name for node in factor.tree.iter_nodes()
            }:
                factor_refs.append(factor.id)
    except OSError:
        pass

    session_refs: list[str] = []
    root = paths.sessions_dir()
    if root.exists():
        for directory in root.iterdir():
            session_file = directory / "session.json"
            if not session_file.exists():
                continue
            try:
                session = json.loads(session_file.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            pinned = {
                item.get("runtime_name") or item.get("name")
                for item in session.get("formula_revisions", [])
            }
            if target_runtime_names & pinned:
                session_refs.append(str(session.get("id") or directory.name))

    active_run_refs: list[str] = []
    for job in _jobs.list():
        if job.status not in {"queued", "running"}:
            continue
        pinned = {
            item.get("runtime_name") or item.get("name")
            for item in job.metadata.get("formula_revisions", [])
            if isinstance(item, dict)
        }
        if target_runtime_names & pinned:
            active_run_refs.append(job.id)

    change = "none"
    if proposed is not None:
        normalized = _normalize_formula_spec(
            proposed,
            revision=current.revision,
            runtime_name=current.runtime_name,
            created_at=current.created_at,
        )
        if (
            normalized.body != current.body
            or normalized.arg_types != current.arg_types
            or normalized.out_type != current.out_type
            or normalized.inputs != current.inputs
            or normalized.constraints != current.constraints
            or normalized.status != current.status
            or normalized.replacement != current.replacement
        ):
            change = "calculation"
        elif (
            normalized.display_name != current.display_name
            or normalized.description != current.description
            or normalized.category != current.category
            or normalized.family != current.family
            or normalized.aliases != current.aliases
            or normalized.family_order != current.family_order
        ):
            change = "metadata"
    return {
        "name": name,
        "runtime_name": current.runtime_name,
        "change": change,
        "direct_formulas": sorted(set(direct)),
        "transitive_formulas": sorted(transitive),
        "factors": sorted(set(factor_refs)),
        "sessions": sorted(set(session_refs)),
        "runs": sorted(set(active_run_refs)),
        "has_references": bool(transitive or factor_refs or session_refs or active_run_refs),
    }


def _active_formula_operator_specs() -> list[dict[str, Any]]:
    return [
        {
            **_model_dump(spec),
            "name": spec.runtime_name,
        }
        for spec in _read_formula_specs()
    ]


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


def _request_timestamp(value: str, *, label: str) -> pd.Timestamp:
    try:
        timestamp = pd.Timestamp(value)
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=f"invalid {label} date {value!r}") from exc
    if pd.isna(timestamp):
        raise HTTPException(status_code=400, detail=f"invalid {label} date {value!r}")
    if timestamp.tzinfo is not None:
        timestamp = timestamp.tz_convert("UTC").tz_localize(None)
    return timestamp.normalize()


def _request_date_range(
    start: str | None,
    end: str | None,
) -> tuple[str | None, str | None]:
    start_ts = _request_timestamp(start, label="start") if start is not None else None
    end_ts = _request_timestamp(end, label="end") if end is not None else None
    if start_ts is not None and end_ts is not None and end_ts < start_ts:
        raise HTTPException(status_code=400, detail="end date must not precede start date")
    return (
        start_ts.date().isoformat() if start_ts is not None else None,
        end_ts.date().isoformat() if end_ts is not None else None,
    )


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
        try:
            symbol = validate_market_symbol(str(raw.get("symbol") or raw.get("ticker") or ""))
        except ValueError:
            continue
        if symbol in seen:
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


_verified_symbols: dict[str, SymbolValidation] = {}


def _validate_symbol(
    symbol: str,
    start: str | None = None,
    end: str | None = None,
    *,
    force: bool = False,
) -> SymbolValidation:
    try:
        clean = validate_market_symbol(symbol)
    except ValueError as exc:
        return SymbolValidation(
            symbol=symbol.strip().upper(),
            valid=False,
            error=str(exc),
        )
    cache_key = f"{clean}|{start}|{end}"
    if not force and cache_key in _verified_symbols:
        return SymbolValidation(**{**_model_dump(_verified_symbols[cache_key]), "cached": True})
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
    result = SymbolValidation(
        symbol=clean,
        valid=True,
        rows=len(frame),
        first_date=_date_iso(frame.index.min()),
        last_date=_date_iso(frame.index.max()),
        provider=_provider_source(provider, clean),
    )
    _verified_symbols[cache_key] = result
    return result


def _coverage_for_symbol(
    symbol: str,
    start: str | None = None,
    end: str | None = None,
    cache: ParquetCache | None = None,
) -> DataCoverage:
    clean = validate_market_symbol(symbol)
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
    return merge_price_frames(frames)


def _sync_one_symbol(
    symbol: str,
    *,
    provider_symbol: str | None = None,
    start: str,
    end: str | None,
    mode: str,
    provider: PriceProvider,
    cache: ParquetCache,
) -> DataSyncResult:
    clean = validate_market_symbol(symbol)
    provider_clean = validate_market_symbol(provider_symbol or clean)
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
                provider_symbol=provider_clean,
                status="skipped",
                rows_cached=len(existing),
                first_date=_date_iso(existing.index.min()),
                last_date=_date_iso(existing.index.max()),
                provider=_provider_source(provider, provider_clean),
            )

        fetched: list[pd.DataFrame] = []
        for range_start, range_end in fetch_ranges:
            frame = provider.get_prices(provider_clean, range_start, range_end)
            if not frame.empty:
                fetched.append(frame)
        if not fetched and existing is None:
            return DataSyncResult(
                symbol=clean,
                provider_symbol=provider_clean,
                status="failed",
                provider=_provider_source(provider, provider_clean),
                error="provider returned no rows",
            )

        frames = [*fetched] if existing is None else [existing, *fetched]
        merged = _merge_price_frames(frames)
        cache.store(clean, merged)
        stored = cache.load(clean)
        return DataSyncResult(
            symbol=clean,
            provider_symbol=provider_clean,
            status="fetched" if fetched else "skipped",
            rows_fetched=sum(len(frame) for frame in fetched),
            rows_cached=len(stored),
            first_date=_date_iso(stored.index.min()),
            last_date=_date_iso(stored.index.max()),
            provider=_provider_source(provider, provider_clean),
        )
    except QuotaExceededError as exc:
        # An exhausted allowance is a batch-level stop, reported distinctly from a symbol failure.
        return DataSyncResult(
            symbol=clean,
            provider_symbol=provider_clean,
            status="quota_exceeded",
            provider=exc.provider or provider.name,
            error=str(exc),
            quota_scope=exc.scope,
        )
    except Exception as exc:  # noqa: BLE001 - one failed symbol should not abort the batch
        return DataSyncResult(
            symbol=clean,
            provider_symbol=provider_clean,
            status="failed",
            provider=provider.name,
            error=str(exc),
        )


def _run_data_sync(
    req: DataSyncRequest,
    progress: SyncProgress | None = None,
    stop: Any = None,
) -> dict[str, Any]:
    provider = _price_provider()
    cache = ParquetCache()
    results: list[DataSyncResult] = []
    stopped = False
    quota: dict[str, str] | None = None
    symbols = [symbol for symbol in req.symbols if symbol.strip()]
    attempted = 0
    for symbol in symbols:
        if stop is not None and stop():
            stopped = True
            break
        attempted += 1
        result = _sync_one_symbol(
            symbol,
            provider_symbol=req.aliases.get(symbol, symbol),
            start=req.start,
            end=req.end,
            mode=req.mode,
            provider=provider,
            cache=cache,
        )
        results.append(result)
        if progress is not None:
            progress.advance(symbol)
        if result.status == "quota_exceeded":
            quota = {
                "provider": result.provider or provider.name,
                "scope": result.quota_scope or "unknown",
                "message": result.error or "",
            }
            break
    failed_count = sum(result.status == "failed" for result in results)
    succeeded_count = sum(result.status in {"fetched", "skipped"} for result in results)
    payload: dict[str, Any] = {
        "mode": req.mode,
        "universe": req.universe,
        "start": req.start,
        "end": req.end,
        "results": [_model_dump(result) for result in results],
        "failed_count": failed_count,
        "succeeded_count": succeeded_count,
        "termination_reason": (
            "quota_exceeded" if quota is not None else "user_stopped" if stopped else "completed"
        ),
    }
    if quota is not None:
        payload["quota"] = quota
        payload["not_attempted"] = symbols[attempted:]
    return payload


# Weekends, market holidays, and ordinary provider lag can leave a short price tail.  A
# longer gap is surfaced for review, but is never treated as proof that the security or an
# index membership ended.  Only an explicit membership/status source may establish an exit.
_STALE_PRICE_REVIEW_DAYS = 10
_EARLIEST_HISTORY_START = "1900-01-01"


def _resolve_membership_dates(
    symbol: str, expected_start: str, provider: PriceProvider
) -> MembershipSyncResult:
    try:
        clean = validate_market_symbol(symbol)
    except ValueError as exc:
        return MembershipSyncResult(symbol=symbol.strip().upper(), status="failed", error=str(exc))
    try:
        frame = provider.get_prices(clean, _EARLIEST_HISTORY_START, _today_iso())
    except QuotaExceededError:
        raise
    except Exception as exc:  # noqa: BLE001 - one symbol's failure must not abort the batch
        return MembershipSyncResult(symbol=clean, status="failed", error=str(exc))
    if frame.empty:
        return MembershipSyncResult(
            symbol=clean, status="failed", error="provider returned no rows"
        )

    list_date = frame.index.min()
    last_date = frame.index.max()
    today = pd.Timestamp(_today_iso())
    stale = bool((today - last_date).days > _STALE_PRICE_REVIEW_DAYS)
    # ``expected_start`` remains part of the request for wire compatibility, but a price
    # provider is not an authoritative source for either index membership or listing status.
    # In particular, first price availability may reflect provider coverage rather than an IPO.
    _ = expected_start
    first_date = _date_iso(list_date)
    if stale:
        return MembershipSyncResult(
            symbol=clean,
            status="unverified_stale",
            # Price history is not authoritative membership data.  In particular, a
            # transient provider failure, symbol change, or stale feed must never mutate
            # the user's entry/exit interval.
            entry=None,
            exit=None,
            delisted=False,
            review_needed=True,
            first_date=first_date,
            list_date=first_date,
            last_date=_date_iso(last_date),
            note=(
                "Price history ends well before today. Membership was left unchanged; "
                "confirm any exit with authoritative constituent or exchange-status data."
            ),
        )
    return MembershipSyncResult(
        symbol=clean,
        status="resolved",
        entry=None,
        exit=None,
        delisted=False,
        first_date=first_date,
        list_date=first_date,
        last_date=_date_iso(last_date),
        note=(
            "Observed price coverage only. Membership entry and exit were left unchanged."
        ),
    )


def _run_membership_sync(
    req: MembershipSyncRequest, progress: SyncProgress | None = None
) -> dict[str, Any]:
    provider = _price_provider()
    results = []
    quota: dict[str, str] | None = None
    for symbol in req.symbols:
        if not symbol.strip():
            continue
        try:
            results.append(_resolve_membership_dates(symbol, req.expected_start, provider))
        except QuotaExceededError as exc:
            quota = {"provider": exc.provider, "scope": exc.scope, "message": str(exc)}
            break
        if progress is not None:
            progress.advance(symbol)
    payload: dict[str, Any] = {
        "expected_start": req.expected_start,
        "results": [_model_dump(result) for result in results],
    }
    if quota is not None:
        payload["termination_reason"] = "quota_exceeded"
        payload["quota"] = quota
    return payload


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
    atomic_write_text(target, json.dumps(payload, indent=2, sort_keys=True))
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
    with _universe_lifecycle_lock:
        if bundled_snapshot_name(spec.name) == spec.name:
            raise HTTPException(
                status_code=400,
                detail="bundled snapshot universe ids are reserved and immutable",
            )
        try:
            universe = _universe_from_spec(spec)
            universe.save()
        except (TypeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        _universes[spec.name] = universe
        return universe


def _migrate_reserved_default_universe() -> None:
    """Move a legacy user ``sp500-lite`` file away from the now-reserved sample id.

    Older releases allowed a persisted custom file to shadow the bundled demonstration
    universe. Preserve that file byte-for-byte under the first free legacy id. A move makes
    the migration idempotent: subsequent loads see no reserved source and create no duplicate.
    """
    with _universe_lifecycle_lock:
        universe_dir = paths.universe_dir()
        source = universe_dir / f"{_DEFAULT_UNIVERSE}.parquet"
        # The reserved id must never survive in memory, including when an app process is upgraded
        # in place after an older version already loaded the custom file.
        _universes.pop(_DEFAULT_UNIVERSE, None)
        if not source.exists():
            return

        stem = f"{_DEFAULT_UNIVERSE}-legacy"

        def heal_linked_destinations() -> bool:
            """Finish or de-duplicate an interrupted/concurrent hard-link move."""
            linked: list[Path] = []
            for candidate in universe_dir.glob(f"{stem}*.parquet"):
                try:
                    if source.samefile(candidate):
                        linked.append(candidate)
                except OSError:
                    continue
            if not linked:
                return False

            def destination_order(candidate: Path) -> tuple[int, int, str]:
                if candidate.stem == stem:
                    return (0, 0, candidate.name)
                suffix_text = candidate.stem.removeprefix(f"{stem}-")
                return (
                    1,
                    int(suffix_text) if suffix_text.isdigit() else 2**31 - 1,
                    candidate.name,
                )

            linked.sort(key=destination_order)
            # Keep the first canonical name. Multiple same-inode destinations can exist if two
            # processes linked before either removed the source; they contain no distinct data.
            for duplicate in linked[1:]:
                try:
                    duplicate.unlink()
                except FileNotFoundError:
                    pass
                except OSError:
                    return True  # preserve the source so a later load can finish safely
                _universes.pop(duplicate.stem, None)
            try:
                source.unlink()
            except FileNotFoundError:
                pass
            except OSError:
                return True
            _universes.pop(linked[0].stem, None)
            return True

        if heal_linked_destinations():
            return

        suffix = 0
        while True:
            destination_stem = stem if suffix == 0 else f"{stem}-{suffix}"
            destination = universe_dir / f"{destination_stem}.parquet"
            if destination.exists():
                # A crash after the link but before source cleanup can leave several names for the
                # same inode. Scan all candidates before advancing so a concurrent migration cannot
                # turn that state into an additional ``legacy-N`` file.
                if heal_linked_destinations():
                    return
                suffix += 1
                continue
            try:
                # Hard-link creation is atomic and never overwrites an existing destination.
                # This closes the check/replace race across app processes on the same cache
                # volume; unlinking the reserved name then completes the move byte-for-byte.
                os.link(source, destination)
            except FileExistsError:
                # Another process may have won this exact hard-link race after our exists()
                # check. Recheck inode identity before trying a second destination.
                if heal_linked_destinations():
                    return
                suffix += 1
                continue
            except OSError:
                # Preserve the reserved source for a later retry if the filesystem cannot link.
                return
            # Centralized healing also handles the unlikely case where another process linked a
            # second destination between our link and cleanup.
            heal_linked_destinations()
            return


def _load_persisted_universes() -> None:
    with _universe_lifecycle_lock:
        universe_dir = paths.universe_dir()
        if not universe_dir.exists():
            _universes.pop(_DEFAULT_UNIVERSE, None)
            return
        _migrate_reserved_default_universe()
        for source in universe_dir.glob("*.parquet"):
            if source.stem == _DEFAULT_UNIVERSE:
                continue
            if source.stem in _universes:
                continue
            try:
                _universes[source.stem] = Universe.load(source.stem, source)
            except Exception:
                continue


def _resolve_universe(name: str, *, status_code: int = 400) -> Universe:
    """Resolve bundled and custom universes through one non-fallback path."""
    _load_persisted_universes()
    if name == _DEFAULT_UNIVERSE:
        return sample_universe(name)
    universe = _universes.get(name)
    if universe is not None:
        return universe
    canonical = bundled_snapshot_name(name)
    if canonical is not None:
        return _universes.get(canonical) or bundled_universe(canonical)
    raise HTTPException(status_code=status_code, detail=f"unknown universe {name!r}")


def _universe_source(universe: Universe) -> str:
    if universe.name == _DEFAULT_UNIVERSE:
        return "sample"
    if _universes.get(universe.name) is universe:
        return "custom"
    return "bundled"


def _universe_definition_pin(name: str) -> dict[str, Any]:
    """Immutable definition metadata captured when work is submitted."""
    universe = _resolve_universe(name)
    return {
        "name": universe.name,
        "requested_name": name,
        "source": _universe_source(universe),
        "mode": universe.mode,
        "definition": dict(universe.definition),
        "fingerprint": universe.fingerprint,
        "provenance": dict(universe.provenance),
        "aliases": dict(universe.aliases),
    }


def _as_of_timestamp(value: str) -> pd.Timestamp:
    timestamp = _request_timestamp(value, label="as_of")
    if timestamp > pd.Timestamp(_today_iso()):
        raise HTTPException(status_code=400, detail="as_of date must not be in the future")
    return timestamp


_CACHE_EDGE_TOLERANCE = pd.Timedelta(days=10)


def _cache_coverage_for(
    universe: Universe,
    cutoff: pd.Timestamp,
    *,
    cache: ParquetCache | None = None,
) -> dict[str, Any]:
    """Describe whether cached prices span every declared membership interval.

    A Parquet file by itself is not sufficient evidence of coverage: a late-starting file can
    silently erase early constituents, while a stale active file shrinks the current
    cross-section. A small calendar tolerance accounts for weekends, holidays, and provider lag.
    """
    store = cache or ParquetCache()
    memberships = (
        list(universe.memberships)
        if universe.mode == "static_snapshot"
        else [membership for membership in universe.memberships if membership.entry <= cutoff]
    )
    by_symbol: dict[str, list[Membership]] = {}
    for membership in memberships:
        by_symbol.setdefault(membership.symbol, []).append(membership)

    eligible = sorted(by_symbol)
    active_symbols: list[str] = []
    exited_symbols: list[str] = []
    cached: list[str] = []
    missing: list[str] = []
    invalid: list[str] = []
    uncovered: set[str] = set()
    late_start: set[str] = set()
    stale: set[str] = set()
    symbol_coverage: dict[str, dict[str, Any]] = {}

    for symbol in eligible:
        symbol_memberships = by_symbol[symbol]
        if universe.mode == "static_snapshot":
            required_start_for_symbol = None
            required_end_for_symbol = cutoff
            membership_status = "current_snapshot"
        else:
            required_start_for_symbol = min(item.entry for item in symbol_memberships)
            interval_ends = [
                min(cutoff, item.exit - pd.Timedelta(days=1))
                if item.exit is not None
                else cutoff
                for item in symbol_memberships
            ]
            required_end_for_symbol = max(interval_ends)
            membership_status = (
                "active"
                if any(item.exit is None or item.exit > cutoff for item in symbol_memberships)
                else "exited"
            )
        if membership_status == "exited":
            exited_symbols.append(symbol)
        else:
            active_symbols.append(symbol)
        required = {
            "required_start": (
                required_start_for_symbol.date().isoformat()
                if required_start_for_symbol is not None
                else None
            ),
            "required_end": required_end_for_symbol.date().isoformat(),
            "membership_status": membership_status,
        }
        if not store.has(symbol):
            missing.append(symbol)
            symbol_coverage[symbol] = {
                "first_date": None,
                "last_date": None,
                "issues": ["missing cache file"],
                **required,
            }
            continue
        cached.append(symbol)
        try:
            frame = store.load(symbol)
        except Exception:  # noqa: BLE001 - corrupt/unsupported Parquet is a coverage result
            invalid.append(symbol)
            symbol_coverage[symbol] = {
                "first_date": None,
                "last_date": None,
                "issues": ["invalid cache file"],
                **required,
            }
            continue

        dates = pd.DatetimeIndex(frame.index)
        dates = dates[dates <= cutoff]
        issues: set[str] = set()
        for membership in symbol_memberships:
            interval_end = cutoff
            if universe.mode == "static_snapshot":
                observed = dates[dates <= interval_end]
                if len(observed) == 0:
                    uncovered.add(symbol)
                    issues.add("no observations on or before the requested date")
                elif observed.max() < interval_end - _CACHE_EDGE_TOLERANCE:
                    stale.add(symbol)
                    issues.add("history ends before the requested date")
                continue
            if membership.exit is not None:
                interval_end = min(interval_end, membership.exit - pd.Timedelta(days=1))
            if interval_end < membership.entry:
                continue
            observed = dates[(dates >= membership.entry) & (dates <= interval_end)]
            if len(observed) == 0:
                uncovered.add(symbol)
                issues.add("no observations during a membership interval")
                continue
            if observed.min() > membership.entry + _CACHE_EDGE_TOLERANCE:
                late_start.add(symbol)
                issues.add("history starts after the membership entry")
            if observed.max() < interval_end - _CACHE_EDGE_TOLERANCE:
                stale.add(symbol)
                issues.add(
                    "history ends before the declared membership exit"
                    if membership.exit is not None and membership.exit <= cutoff
                    else "history ends before the active membership interval"
                )

        symbol_coverage[symbol] = {
            "first_date": dates.min().date().isoformat() if len(dates) else None,
            "last_date": dates.max().date().isoformat() if len(dates) else None,
            "issues": sorted(issues),
            **required,
        }

    incomplete = sorted(set(missing) | set(invalid) | uncovered | late_start | stale)
    required_start = (
        min((membership.entry for membership in memberships), default=None)
        if universe.mode == "point_in_time"
        else None
    )
    required_end = max(
        (
            cutoff
            if universe.mode == "static_snapshot" or membership.exit is None
            else min(cutoff, membership.exit - pd.Timedelta(days=1))
            for membership in memberships
        ),
        default=cutoff,
    )
    return {
        "as_of": cutoff.date().isoformat(),
        "required_start": required_start.date().isoformat() if required_start is not None else None,
        "required_end": required_end.date().isoformat(),
        "eligible_symbols": eligible,
        "active_symbols": sorted(active_symbols),
        "exited_symbols": sorted(exited_symbols),
        "cached_symbols": sorted(cached),
        "missing_symbols": sorted(missing),
        "invalid_symbols": sorted(invalid),
        "uncovered_symbols": sorted(uncovered),
        "late_start_symbols": sorted(late_start),
        "stale_symbols": sorted(stale),
        "incomplete_symbols": incomplete,
        "symbol_coverage": symbol_coverage,
        "complete": bool(eligible) and not incomplete,
    }


def _cache_coverage(universe: Universe, as_of: str) -> dict[str, Any]:
    cutoff = _as_of_timestamp(as_of)
    return _cache_coverage_for(universe, cutoff)


def _universe_api_payload(
    universe: Universe,
    *,
    source: str,
    summary: bool = False,
) -> dict[str, Any]:
    preset = next(
        (
            item
            for item in universe_presets()
            if item["id"] == universe.name or item.get("snapshot_universe") == universe.name
        ),
        None,
    )
    display_name = (
        str(preset["display_name"])
        if preset is not None and source in {"sample", "bundled"}
        else universe.name
    )
    integrity = universe_integrity(universe.name, source=source)
    provenance = universe.provenance or {
        "provider": "User-supplied",
        "source_url": None,
        "retrieved_at": None,
        "note": "Saved local point-in-time membership intervals.",
    }
    base = {
        "name": universe.name,
        "display_name": display_name,
        "source": source,
        "mode": universe.mode,
        "definition": dict(universe.definition),
        "fingerprint": universe.fingerprint,
        "provenance": provenance,
        "aliases": dict(universe.aliases),
        "membership_count": len(universe.memberships),
        "symbol_count": len(universe.all_symbols()),
        "integrity": integrity,
    }
    if summary:
        issues = []
        if universe.mode == "static_snapshot":
            issues.append("Static snapshot is survivorship-biased outside its snapshot date.")
        issues.append("Open universe details to validate price-date coverage.")
        return {
            **base,
            "readiness": {
                "membership_ready": True,
                "price_ready": None,
                "training_ready": None,
                "research_ready": False,
                "coverage_checked": False,
                "issues": issues,
            },
        }

    coverage = _cache_coverage(universe, _DEFAULT_AS_OF)
    issues = []
    if universe.mode == "static_snapshot":
        issues.append("Static snapshot is survivorship-biased outside its snapshot date.")
    if not coverage["complete"]:
        issues.append("Price cache is missing or incomplete for one or more symbols.")
    research_ready = bool(integrity.get("research_ready")) and coverage["complete"]
    return {
        **_universe_to_spec(universe),
        **base,
        "symbols": universe.all_symbols(),
        "cache_coverage": coverage,
        "readiness": {
            "membership_ready": True,
            "price_ready": coverage["complete"],
            "training_ready": coverage["complete"],
            "research_ready": research_ready,
            "coverage_checked": True,
            "issues": issues,
        },
    }


def _panel_from_universe_cache(universe: Universe, as_of: pd.Timestamp) -> Panel:
    """Load every cached historical member and reject silent constituent omissions."""
    candidates = universe.members_through(as_of)
    if not candidates:
        raise HTTPException(
            status_code=400,
            detail=(
                f"universe {universe.name!r} has no memberships on or before "
                f"{as_of.date().isoformat()}"
            ),
        )
    cache = ParquetCache()
    coverage = _cache_coverage_for(universe, as_of, cache=cache)
    cached = coverage["cached_symbols"]
    missing_candidates = coverage["missing_symbols"]
    if missing_candidates:
        preview = ", ".join(missing_candidates[:8])
        suffix = "..." if len(missing_candidates) > 8 else ""
        raise HTTPException(
            status_code=400,
            detail=(
                f"universe {universe.name!r} is missing cached price history for {preview}{suffix}"
            ),
        )
    incomplete = coverage["incomplete_symbols"]
    if incomplete:
        preview = ", ".join(incomplete[:8])
        suffix = "..." if len(incomplete) > 8 else ""
        raise HTTPException(
            status_code=400,
            detail=(
                f"universe {universe.name!r} has incomplete membership-period price coverage "
                f"for {preview}{suffix}; sync full history or correct its entry/exit dates"
            ),
        )
    try:
        panel = Panel.from_cache(cached, cache=cache, end=as_of.date().isoformat())
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if len(panel.dates) == 0:
        raise HTTPException(
            status_code=400,
            detail=f"universe {universe.name!r} has no cached dates through {as_of.date()}",
        )
    start = pd.Timestamp(panel.dates.min()).normalize()
    required = set(universe.members_overlapping(start, as_of))
    missing = sorted(required - set(cached))
    if missing:
        preview = ", ".join(missing[:8])
        suffix = "..." if len(missing) > 8 else ""
        raise HTTPException(
            status_code=400,
            detail=(
                f"universe {universe.name!r} is missing cached price history for {preview}{suffix}"
            ),
        )
    return panel


def _mask_universe_panel(
    universe: Universe, panel: Panel, as_of: pd.Timestamp, *, validate_coverage: bool = True
) -> Panel:
    """Cap dates and mask every field outside each symbol's membership intervals."""
    dates = pd.DatetimeIndex(panel.dates)
    keep_dates = dates <= as_of
    if not keep_dates.any():
        raise HTTPException(
            status_code=400,
            detail=f"panel has no dates on or before {as_of.date().isoformat()}",
        )
    capped = {name: frame.loc[keep_dates] for name, frame in panel.fields.items()}
    dates = pd.DatetimeIndex(capped["close"].index)
    required = universe.members_overlapping(dates.min(), as_of)
    columns = [symbol for symbol in required if symbol in capped["close"].columns]

    # Tests and embedders may override ``get_panel`` with an abstract synthetic panel. The
    # production dependency is validated against cache before reaching this branch.
    if not columns and not validate_coverage:
        return Panel(capped)
    if not columns:
        raise HTTPException(
            status_code=400,
            detail=f"universe {universe.name!r} has no cached members in the research range",
        )

    missing = sorted(set(required) - set(columns))
    if validate_coverage and missing:
        raise HTTPException(
            status_code=400,
            detail=f"universe {universe.name!r} is missing panel columns: {', '.join(missing)}",
        )
    membership = universe.membership_mask(dates, columns)
    masked = {name: frame.loc[:, columns].where(membership) for name, frame in capped.items()}
    # ``Panel.from_cache`` derives returns before masking. Recompute so the first active date
    # cannot use a pre-membership close (and likewise after a re-entry).
    masked["returns"] = masked["close"].pct_change(fill_method=None)

    usable = [symbol for symbol in columns if masked["close"][symbol].notna().any()]
    uncovered = sorted(set(required) - set(usable))
    if validate_coverage and uncovered:
        raise HTTPException(
            status_code=400,
            detail=(
                f"universe {universe.name!r} has no in-membership price observations for "
                f"{', '.join(uncovered)}"
            ),
        )
    masked = {name: frame.loc[:, usable] for name, frame in masked.items()}
    active_dates = masked["close"].notna().any(axis=1)
    if not usable or not active_dates.any():
        raise HTTPException(
            status_code=400,
            detail=f"universe {universe.name!r} produced an empty point-in-time panel",
        )
    return Panel({name: frame.loc[active_dates] for name, frame in masked.items()})


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


def get_panel() -> None:
    """Lazy dependency marker; the request's selected universe determines what is loaded."""
    return None


def _factor_store() -> FactorStore:
    """A factor store rooted at the currently configured directory (resolved fresh)."""
    return FactorStore(paths.factors_dir())


def _panel_for_universe(universe: str, as_of: str, default_panel: Panel | None) -> Panel:
    """Resolve, cap, and date-mask a universe without a silent unknown-name fallback."""
    resolved = _resolve_universe(universe)
    cutoff = _as_of_timestamp(as_of)
    if default_panel is not None and universe == _DEFAULT_UNIVERSE:
        # A no-overlap dependency panel is an intentional test/embed override. Production's
        # ``get_panel`` loads and validates this universe's actual cache first.
        overlap = set(resolved.members_through(cutoff)) & set(default_panel.symbols)
        return _mask_universe_panel(
            resolved, default_panel, cutoff, validate_coverage=bool(overlap)
        )

    # An injected panel can also stand in for a custom universe when it contains all members;
    # otherwise use the real cache and enforce complete historical-member coverage.
    if default_panel is not None:
        capped_dates = pd.DatetimeIndex(default_panel.dates)
        capped_dates = capped_dates[capped_dates <= cutoff]
        if len(capped_dates):
            required = set(resolved.members_overlapping(capped_dates.min(), cutoff))
            if required and required <= set(default_panel.symbols):
                return _mask_universe_panel(resolved, default_panel, cutoff)
    source = _panel_from_universe_cache(resolved, cutoff)
    return _mask_universe_panel(resolved, source, cutoff)


def _load_seed_factors(
    factor_ids: list[str],
) -> tuple[list[Any], int, int, list[dict[str, Any]]]:
    """Resolve saved factors into seed trees, re-registering their operators (P4).

    Returns trees, trial/test baselines, and non-secret evidence provenance.
    """
    store = _factor_store()
    seeds: list[Any] = []
    trial_baseline = 0
    test_reads_baseline = 0
    evidence_sources: list[dict[str, Any]] = []
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
                    policy=spec.get("policy"),
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
            evidence_sources.append(
                {
                    "factor_id": factor_id,
                    "session_id": provenance.get("session_id"),
                    "holdout_fingerprint": provenance.get("holdout_fingerprint"),
                    "test_reads": int(provenance.get("test_reads", 0) or 0),
                }
            )
    return seeds, trial_baseline, test_reads_baseline, evidence_sources


def _gp_config_from_request(data: dict[str, Any]) -> GPConfig:
    """Parse untrusted config data and consistently expose mistakes as client errors."""
    try:
        return GPConfig.from_dict(data)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"invalid GP config: {exc}") from exc


def _validate_panel_config(config: GPConfig, panel: Panel) -> None:
    if config.min_names > len(panel.symbols):
        raise HTTPException(
            status_code=400,
            detail=(
                f"min_names ({config.min_names}) exceeds the panel's symbol count "
                f"({len(panel.symbols)})"
            ),
        )


def _preflight_split(
    config: GPConfig,
    panel: Panel,
    *,
    train: float = 0.6,
    valid: float = 0.2,
    embargo: int = 5,
) -> None:
    _validate_panel_config(config, panel)
    try:
        time_split(
            panel.dates,
            train=train,
            valid=valid,
            embargo=embargo,
            horizon=label_span(config.horizon, config.execution),
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"invalid time split: {exc}") from exc


def _session_name(req: SessionCreateRequest) -> str:
    name = req.name.strip()
    if not name:
        raise HTTPException(status_code=400, detail="session name must not be blank")
    if req.train + req.valid >= 1.0:
        raise HTTPException(status_code=400, detail="train + valid must be less than 1")
    if any(not factor_id.strip() for factor_id in req.seed_factor_ids):
        raise HTTPException(status_code=400, detail="seed factor ids must not be blank")
    if len(set(req.seed_factor_ids)) != len(req.seed_factor_ids):
        raise HTTPException(status_code=400, detail="seed factor ids must not contain duplicates")
    return name


class FormulaTestProgress:
    """Small thread-safe phase tracker used by exploratory formula-test jobs."""

    def __init__(self) -> None:
        self._phase = "queued"
        self._lock = threading.Lock()

    def set_phase(self, phase: str) -> None:
        with self._lock:
            self._phase = phase

    def finish(self, reason: str) -> None:
        with self._lock:
            self._phase = "stopped" if reason == "user_stopped" else "done"

    def snapshot(self) -> dict[str, str]:
        with self._lock:
            return {"phase": self._phase}


def _substitute_formula_args(body: Node, arguments: list[Node]) -> Node:
    if body.name == "$arg":
        if isinstance(body.value, bool) or not isinstance(body.value, int):
            raise HTTPException(status_code=400, detail="$arg index must be an integer")
        if body.value < 0 or body.value >= len(arguments):
            raise HTTPException(
                status_code=400,
                detail=f"$arg index {body.value} out of range for {len(arguments)} bindings",
            )
        if body.children:
            raise HTTPException(status_code=400, detail="$arg placeholder must have no children")
        return arguments[body.value]
    return Node(
        body.name,
        tuple(_substitute_formula_args(child, arguments) for child in body.children),
        body.value,
    )


def _validate_formula_expansion(spec: FormulaSpec) -> None:
    """Apply the same expanded-expression safety ceiling used by tests and training seeds."""
    arguments: list[Node] = []
    for item in spec.inputs:
        input_type = DType(item.type)
        if input_type in {DType.SERIES, DType.SIGNAL}:
            arguments.append(Node("close"))
        elif input_type is DType.BOOL:
            arguments.append(Node("gt", (Node("close"), Node("close"))))
        elif input_type is DType.WINDOW:
            arguments.append(Node("window", value=int(item.default or 1)))
        else:
            arguments.append(Node("const", value=float(item.default or 0.0)))
    try:
        body = _substitute_formula_args(tree_from_dict(spec.body), arguments)
        expand_all(body)
    except (AttributeError, InvalidOperator, KeyError, TypeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=f"invalid formula expansion: {exc}") from exc


def _formula_dependency_revisions(tree: Node) -> list[dict[str, Any]]:
    by_runtime = {spec.runtime_name: spec for spec in _all_formula_specs()}
    seen: set[str] = set()
    active: list[str] = []
    ordered: list[dict[str, Any]] = []

    def scan(node: Node) -> None:
        for item in node.iter_nodes():
            spec = by_runtime.get(item.name)
            if spec is None or spec.runtime_name in seen:
                continue
            if spec.runtime_name in active:
                start = active.index(spec.runtime_name)
                cycle = [*active[start:], spec.runtime_name]
                raise HTTPException(
                    status_code=400,
                    detail=f"formula dependency cycle: {' -> '.join(cycle)}",
                )
            active.append(spec.runtime_name)
            scan(tree_from_dict(spec.body))
            active.pop()
            seen.add(spec.runtime_name)
            ordered.append(
                {
                    "name": spec.name,
                    "revision": spec.revision,
                    "runtime_name": spec.runtime_name,
                }
            )

    scan(tree)
    return ordered


def _result_binding_tree(result_id: str) -> Node:
    try:
        result = _factor_store().get(result_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if result is None:
        raise HTTPException(status_code=400, detail=f"unknown formula result {result_id!r}")
    return result.expanded_tree or result.tree


def _formula_binding_node(
    binding: FormulaBinding,
    expected: DType,
    *,
    input_name: str,
) -> Node:
    if binding.kind == "field":
        field = (binding.field or "").strip()
        prim = REGISTRY.get(field)
        if prim is None or prim.kind.value != "operand":
            raise HTTPException(
                status_code=400, detail=f"unknown market field {field!r} for {input_name!r}"
            )
        node = Node(field)
    elif binding.kind == "formula":
        runtime_name = (binding.runtime_name or "").strip()
        spec = _formula_for_runtime(runtime_name)
        if spec is None:
            raise HTTPException(
                status_code=400, detail=f"unknown formula revision {runtime_name!r}"
            )
        if spec.arg_types:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"formula revision {runtime_name!r} still has exposed inputs; "
                    "bind it in a concrete wrapper first"
                ),
            )
        if runtime_name not in USER_OPERATORS:
            raise HTTPException(
                status_code=400, detail=f"formula revision {runtime_name!r} is unavailable"
            )
        node = Node(runtime_name)
    elif binding.kind == "result":
        result_id = (binding.result_id or "").strip()
        if not result_id:
            raise HTTPException(status_code=400, detail=f"result binding {input_name!r} is empty")
        node = _result_binding_tree(result_id)
    else:
        value = binding.value
        if expected is DType.WINDOW:
            if isinstance(value, bool) or not isinstance(value, int):
                raise HTTPException(
                    status_code=400,
                    detail=f"window binding {input_name!r} must be an integer",
                )
            node = Node("window", value=value)
        elif expected is DType.SCALAR:
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
            ):
                raise HTTPException(
                    status_code=400,
                    detail=f"scalar binding {input_name!r} must be finite",
                )
            node = Node("const", value=float(value))
        else:
            raise HTTPException(
                status_code=400,
                detail=f"literal binding cannot satisfy {expected.value} input {input_name!r}",
            )

    try:
        validate_tree(node)
        actual = node.out_type
    except Exception as exc:  # noqa: BLE001 - malformed persisted results are client-visible
        raise HTTPException(
            status_code=400, detail=f"invalid binding for {input_name!r}: {exc}"
        ) from exc
    if not is_subtype(actual, expected):
        raise HTTPException(
            status_code=400,
            detail=(
                f"binding {input_name!r} produces {actual.value}, "
                f"but the input expects {expected.value}"
            ),
        )
    return node


def _resolve_formula_test_tree(
    req: FormulaTestRequest,
) -> tuple[Node, dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    loaded = _load_persisted_formulas()
    unavailable = {
        str(item.get("runtime_name")): str(item.get("error"))
        for item in loaded
        if not item.get("registered")
    }
    source = req.source
    if source.kind == "saved":
        runtime_name = (source.runtime_name or "").strip()
        spec = _formula_for_runtime(runtime_name)
        if spec is None:
            raise HTTPException(
                status_code=400, detail=f"unknown formula revision {runtime_name!r}"
            )
        if runtime_name in unavailable or runtime_name not in USER_OPERATORS:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"formula revision {runtime_name!r} is unavailable: "
                    f"{unavailable.get(runtime_name, '')}"
                ),
            )
        inputs = list(spec.inputs)
        out_type = DType(spec.out_type)
        body = None
        source_payload = {
            "kind": "saved",
            "name": spec.name,
            "revision": spec.revision,
            "runtime_name": spec.runtime_name,
        }
    else:
        if source.body is None or source.out_type is None:
            raise HTTPException(
                status_code=400, detail="draft source requires body, inputs, and out_type"
            )
        try:
            input_types = [DType(item.type) for item in source.inputs]
            out_type = DType(source.out_type)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=f"unknown formula type: {exc}") from exc
        inputs = _formula_inputs([item.value for item in input_types], list(source.inputs))
        try:
            body = tree_from_dict(source.body)
            inferred = infer_macro_type(body, input_types)
        except (InvalidOperator, KeyError, TypeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=f"invalid draft formula: {exc}") from exc
        if not is_subtype(inferred, out_type):
            raise HTTPException(
                status_code=400,
                detail=(
                    f"draft body produces {inferred.value}, not the declared {out_type.value}"
                ),
            )
        source_payload = {
            "kind": "draft",
            "body": source.body,
            "inputs": [_model_dump(item) for item in inputs],
            "out_type": out_type.value,
        }

    if out_type not in {DType.SERIES, DType.SIGNAL}:
        raise HTTPException(
            status_code=400,
            detail=f"formula tests require a series or signal output, got {out_type.value}",
        )
    expected_names = {item.name for item in inputs}
    supplied_names = set(req.bindings)
    missing = sorted(expected_names - supplied_names)
    unexpected = sorted(supplied_names - expected_names)
    if missing or unexpected:
        detail = []
        if missing:
            detail.append(f"missing bindings: {', '.join(missing)}")
        if unexpected:
            detail.append(f"unexpected bindings: {', '.join(unexpected)}")
        raise HTTPException(status_code=400, detail="; ".join(detail))

    arguments = [
        _formula_binding_node(req.bindings[item.name], DType(item.type), input_name=item.name)
        for item in inputs
    ]
    if source.kind == "saved":
        assert source_payload.get("runtime_name")
        bound = Node(str(source_payload["runtime_name"]), tuple(arguments))
    else:
        assert body is not None
        bound = _substitute_formula_args(body, arguments)
    try:
        validate_tree(bound)
        dependencies = _formula_dependency_revisions(bound)
        expanded = expand_all(bound)
        validate_tree(expanded)
    except (InvalidOperator, KeyError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=f"invalid bound formula: {exc}") from exc
    binding_payload = {name: _model_dump(binding) for name, binding in req.bindings.items()}
    return expanded, source_payload, binding_payload, dependencies


def _slice_formula_test_panel(
    req: FormulaTestRequest,
    default_panel: Panel | None,
) -> tuple[Panel, pd.DatetimeIndex]:
    """Return the full warm-up panel through ``end`` plus the requested report dates."""
    start = _request_timestamp(req.start, label="start") if req.start else None
    end = _request_timestamp(req.end, label="end") if req.end else None
    if start is not None and end is not None and start > end:
        raise HTTPException(status_code=400, detail="start date must not be after end date")
    cutoff = (end or pd.Timestamp(_DEFAULT_AS_OF)).date().isoformat()
    panel = _panel_for_universe(req.universe, cutoff, default_panel)
    dates = pd.DatetimeIndex(panel.dates)
    warmup_keep = np.ones(len(dates), dtype=bool)
    if end is not None:
        warmup_keep &= np.asarray(dates <= end)
    if not warmup_keep.any():
        raise HTTPException(status_code=400, detail="formula test date range has no cached data")
    warmup_panel = Panel(
        {name: frame.loc[warmup_keep] for name, frame in panel.fields.items()}
    )
    report_dates = pd.DatetimeIndex(warmup_panel.dates)
    report_keep = np.ones(len(report_dates), dtype=bool)
    if start is not None:
        report_keep &= np.asarray(report_dates >= start)
    if end is not None:
        report_keep &= np.asarray(report_dates <= end)
    report_dates = report_dates[report_keep]
    if report_dates.empty:
        raise HTTPException(status_code=400, detail="formula test date range has no cached data")
    if len(warmup_panel.dates) <= label_span(req.horizon, req.execution):
        raise HTTPException(
            status_code=400,
            detail="formula test date range is shorter than the forward-return horizon",
        )
    return warmup_panel, report_dates


def _formula_test_data_metadata(
    panel: Panel,
    universe: str,
    report_dates: pd.DatetimeIndex,
) -> tuple[dict[str, Any], str]:
    close = panel["close"]
    report_close = close.reindex(report_dates)
    values = report_close.to_numpy(dtype="float64", na_value=np.nan)
    missing = float(np.isnan(values).mean()) if values.size else 1.0
    coverage = {
        "universe": universe,
        "start": pd.Timestamp(report_dates.min()).date().isoformat(),
        "end": pd.Timestamp(report_dates.max()).date().isoformat(),
        "dates": len(report_dates),
        "symbols": len(panel.symbols),
        "missing_fraction": missing,
        "needs_sync": bool(missing > 0.0),
        "fields": sorted(panel.fields),
        "warmup_start": pd.Timestamp(panel.dates.min()).date().isoformat(),
        "warmup_observations": len(panel.dates),
    }
    coverage.update(
        {
            "first_date": coverage["start"],
            "last_date": coverage["end"],
            "observations": coverage["dates"],
            "warnings": (
                [f"{missing:.1%} of close observations are missing"] if missing > 0.0 else []
            ),
        }
    )
    digest = hashlib.sha256(
        json.dumps(coverage, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )
    digest.update(pd.util.hash_pandas_object(close, index=True).to_numpy().tobytes())
    digest.update("\x1f".join(map(str, close.columns)).encode("utf-8"))
    return coverage, digest.hexdigest()


def _formula_test_result(
    *,
    tree: Node,
    panel: Panel,
    req: FormulaTestRequest,
    source: dict[str, Any],
    bindings: dict[str, Any],
    dependencies: list[dict[str, Any]],
    report_dates: pd.DatetimeIndex,
    data_coverage: dict[str, Any],
    data_revision: str,
    universe_definition: dict[str, Any],
    progress: FormulaTestProgress,
    cancel: threading.Event,
) -> dict[str, Any]:
    if cancel.is_set():
        raise TrainingCancelled("formula test stopped")
    progress.set_phase("evaluating")
    factor = evaluate(tree, panel)
    if not isinstance(factor, pd.DataFrame):
        raise ValueError("formula did not evaluate to a panel")
    if cancel.is_set():
        raise TrainingCancelled("formula test stopped")
    progress.set_phase("backtesting")
    fwd = forward_returns(panel, req.horizon, req.execution)
    if req.strategies is not None:
        strategy_specs = [item.to_spec() for item in req.strategies]
    else:
        strategy_specs = [
            PortfolioStrategySpec(
                "primary",
                req.weighting_scheme,
                req.quantile if req.weighting_scheme == "quantile_ls" else None,
            )
        ]
    if len({item.id for item in strategy_specs}) != len(strategy_specs):
        raise ValueError("strategy ids must be unique")
    primary_strategy_id = req.primary_strategy_id or strategy_specs[0].id
    if primary_strategy_id not in {item.id for item in strategy_specs}:
        raise ValueError("primary_strategy_id must reference a strategy")
    costs = TransactionCostModel(req.commission_bps, req.slippage_bps)
    strategy_results: list[dict[str, Any]] = []
    for strategy_spec in strategy_specs:
        tested = backtest_report(
            factor,
            panel,
            fwd,
            strategy_spec.weighting_scheme(),
            costs,
            report_dates,
            horizon=req.horizon,
            execution=req.execution,
        )
        strategy_results.append(
            {
                "strategy_id": strategy_spec.id,
                "spec": strategy_spec.to_dict(),
                "role": (
                    "primary"
                    if strategy_spec.id == primary_strategy_id
                    else "comparison"
                ),
                "oos_backtest": tested,
            }
        )
    reported = next(
        item["oos_backtest"]
        for item in strategy_results
        if item["strategy_id"] == primary_strategy_id
    )
    primary_strategy = next(
        item for item in strategy_specs if item.id == primary_strategy_id
    )
    if cancel.is_set():
        raise TrainingCancelled("formula test stopped")
    # This is research-only reporting, not a locked holdout read; it remains cancellable.
    progress.set_phase("reporting")
    if cancel.is_set():
        raise TrainingCancelled("formula test stopped")
    configuration = {
        "universe": req.universe,
        "universe_definition": universe_definition,
        "start": data_coverage["start"],
        "end": data_coverage["end"],
        "horizon": req.horizon,
        "execution": req.execution,
        "weighting_scheme": primary_strategy.scheme,
        "quantile": primary_strategy.quantile,
        "commission_bps": req.commission_bps,
        "slippage_bps": req.slippage_bps,
        "primary_strategy_id": primary_strategy_id,
        "strategies": [item.to_dict() for item in strategy_specs],
    }
    return {
        "kind": "backtest",
        "tree": tree_to_dict(tree),
        "expanded_tree": tree_to_dict(tree),
        "source": source,
        "bindings": bindings,
        "dependency_revisions": [str(item["runtime_name"]) for item in dependencies],
        "expression_fingerprint": hashlib.sha256(tree_to_json(tree).encode("utf-8")).hexdigest(),
        **configuration,
        "configuration": configuration,
        "data_coverage": data_coverage,
        "data_revision": data_revision,
        "universe_definition": universe_definition,
        "metrics": reported["metrics"],
        "returns": reported["returns"],
        "normalized_equity": reported["normalized_equity"],
        "oos_backtest": reported,
        "primary_strategy_id": primary_strategy_id,
        "strategy_results": strategy_results,
        "portfolio_schema_version": PORTFOLIO_SCHEMA_VERSION,
        "disclaimer": DISCLAIMER,
        "exploratory": True,
        "termination_reason": "completed",
    }


def _formula_test_job_payload(job: Any) -> dict[str, Any]:
    return {
        "job_id": job.id,
        "status": job.status,
        "progress": job.progress.snapshot() if job.progress is not None else None,
        "result": job.result,
        "error": job.error,
        "termination_reason": job.termination_reason,
    }


# --- endpoints -------------------------------------------------------------------
@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/primitives")
def list_primitives() -> list[dict[str, Any]]:
    """The primitive palette (built-ins + user operators) for the operator composer."""
    _load_persisted_formulas()
    cats = _formula_categories()
    return [_primitive_info(p, cats) for p in REGISTRY.values()]


@app.post("/operators")
def add_operator(spec: OperatorSpec) -> dict[str, Any]:
    return _primitive_info(_register(spec))


@app.get("/operators")
def list_operators() -> list[dict[str, Any]]:
    return [_primitive_info(p) for p in USER_OPERATORS.values()]


@app.delete("/operators/{name}")
def remove_operator(name: str) -> dict[str, str]:
    formula = _formula_for_runtime(name)
    if formula is not None and not formula.editable:
        raise HTTPException(status_code=403, detail="catalog formulas cannot be removed")
    unregister_operator(name)
    return {"removed": name}


@app.get("/formulas")
def list_formulas() -> list[dict[str, Any]]:
    return _load_persisted_formulas()


@app.post("/formulas")
def add_formula(spec: FormulaSpec) -> dict[str, Any]:
    spec = FormulaSpec(
        **{
            **_model_dump(spec),
            "origin": "user_formula",
            "editable": True,
            "catalog_revision": None,
        }
    )
    spec = _normalize_formula_spec(spec, revision=1)
    store = _read_formula_store()
    if _formula_family(store, spec.name) is not None:
        raise HTTPException(status_code=400, detail="formula name already exists")
    store["families"].append(
        {"name": spec.name, "latest_revision": 1, "revisions": [_model_dump(spec)]}
    )
    try:
        _assert_latest_formula_dag(store)
    except InvalidOperator as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    _validate_formula_expansion(spec)
    _register(_formula_operator_spec(spec))
    _write_formula_store(store)
    return {**_model_dump(spec), "registered": True, "error": None}


@app.post("/formulas/validate")
def validate_formula(spec: FormulaSpec) -> dict[str, Any]:
    """Type-check a formula body without registering it or touching disk (live editor feedback)."""
    try:
        normalized = _normalize_formula_spec(spec)
        name = normalized.name
    except HTTPException as exc:
        return {"ok": False, "error": str(exc.detail), "name": None}
    try:
        store = _read_formula_store()
        family = _formula_family(store, normalized.name)
        planned = copy.deepcopy(store)
        if family is None:
            planned["families"].append(
                {
                    "name": normalized.name,
                    "latest_revision": normalized.revision,
                    "revisions": [_model_dump(normalized)],
                }
            )
        else:
            current = _family_latest(family)
            candidate = _normalize_formula_spec(
                normalized,
                revision=current.revision + 1,
                runtime_name=f"{current.name}__r{current.revision + 1}",
                created_at=current.created_at,
            )
            planned_family = _formula_family(planned, normalized.name)
            assert planned_family is not None
            planned_family["revisions"].append(_model_dump(candidate))
            planned_family["latest_revision"] = candidate.revision
        _assert_latest_formula_dag(planned)
        out = infer_macro_type(
            tree_from_dict(normalized.body), [DType(t) for t in normalized.arg_types]
        )
        _validate_formula_expansion(normalized)
    except (HTTPException, InvalidOperator, ValueError, KeyError) as exc:
        error = exc.detail if isinstance(exc, HTTPException) else str(exc)
        return {"ok": False, "error": str(error), "name": name}
    except (AttributeError, TypeError) as exc:
        return {"ok": False, "error": str(exc), "name": name}
    declared = DType(normalized.out_type)
    if not is_subtype(out, declared):
        return {
            "ok": False,
            "error": f"body produces {out.value}, not the declared output {declared.value}",
            "name": name,
        }
    return {"ok": True, "out_type": out.value, "name": name, "error": None}


@app.get("/formulas/{name}")
def get_formula(name: str) -> dict[str, Any]:
    normalized = _normalize_formula_name(name)
    store = _read_formula_store()
    family = _formula_family(store, normalized)
    if family is None:
        raise HTTPException(status_code=404, detail="unknown formula")
    latest = _family_latest(family)
    return {
        **_model_dump(latest),
        "revisions": family["revisions"],
        "impact": _formula_impact(normalized),
    }


@app.post("/formulas/{name}/impact")
def formula_impact(name: str, spec: FormulaSpec) -> dict[str, Any]:
    normalized = _normalize_formula_name(name)
    return _formula_impact(normalized, spec)


@app.put("/formulas/{name}")
def update_formula(name: str, spec: FormulaSpec, strategy: str = "update") -> dict[str, Any]:
    normalized = _normalize_formula_name(name)
    store = _read_formula_store()
    family = _formula_family(store, normalized)
    if family is None:
        raise HTTPException(status_code=404, detail="unknown formula")
    existing = _family_latest(family)
    if not existing.editable:
        raise HTTPException(
            status_code=403,
            detail="catalog formulas are immutable; copy one to edit",
        )
    requested = FormulaSpec(
        **{
            **_model_dump(spec),
            "name": normalized,
            "origin": existing.origin,
            "editable": existing.editable,
            "catalog_revision": existing.catalog_revision,
        }
    )
    updated = _normalize_formula_spec(
        requested,
        revision=existing.revision,
        runtime_name=existing.runtime_name,
        created_at=existing.created_at,
    )
    impact = _formula_impact(normalized, updated)
    try:
        out = infer_macro_type(tree_from_dict(updated.body), [DType(t) for t in updated.arg_types])
        if not is_subtype(out, DType(updated.out_type)):
            raise InvalidOperator(
                f"body produces {out.value}, not the declared output {updated.out_type}"
            )
    except (InvalidOperator, ValueError, KeyError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    _validate_formula_expansion(updated)

    if impact["change"] != "calculation":
        family["revisions"] = [
            _model_dump(updated) if int(item.get("revision", 1)) == existing.revision else item
            for item in family["revisions"]
        ]
        _write_formula_store(store)
        return {**_model_dump(updated), "registered": True, "error": None, "upgraded": []}

    if impact["has_references"] and strategy != "upgrade_references":
        raise HTTPException(status_code=400, detail={"message": "formula is in use", **impact})

    if strategy not in {"update", "upgrade_references"}:
        raise HTTPException(status_code=400, detail="unknown formula update strategy")

    if not impact["has_references"]:
        revision = existing.revision + 1
        target = _normalize_formula_spec(
            requested,
            revision=revision,
            runtime_name=f"{normalized}__r{revision}",
            created_at=existing.created_at,
        )
        planned = copy.deepcopy(store)
        planned_family = _formula_family(planned, normalized)
        assert planned_family is not None
        planned_family["revisions"].append(_model_dump(target))
        planned_family["latest_revision"] = target.revision
        try:
            _assert_latest_formula_dag(planned)
        except InvalidOperator as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        _register(_formula_operator_spec(target))
        _write_formula_store(planned)
        return {**_model_dump(target), "registered": True, "error": None, "upgraded": []}

    if strategy != "upgrade_references":
        raise HTTPException(status_code=400, detail="unknown formula update strategy")

    replacements: dict[str, str] = {}
    candidates: list[FormulaSpec] = []
    target_revision = existing.revision + 1
    target = _normalize_formula_spec(
        requested,
        revision=target_revision,
        runtime_name=f"{normalized}__r{target_revision}",
        created_at=existing.created_at,
    )
    replacements[existing.runtime_name] = target.runtime_name
    candidates.append(target)

    for dependent in _topo_sort_formulas(_read_formula_specs()):
        if dependent.name == normalized:
            continue
        if not (_tree_names(dependent.body) & replacements.keys()):
            continue
        revision = dependent.revision + 1
        candidate = _normalize_formula_spec(
            FormulaSpec(
                **{
                    **_model_dump(dependent),
                    "body": _replace_tree_names(dependent.body, replacements),
                }
            ),
            revision=revision,
            runtime_name=f"{dependent.name}__r{revision}",
            created_at=dependent.created_at,
        )
        replacements[dependent.runtime_name] = candidate.runtime_name
        candidates.append(candidate)

    planned = copy.deepcopy(store)
    for candidate in candidates:
        candidate_family = _formula_family(planned, candidate.name)
        if candidate_family is None:
            continue
        candidate_family["revisions"].append(_model_dump(candidate))
        candidate_family["latest_revision"] = candidate.revision
    try:
        _assert_latest_formula_dag(planned)
    except InvalidOperator as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    registered: list[str] = []
    try:
        for candidate in candidates:
            _register(_formula_operator_spec(candidate))
            registered.append(candidate.runtime_name)
    except HTTPException:
        for runtime_name in registered:
            unregister_operator(runtime_name)
        raise

    _write_formula_store(planned)
    return {
        **_model_dump(target),
        "registered": True,
        "error": None,
        "upgraded": [item.name for item in candidates[1:]],
    }


@app.delete("/formulas/{name}")
def delete_formula(name: str) -> dict[str, str]:
    with _formula_lifecycle_lock:
        normalized = _normalize_formula_name(name)
        store = _read_formula_store()
        family = _formula_family(store, normalized)
        if family is None:
            raise HTTPException(status_code=404, detail="unknown formula")
        if not _family_latest(family).editable:
            raise HTTPException(
                status_code=403,
                detail="catalog formulas are immutable; copy one to edit",
            )
        impact = _formula_impact(normalized)
        if impact["has_references"]:
            raise HTTPException(
                status_code=400, detail={"message": "formula is in use", **impact}
            )
        for item in family["revisions"]:
            unregister_operator(str(item.get("runtime_name") or item.get("name")))
        store["families"] = [
            item for item in store["families"] if item["name"] != normalized
        ]
        _write_formula_store(store)
    return {"removed": normalized}


def _normalized_category_order(values: object) -> list[str]:
    normalized: list[str] = []
    seen: set[str] = set()
    items = values if isinstance(values, list) else []
    for item in items:
        if not isinstance(item, str) or not item.strip():
            continue
        category = item.strip()
        if category.casefold() in _RESERVED_LEAF_CATEGORIES:
            category = category.casefold()
        if category not in seen:
            seen.add(category)
            normalized.append(category)
    for category in core_categories.DEFAULT_CATEGORY_ORDER:
        if category not in seen:
            seen.add(category)
            normalized.append(category)
    return normalized


def _category_settings() -> dict[str, Any]:
    """The persisted category list + overrides, seeded with defaults when absent."""
    # Category overrides may target persisted formula runtimes. Register them before sanitizing
    # so a fresh process does not mistake a valid formula override for an unknown primitive.
    _load_persisted_formulas()
    stored = paths.read_categories()
    order = _normalized_category_order(stored.get("order"))
    overrides = stored.get("overrides")
    if not isinstance(overrides, dict):
        overrides = {}
    sanitized = {
        primitive: category
        for primitive, category in overrides.items()
        if _category_override_allowed(primitive, category)
    }
    settings = {"order": order, "overrides": sanitized}
    # One-time migration for legacy workspaces that placed operands in arbitrary groups or put a
    # calculation under the reserved data/constant headings (the source of duplicate Data groups).
    if settings != stored:
        paths.write_categories(settings)
    return settings


@app.get("/categories")
def get_categories() -> dict[str, Any]:
    return _category_settings()


@app.put("/categories")
def put_categories(update: CategoryUpdate) -> dict[str, Any]:
    settings = _category_settings()
    if update.order is not None:
        settings["order"] = _normalized_category_order(update.order)
    if update.overrides is not None:
        invalid = [
            primitive
            for primitive, category in update.overrides.items()
            if not _category_override_allowed(primitive, category)
        ]
        if invalid:
            raise HTTPException(
                status_code=400,
                detail=(
                    "only calculation operators may be assigned to non-reserved categories: "
                    + ", ".join(sorted(invalid))
                ),
            )
        settings["overrides"] = {**settings["overrides"], **update.overrides}
    # Any category referenced by an override but missing from the order is appended.
    for category in settings["overrides"].values():
        if category not in settings["order"]:
            settings["order"].append(category)
    paths.write_categories(settings)
    return settings


@app.put("/categories/{primitive}")
def set_primitive_category(primitive: str, update: PrimitiveCategoryUpdate) -> dict[str, Any]:
    _load_persisted_formulas()
    if primitive not in REGISTRY:
        raise HTTPException(status_code=404, detail="unknown primitive")
    if not _category_override_allowed(primitive, update.category):
        raise HTTPException(
            status_code=400,
            detail="only calculation operators may be assigned to non-reserved categories",
        )
    settings = _category_settings()
    settings["overrides"][primitive] = update.category
    if update.category not in settings["order"]:
        settings["order"].append(update.category)
    paths.write_categories(settings)
    return settings


@app.get("/universes")
def list_universes(
    view: str | None = None,
    summary: bool = False,
) -> list[dict[str, Any]]:
    if view not in {None, "detail", "summary"}:
        raise HTTPException(status_code=400, detail="universe view must be 'detail' or 'summary'")
    summary_view = summary or view == "summary"
    _load_persisted_universes()
    sample = sample_universe(_DEFAULT_UNIVERSE)
    custom_names = set(_universes)
    bundled = [
        _universe_api_payload(
            bundled_universe(str(spec["id"])), source="bundled", summary=summary_view
        )
        for spec in bundled_snapshot_specs()
        if spec["id"] not in custom_names
    ]
    custom = [
        _universe_api_payload(universe, source="custom", summary=summary_view)
        for universe in sorted(_universes.values(), key=lambda u: u.name)
    ]
    items = [
        _universe_api_payload(sample, source="sample", summary=summary_view),
        *bundled,
        *custom,
    ]
    layout = universe_folders.tree(item["name"] for item in items)
    for item in items:
        folder_id = layout["placements"].get(item["name"])
        item["folder_id"] = folder_id
        item["folder_path"] = universe_folders.folder_path(layout["folders"], folder_id)
    return items


def _known_universe_names() -> list[str]:
    _load_persisted_universes()
    names = [_DEFAULT_UNIVERSE]
    names.extend(str(spec["id"]) for spec in bundled_snapshot_specs())
    names.extend(sorted(_universes))
    return list(dict.fromkeys(names))


def _folder_http_error(exc: universe_folders.FolderError) -> HTTPException:
    return HTTPException(status_code=exc.status, detail=str(exc))


@app.get("/universe-folders")
def list_universe_folders() -> dict[str, Any]:
    """Presentation folders and the folder of every known universe (``null`` = top level)."""
    return universe_folders.tree(_known_universe_names())


@app.post("/universe-folders")
def create_universe_folder(req: UniverseFolderCreate) -> dict[str, Any]:
    try:
        return universe_folders.create_folder(req.name, req.parent)
    except universe_folders.FolderError as exc:
        raise _folder_http_error(exc) from exc


@app.patch("/universe-folders/{folder_id}")
def update_universe_folder(folder_id: str, req: UniverseFolderUpdate) -> dict[str, Any]:
    fields: set[str] = set(getattr(req, "model_fields_set", set()))
    changes: dict[str, Any] = {}
    if "name" in fields and req.name is not None:
        changes["name"] = req.name
    if "parent" in fields:
        changes["parent"] = req.parent
    try:
        return universe_folders.update_folder(folder_id, **changes)
    except universe_folders.FolderError as exc:
        raise _folder_http_error(exc) from exc


@app.delete("/universe-folders/{folder_id}")
def delete_universe_folder(folder_id: str) -> dict[str, Any]:
    try:
        return universe_folders.delete_folder(folder_id, _known_universe_names())
    except universe_folders.FolderError as exc:
        raise _folder_http_error(exc) from exc


@app.put("/universe-folders/placements")
def move_universes_to_folder(req: UniversePlacementUpdate) -> dict[str, Any]:
    known = set(_known_universe_names())
    unknown = [name for name in req.universes if name not in known]
    if unknown:
        raise HTTPException(status_code=404, detail=f"unknown universe(s): {', '.join(unknown)}")
    try:
        return universe_folders.move_universes(req.universes, req.folder)
    except universe_folders.FolderError as exc:
        raise _folder_http_error(exc) from exc


@app.get("/universe-presets")
def list_universe_presets() -> list[dict[str, Any]]:
    """Common indexes expose a bundled snapshot and a separate PIT-import lifecycle."""
    _load_persisted_universes()
    installed = set(_universes)
    results: list[dict[str, Any]] = []
    for preset in universe_presets():
        item = dict(preset)
        snapshot_name = str(item.get("snapshot_universe") or "")
        if snapshot_name:
            if snapshot_name == _DEFAULT_UNIVERSE:
                snapshot = sample_universe(snapshot_name)
                snapshot_source = "sample"
            else:
                snapshot = bundled_universe(snapshot_name)
                snapshot_source = "bundled"
            snapshot_payload = _universe_api_payload(
                snapshot,
                source=snapshot_source,
                summary=True,
            )
            item.update(
                {
                    "definition": snapshot_payload["definition"],
                    "fingerprint": snapshot_payload["fingerprint"],
                    "provenance_detail": snapshot_payload["provenance"],
                    "readiness": snapshot_payload["readiness"],
                    "aliases": snapshot_payload["aliases"],
                }
            )
        pit_names = [str(item.get("pit_import_name") or ""), str(item["id"])]
        pit_universe = next((name for name in pit_names if name and name in installed), None)
        item.update(
            {
                "pit_available": pit_universe is not None,
                "pit_universe": pit_universe,
            }
        )
        results.append(item)
    return results


@app.post("/universes")
def add_universe(spec: UniverseSpec) -> dict[str, Any]:
    if spec.name == _DEFAULT_UNIVERSE:
        raise HTTPException(status_code=400, detail="sample universes cannot be replaced")
    universe = _persist_universe(spec)
    return {"name": spec.name, "symbols": universe.all_symbols()}


@app.get("/universes/{name}")
def get_universe(name: str) -> dict[str, Any]:
    universe = _resolve_universe(name, status_code=404)
    if universe.name == _DEFAULT_UNIVERSE:
        source = "sample"
    elif _universes.get(universe.name) is universe:
        source = "custom"
    else:
        source = "bundled"
    payload = _universe_api_payload(universe, source=source)
    payload["folder_id"] = universe_folders.tree([universe.name])["placements"][universe.name]
    return payload


@app.get("/universes/{name}/coverage")
def get_universe_coverage(name: str, as_of: str = _DEFAULT_AS_OF) -> dict[str, Any]:
    universe = _resolve_universe(name, status_code=404)
    return {"name": name, **_cache_coverage(universe, as_of)}


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
    with _universe_lifecycle_lock:
        _load_persisted_universes()
        if name not in _universes and bundled_snapshot_name(name) is not None:
            raise HTTPException(status_code=400, detail="bundled snapshots cannot be deleted")
        try:
            target = child_path(paths.universe_dir(), name, ".parquet", label="universe name")
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        if name not in _universes and not target.exists():
            raise HTTPException(status_code=404, detail="unknown universe")
        _universes.pop(name, None)
        if target.exists():
            target.unlink()
        universe_folders.forget_universe(name)
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


@app.get("/training/capabilities")
def get_training_capabilities() -> dict[str, Any]:
    accelerated, reason = _acceleration_status()
    return {
        **training_capabilities(accelerated=accelerated, fallback_reason=reason),
        "evaluator": cpp.selected_backend(),
        "cpp_available": cpp.available(),
        "fallback_reason": reason,
    }


@app.post("/runs", response_model=JobResponse)
def submit_run(
    req: RunRequest,
    panel: Panel | None = Depends(get_panel),  # noqa: B008
) -> JobResponse:
    config = _gp_config_from_request(req.config)
    resources = _resolve_training_resources(req.resources, ic_method=config.ic_method)
    universe_definition = _universe_definition_pin(req.universe)
    run_panel = _panel_for_universe(req.universe, req.as_of, panel)
    _preflight_split(config, run_panel)
    progress = RunProgress(
        target_generations=config.generations,
        resources=resources.to_dict(),
    )
    cancel = threading.Event()
    pinned_revisions: list[dict[str, Any]] = []

    def _task() -> dict[str, Any]:
        progress.set_phase("initializing")
        result = run_search(
            config,
            run_panel,
            progress=progress,
            stop=cancel.is_set,
            allowed_operators=allowed,
            resources=resources,
            scheduler=TRAINING_SCHEDULER,
        )
        result.setdefault("resources", resources.to_dict())
        result.setdefault("formula_revisions", pinned_revisions)
        result.setdefault("universe_definition", universe_definition)
        result.setdefault("context", {}).update(
            {
                "universe": universe_definition["name"],
                "universe_revision": universe_definition["fingerprint"],
                "requested_universe": req.universe,
                "universe_definition": universe_definition,
                "as_of": req.as_of,
            }
        )
        if "termination_reason" not in result:
            result["termination_reason"] = "user_stopped" if cancel.is_set() else "completed"
        return result

    with _formula_lifecycle_lock:
        _load_persisted_formulas()
        for spec in req.operators:  # register user operators before the search sees them
            _register(spec)
        formulas = {spec.runtime_name for spec in _read_formula_specs()}
        allowed = _allowed_operators(config, formulas, {spec.name for spec in req.operators})
        pinned_by_runtime: dict[str, dict[str, Any]] = {}
        allowed_formulas = formulas if allowed is None else formulas & allowed
        for runtime_name in sorted(allowed_formulas):
            for revision in _formula_dependency_revisions(Node(runtime_name)):
                pinned_by_runtime.setdefault(str(revision["runtime_name"]), revision)
        allowed_explicit = (
            {spec.name for spec in req.operators}
            if allowed is None
            else {spec.name for spec in req.operators} & allowed
        )
        for spec in req.operators:
            if spec.name not in allowed_explicit:
                continue
            for revision in _formula_dependency_revisions(tree_from_dict(spec.body)):
                pinned_by_runtime.setdefault(str(revision["runtime_name"]), revision)
        pinned_revisions = list(pinned_by_runtime.values())
        job_id = _jobs.submit(
            _task,
            progress=progress,
            metadata={
                "formula_revisions": pinned_revisions,
                "universe_definition": universe_definition,
            },
            cancel=cancel,
            on_success=_save_run_workspace,
        )
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
        "termination_reason": job.termination_reason,
        "progress": job.progress.snapshot() if job.progress is not None else None,
    }


@app.post("/runs/{job_id}/stop")
def stop_run(job_id: str) -> dict[str, bool]:
    """Cooperatively stop queued, scoring, or research-report work when still safe."""
    if _jobs.get(job_id) is None:
        raise HTTPException(status_code=404, detail="unknown job")
    return {"stopping": _jobs.cancel(job_id)}


@app.get("/runs/{job_id}/lineage")
def get_lineage(job_id: str) -> dict[str, Any]:
    job = _jobs.get(job_id)
    if job is None or not isinstance(job.result, dict):
        raise HTTPException(status_code=404, detail="no lineage yet")
    lineage: dict[str, Any] = job.result.get("lineage", {})
    return lineage


# --- exploratory formula tests --------------------------------------------------
@app.post("/formula-tests", response_model=JobResponse)
def submit_formula_test(
    req: FormulaTestRequest,
    panel: Panel | None = Depends(get_panel),  # noqa: B008
) -> JobResponse:
    universe_definition = _universe_definition_pin(req.universe)
    tree, source, bindings, dependencies = _resolve_formula_test_tree(req)
    test_panel, report_dates = _slice_formula_test_panel(req, panel)
    data_coverage, data_revision = _formula_test_data_metadata(
        test_panel, req.universe, report_dates
    )
    progress = FormulaTestProgress()
    cancel = threading.Event()

    def _task() -> dict[str, Any]:
        progress.set_phase("initializing")
        return _formula_test_result(
            tree=tree,
            panel=test_panel,
            req=req,
            source=source,
            bindings=bindings,
            dependencies=dependencies,
            report_dates=report_dates,
            data_coverage=data_coverage,
            data_revision=data_revision,
            universe_definition=universe_definition,
            progress=progress,
            cancel=cancel,
        )

    job_id = _formula_test_jobs.submit(
        _task,
        progress=progress,
        metadata={"universe_definition": universe_definition},
        cancel=cancel,
    )
    return JobResponse(job_id=job_id, status="queued")


@app.get("/formula-tests")
def list_formula_tests() -> list[dict[str, Any]]:
    return [_formula_test_job_payload(job) for job in reversed(_formula_test_jobs.list())]


@app.get("/formula-tests/{job_id}")
def get_formula_test(job_id: str) -> dict[str, Any]:
    job = _formula_test_jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="unknown formula test")
    return _formula_test_job_payload(job)


@app.post("/formula-tests/{job_id}/stop")
def stop_formula_test(job_id: str) -> dict[str, bool]:
    if _formula_test_jobs.get(job_id) is None:
        raise HTTPException(status_code=404, detail="unknown formula test")
    return {"stopping": _formula_test_jobs.cancel(job_id)}


@app.delete("/formula-tests/{job_id}")
def delete_formula_test(job_id: str) -> dict[str, str]:
    job = _formula_test_jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="unknown formula test")
    if job.status in {"queued", "running"}:
        raise HTTPException(
            status_code=409, detail="stop the active formula test before clearing it"
        )
    if not _formula_test_jobs.delete(job_id):
        raise HTTPException(status_code=409, detail="formula test could not be cleared")
    return {"removed": job_id}


@app.post("/formula-tests/{job_id}/keep")
def keep_formula_test(
    job_id: str,
    req: FormulaTestKeepRequest | None = None,
) -> dict[str, Any]:
    req = req or FormulaTestKeepRequest()
    job = _formula_test_jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="unknown formula test")
    if job.status != "done" or not isinstance(job.result, dict):
        raise HTTPException(status_code=409, detail="formula test has not completed")
    previous = str(job.result.get("kept_result_id") or "")
    if previous:
        saved = _factor_store().get(previous)
        if saved is not None:
            return saved.to_dict()
    name = req.name.strip()
    if not name:
        raise HTTPException(status_code=400, detail="formula result name must not be blank")
    result = job.result
    saved = _factor_store().save(
        name=name,
        tree=validate_tree(tree_from_dict(result["tree"])),
        metrics=result.get("metrics", {}),
        provenance={
            "source": "formula_test",
            "formula_test_id": job_id,
            "universe": result.get("universe"),
            "universe_definition": result.get("universe_definition"),
            "start": result.get("start"),
            "end": result.get("end"),
            "cumulative_trials": 0,
            "test_reads": 0,
            "exploratory": True,
        },
        notes=req.notes,
        saved_at=_now_iso(),
        kind="backtest",
        source=result.get("source", {}),
        bindings=result.get("bindings", {}),
        dependency_revisions=result.get("dependency_revisions", []),
        expression_fingerprint=str(result.get("expression_fingerprint") or ""),
        configuration=result.get("configuration", {}),
        data_coverage=result.get("data_coverage", {}),
        data_revision=str(result.get("data_revision") or ""),
        returns=result.get("returns", []),
        normalized_equity=result.get("normalized_equity", []),
        out_type=validate_tree(tree_from_dict(result["tree"])).out_type.value,
    )
    result["kept_result_id"] = saved.id
    return saved.to_dict()


# --- factor library --------------------------------------------------------------
@app.get("/formula-results")
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


@app.get("/formula-results/{factor_id}")
@app.get("/factors/{factor_id}")
def get_factor(factor_id: str) -> dict[str, Any]:
    try:
        factor = _factor_store().get(factor_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if factor is None:
        raise HTTPException(status_code=404, detail="unknown factor")
    return factor.to_dict()


@app.patch("/formula-results/{factor_id}")
@app.patch("/factors/{factor_id}")
def patch_factor(factor_id: str, patch: FactorPatch) -> dict[str, Any]:
    try:
        factor = _factor_store().update(factor_id, name=patch.name, notes=patch.notes)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if factor is None:
        raise HTTPException(status_code=404, detail="unknown factor")
    return factor.to_dict()


@app.delete("/formula-results/{factor_id}")
@app.delete("/factors/{factor_id}")
def delete_factor(factor_id: str) -> dict[str, str]:
    try:
        deleted = _factor_store().delete(factor_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if not deleted:
        raise HTTPException(status_code=404, detail="unknown factor")
    return {"removed": factor_id}


# --- settings --------------------------------------------------------------------
@app.get("/settings")
def get_settings() -> dict[str, Any]:
    stored = paths.read_settings()
    environment_key_set = bool(os.environ.get("TIINGO_API_KEY", "").strip())
    stored_key_set = bool(str(stored.get("tiingo_api_key") or "").strip())
    key_source = (
        "environment"
        if environment_key_set
        else "stored"
        if stored_key_set
        else "none"
    )
    return {
        "factors_dir": str(paths.factors_dir()),
        # Never echo, mask, size, or otherwise reveal the secret itself.
        "tiingo_api_key_set": key_source != "none",
        "tiingo_api_key_source": key_source,
        "tiingo_stored_key_set": stored_key_set,
        "evaluator": stored.get("evaluator", "auto"),
        "cpp_available": cpp.available(),
        # The LLM key is reported the same way: presence and source only, never the value.
        **explain_credentials.load_config(settings=stored).public_dict(),
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
    if any(
        value is not None
        for value in (
            update.llm_provider,
            update.llm_model,
            update.llm_base_url,
            update.llm_api_key,
        )
    ):
        try:
            explain_credentials.update_settings(
                settings,
                provider=update.llm_provider,
                model=update.llm_model,
                base_url=update.llm_base_url,
                api_key=update.llm_api_key,
                api_key_provider=update.llm_api_key_provider,
            )
        except LLMError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
    paths.write_settings(settings)
    return get_settings()


# --- local data usage + cleanup --------------------------------------------------
def _benchmark_definition(benchmark_id: str) -> dict[str, str]:
    for item in _BENCHMARK_CATALOG:
        if item["id"] == benchmark_id:
            return item
    raise HTTPException(status_code=404, detail="unknown benchmark")


def _benchmark_series_payload(
    benchmark: dict[str, str],
    start: str,
    end: str,
) -> dict[str, Any]:
    """Read a split-adjusted price-return benchmark without provider traffic.

    Cash dividends are deliberately excluded; only split normalization is applied before the
    cached close series is rebased to one.
    """
    symbol = benchmark["symbol"]
    cache = ParquetCache()
    coverage = _coverage_for_symbol(symbol, start, end, cache=cache)
    payload: dict[str, Any] = {
        **benchmark,
        "return_type": "price_return",
        "requested_start": start,
        "requested_end": end,
        "coverage": _model_dump(coverage),
        "status": "needs_sync",
        "message": "No cached benchmark prices cover this holdout.",
        "normalized_equity": [],
        "sync_request": {
            "symbols": [symbol],
            "start": start,
            # yfinance treats ``end`` as exclusive; the extra day is harmless for inclusive
            # providers and ensures the final holdout session can actually be cached.
            "end": _next_day_iso(end),
            "mode": "incremental",
        },
    }
    if not coverage.cached:
        return payload

    frame = cache.load(symbol)
    prices = split_adjusted_close(frame).loc[pd.Timestamp(start) : pd.Timestamp(end)]
    prices = prices.replace([np.inf, -np.inf], np.nan).dropna()
    prices = prices[prices > 0.0]
    if prices.empty:
        payload["message"] = "Cached prices contain no usable observations in this holdout."
        return payload

    # The requested end can be a weekend even though the last possible observation is Friday.
    # Use the observed trading-date boundary instead of repeatedly requesting impossible rows.
    observed_start = pd.Timestamp(prices.index.min()).normalize()
    observed_end = pd.Timestamp(prices.index.max()).normalize()
    required_end = pd.Timestamp(end)
    while required_end.weekday() >= 5:
        required_end -= pd.Timedelta(days=1)
    edge_complete = (
        observed_start <= pd.Timestamp(start)
        and observed_end >= required_end
    )
    payload["coverage"]["needs_sync"] = not edge_complete
    base = float(prices.iloc[0])
    payload["normalized_equity"] = [
        {
            "date": pd.Timestamp(date).date().isoformat(),
            "value": float(value) / base,
        }
        for date, value in prices.items()
    ]
    if not edge_complete:
        payload["status"] = "partial"
        payload["message"] = (
            "Cached benchmark history is partial for this holdout. "
            "The visible overlap is labelled and can be completed with Sync data."
        )
    else:
        payload["status"] = "ready"
        payload["message"] = None
    return payload


@app.get("/benchmarks")
def list_benchmarks() -> list[dict[str, str]]:
    """List supported comparison indices; this never downloads market data."""
    return [{**item, "return_type": "price_return"} for item in _BENCHMARK_CATALOG]


@app.get("/benchmarks/{benchmark_id}/series")
def get_benchmark_series(
    benchmark_id: str,
    start: str,
    end: str,
) -> dict[str, Any]:
    """Return cached split-adjusted price levels rebased to one over the holdout.

    The series excludes cash dividends and therefore is a price-return, not total-return,
    comparison.
    """
    clean_start, clean_end = _request_date_range(start, end)
    assert clean_start is not None and clean_end is not None
    return _benchmark_series_payload(
        _benchmark_definition(benchmark_id),
        clean_start,
        clean_end,
    )


@app.get("/symbols/search", response_model=list[SymbolCandidate])
def search_symbols(query: str, limit: int = 8) -> list[SymbolCandidate]:
    try:
        return _search_symbol_candidates(query, limit=limit)
    except Exception as exc:  # noqa: BLE001 - surface provider/search failures clearly
        raise HTTPException(status_code=502, detail=f"symbol search failed: {exc}") from exc


@app.post("/symbols/validate", response_model=SymbolValidation)
def validate_symbol(req: SymbolValidationRequest) -> SymbolValidation:
    start, end = _request_date_range(req.start, req.end)
    return _validate_symbol(req.symbol, start, end, force=req.force)


@app.get("/data/usage")
def data_usage() -> list[dict[str, Any]]:
    return usage.usage()


@app.get("/data/coverage", response_model=list[DataCoverage])
def data_coverage(
    symbols: str,
    start: str | None = None,
    end: str | None = None,
) -> list[DataCoverage]:
    start, end = _request_date_range(start, end)
    parsed = _market_symbols(symbols.split(","))
    return [_coverage_for_symbol(symbol, start, end) for symbol in parsed]


@app.post("/data/sync")
def data_sync(req: DataSyncRequest) -> dict[str, Any]:
    mode = req.mode.lower()
    if mode not in _SYNC_MODES:
        raise HTTPException(status_code=400, detail=f"mode must be one of {sorted(_SYNC_MODES)}")
    start, end = _request_date_range(req.start, req.end)
    assert start is not None
    symbols = set(_market_symbols(req.symbols))
    aliases: dict[str, str] = {}
    universe_pin = None
    resolved_universe = None
    if req.universe:
        universe = _resolve_universe(req.universe)
        resolved_universe = universe.name
        universe_pin = _universe_definition_pin(req.universe)
        aliases = {
            normalize_market_symbol(source): normalize_market_symbol(provider)
            for source, provider in universe.aliases.items()
        }
        if universe.mode == "static_snapshot":
            symbols.update(universe.all_symbols())
        else:
            symbols.update(universe.members_overlapping(start, end or _today_iso()))
    if not symbols:
        raise HTTPException(
            status_code=400,
            detail="at least one symbol or a universe with memberships in range is required",
        )
    request = DataSyncRequest(
        symbols=sorted(symbols),
        universe=resolved_universe,
        start=start,
        end=end,
        mode=mode,
        aliases={symbol: aliases.get(symbol, symbol) for symbol in sorted(symbols)},
    )
    progress = SyncProgress(total=len(symbols))
    cancel = threading.Event()

    def task() -> dict[str, Any]:
        result = _run_data_sync(request, progress, cancel.is_set)
        result["universe_definition"] = universe_pin
        return result

    metadata = {
        "kind": "data_sync",
        "request": _model_dump(request),
        "universe_definition": universe_pin,
    }
    # Repeated clicks/navigation re-entry should attach to the same immutable request instead of
    # launching duplicate provider traffic. The lock closes the check/submit race between clients.
    with _data_sync_submit_lock:
        for job in reversed(_data_jobs.list()):
            if (
                job.metadata.get("kind") == "data_sync"
                and job.status in {"queued", "running"}
                and not job.cancel.is_set()
                and job.metadata.get("request") == metadata["request"]
                and job.metadata.get("universe_definition") == universe_pin
            ):
                return {"job_id": job.id, "status": job.status, "reused": True}
        job_id = _data_jobs.submit(
            task,
            progress=progress,
            metadata=metadata,
            cancel=cancel,
        )
    return {"job_id": job_id, "status": "queued", "reused": False}


def _data_sync_job_payload(job: Any) -> dict[str, Any]:
    return {
        "job_id": job.id,
        "status": job.status,
        "result": job.result,
        "error": job.error,
        "termination_reason": job.termination_reason,
        "stopping": job.cancel.is_set() and job.status in {"queued", "running"},
        "request": job.metadata.get("request"),
        "universe_definition": job.metadata.get("universe_definition"),
        "progress": job.progress.snapshot() if job.progress else None,
    }


@app.get("/data/sync")
def list_data_sync_jobs(active_only: bool = False) -> list[dict[str, Any]]:
    jobs = [
        job
        for job in _data_jobs.list()
        if job.metadata.get("kind") == "data_sync"
        and (not active_only or job.status in {"queued", "running"})
    ]
    return [_data_sync_job_payload(job) for job in reversed(jobs)]


@app.get("/data/sync/{job_id}")
def data_sync_status(job_id: str) -> dict[str, Any]:
    job = _data_jobs.get(job_id)
    if job is None or job.metadata.get("kind") != "data_sync":
        raise HTTPException(status_code=404, detail="unknown sync job")
    return _data_sync_job_payload(job)


@app.post("/data/sync/{job_id}/stop")
def stop_data_sync(job_id: str) -> dict[str, bool]:
    job = _data_jobs.get(job_id)
    if job is None or job.metadata.get("kind") != "data_sync":
        raise HTTPException(status_code=404, detail="unknown sync job")
    return {"stopping": _data_jobs.cancel(job_id)}


@app.post("/universes/sync-dates")
def universes_sync_dates(req: MembershipSyncRequest) -> dict[str, str]:
    symbols = _market_symbols(req.symbols)
    if not symbols:
        raise HTTPException(status_code=400, detail="at least one symbol is required")
    expected_start = _request_timestamp(req.expected_start, label="expected start")
    request = MembershipSyncRequest(
        symbols=symbols,
        expected_start=expected_start.date().isoformat(),
    )
    progress = SyncProgress(total=len(symbols))
    job_id = _data_jobs.submit(lambda: _run_membership_sync(request, progress), progress=progress)
    return {"job_id": job_id, "status": "queued"}


@app.get("/universes/sync-dates/{job_id}")
def universes_sync_dates_status(job_id: str) -> dict[str, Any]:
    job = _data_jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="unknown sync job")
    return {
        "job_id": job.id,
        "status": job.status,
        "result": job.result,
        "error": job.error,
        "progress": job.progress.snapshot() if job.progress else None,
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


def _session_exists(session_id: str) -> bool:
    try:
        return sessions.exists(session_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def _live_session_job(session: dict[str, Any]) -> Any | None:
    """Return an active in-process job and clear a stale persisted reservation."""
    active_id = str(session.get("active_job_id") or "")
    legacy_id = str(session.get("last_job_id") or "")
    job = _jobs.get(active_id or legacy_id)
    if job is not None and job.status in ("queued", "running"):
        return job
    if active_id:
        status = job.status if job is not None else "interrupted"
        error = job.error if job is not None else "application restarted during the segment"
        sessions.finish_job(session["id"], active_id, status, error=error)
    return None


def _session_job_view(session: dict[str, Any]) -> dict[str, Any] | None:
    job = _live_session_job(session)
    if job is not None:
        return {
            "id": job.id,
            "status": job.status,
            "termination_reason": job.termination_reason,
            "progress": job.progress.snapshot() if job.progress is not None else None,
        }
    persisted = session.get("last_job")
    if isinstance(persisted, dict):
        return {**persisted, "progress": None}
    return None


def _session_finalization_job_view(
    session: dict[str, Any],
) -> dict[str, Any] | None:
    def normalized(payload: dict[str, Any]) -> dict[str, Any]:
        metadata = dict(payload.get("metadata") or {})
        for key in (
            "evaluation_id",
            "round_index",
            "strategy_plan_id",
            "comparison_id",
            "primary_strategy_id",
        ):
            if key not in metadata:
                metadata[key] = payload.get(key)
        return {
            **payload,
            "id": str(payload.get("id") or payload.get("job_id") or ""),
            "progress": payload.get("progress"),
            "metadata": metadata,
        }

    active_id = str(session.get("active_finalization_job_id") or "")
    job = _jobs.get(active_id)
    if job is not None and job.status in ("queued", "running"):
        return normalized({
            "id": job.id,
            "status": job.status,
            "termination_reason": job.termination_reason,
            "progress": job.progress.snapshot() if job.progress is not None else None,
            "metadata": dict(job.metadata),
        })
    if active_id:
        status = job.status if job is not None else "interrupted"
        error = job.error if job is not None else "application restarted during finalization"
        sessions.finish_finalization_job(
            session["id"],
            active_id,
            status,
            error=error,
        )
    persisted = session.get("last_finalization_job")
    return normalized(dict(persisted)) if isinstance(persisted, dict) else None


@app.post("/sessions")
def create_session(
    req: SessionCreateRequest,
    panel: Panel | None = Depends(get_panel),  # noqa: B008
) -> dict[str, str]:
    name = _session_name(req)
    config = _gp_config_from_request(req.config)
    requested_resources = _requested_resources(req.resources)
    resources = _resolve_training_resources(req.resources, ic_method=config.ic_method)
    resolved_as_of = _as_of_timestamp(req.as_of).date().isoformat()
    universe_definition = _universe_definition_pin(req.universe)
    panel = _panel_for_universe(req.universe, resolved_as_of, panel)
    _validate_panel_config(config, panel)
    try:
        boundaries = sessions.derive_boundaries(
            panel.dates,
            train=req.train,
            valid=req.valid,
            embargo=req.embargo,
            horizon=label_span(config.horizon, config.execution),
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    _load_persisted_formulas()
    for spec in req.operators:
        _register(spec)
    (
        seeds,
        trial_baseline,
        test_reads_baseline,
        inherited_evidence_sources,
    ) = _load_seed_factors(req.seed_factor_ids)
    _validate_seeds(seeds, config)

    formula_revisions = _active_formula_operator_specs()
    allowed = _allowed_operators(
        config,
        {str(item["name"]) for item in formula_revisions},
        {spec.name for spec in req.operators},
    )
    session = sessions.new_session(
        name=name,
        universe=str(universe_definition["name"]),
        universe_definition=universe_definition,
        as_of=resolved_as_of,
        config=config,
        operators=[_model_dump(spec) for spec in req.operators],
        formula_revisions=formula_revisions,
        seed_factor_ids=req.seed_factor_ids,
        inherited_evidence_sources=inherited_evidence_sources,
        boundaries=boundaries,
        trial_baseline=trial_baseline,
        test_reads_baseline=test_reads_baseline,
        created_at=_now_iso(),
        resources=requested_resources.to_dict(),
    )
    session_id = session["id"]
    progress = RunProgress(
        target_generations=config.generations,
        resources=resources.to_dict(),
    )
    cancel = threading.Event()
    job_id = uuid.uuid4().hex
    if not sessions.claim_job(
        session_id,
        job_id,
        requested_generations=config.generations,
        config=config.to_dict(),
        resources=requested_resources.to_dict(),
    ):
        raise HTTPException(status_code=409, detail="a segment is already running")

    def _task() -> dict[str, Any]:
        sessions.update_job_status(session_id, job_id, "running")
        progress.set_phase("initializing")
        try:
            result = sessions.run_segment(
                session_id,
                job_id=job_id,
                panel=panel,
                config=config,
                generations=config.generations,
                seeds=seeds,
                progress=progress,
                stop=cancel.is_set,
                allowed_operators=allowed,
                resources=resources,
                scheduler=TRAINING_SCHEDULER,
            )
        except (TrainingCancelled, TrainingLeaseCancelled):
            sessions.finish_job(session_id, job_id, "stopped")
            raise
        except Exception as exc:
            sessions.finish_job(session_id, job_id, "failed", error=repr(exc))
            raise
        sessions.finish_job(
            session_id,
            job_id,
            "stopped" if result.get("termination_reason") == "user_stopped" else "done",
        )
        return result

    try:
        _jobs.submit(
            _task,
            job_id=job_id,
            progress=progress,
            metadata={"universe_definition": universe_definition},
            cancel=cancel,
        )
    except Exception as exc:
        sessions.finish_job(session_id, job_id, "failed", error=repr(exc))
        raise
    return {"session_id": session_id, "job_id": job_id}


@app.post("/sessions/{session_id}/continue")
def continue_session(
    session_id: str,
    req: SessionContinueRequest,
    panel: Panel | None = Depends(get_panel),  # noqa: B008
) -> dict[str, str]:
    _load_persisted_formulas()
    if not _session_exists(session_id):
        raise HTTPException(status_code=404, detail="unknown session")
    session = sessions.load_session(session_id)

    if _live_session_job(session) is not None:
        raise HTTPException(status_code=409, detail="a segment is already running")
    active_finalization_id = str(session.get("active_finalization_job_id") or "")
    active_finalization = _jobs.get(active_finalization_id)
    if active_finalization is not None and active_finalization.status in {
        "queued",
        "running",
    }:
        raise HTTPException(status_code=409, detail="holdout finalization is running")
    restart_reason = sessions.continuation_restart_reason(session_id)
    if restart_reason:
        raise HTTPException(
            status_code=409,
            detail=f"Restart with the same setup: {restart_reason}",
        )

    if req.resources is None:
        stored_resources = session.get("resources") or {
            "profile": "auto",
            "cpu_budget_percent": None,
        }
        if (
            "cpu_budget_percent" not in stored_resources
            and "custom_percent" in stored_resources
        ):
            stored_resources = {
                **stored_resources,
                "cpu_budget_percent": stored_resources.get("custom_percent"),
            }
        resource_request = TrainingResourcesRequest(**stored_resources)
    else:
        resource_request = req.resources
    requested_resources = _requested_resources(resource_request)
    stored_config = _gp_config_from_request(session["config"])
    merged = {**session["config"], **req.config}
    config = _gp_config_from_request(merged)
    resources = _resolve_training_resources(resource_request, ic_method=config.ic_method)
    if config.population_size * req.generations > MAX_SEARCH_EVALUATIONS:
        raise HTTPException(
            status_code=400,
            detail=(
                "population_size * additional generations must not exceed "
                f"{MAX_SEARCH_EVALUATIONS:,} evaluations"
            ),
        )
    boundaries = sessions.Boundaries.from_dict(session["boundaries"])
    if config.execution != stored_config.execution:
        # Execution timing defines what every stored IC, backtest and holdout fingerprint in this
        # session measures. Changing it mid-session would silently mix incomparable evidence.
        raise HTTPException(
            status_code=400,
            detail=(
                f"execution timing is frozen for this session ({stored_config.execution}); "
                "start a new session to research a different timing"
            ),
        )
    span = label_span(config.horizon, config.execution)
    if span > boundaries.embargo:
        raise HTTPException(
            status_code=400,
            detail=(
                f"horizon ({config.horizon}) plus the execution delay ({span - config.horizon}) "
                f"exceeds the session's frozen embargo ({boundaries.embargo})"
            ),
        )

    for spec in req.operators:  # newly added operators for this segment
        try:
            ensure_operator(
                spec.name,
                [DType(t) for t in spec.arg_types],
                DType(spec.out_type),
                spec.body,
                policy=spec.policy,
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
                policy=stored.get("policy"),
            )
        except (InvalidOperator, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
    pinned_formulas = session.get("formula_revisions") or _active_formula_operator_specs()
    for stored in pinned_formulas:
        try:
            ensure_operator(
                stored.get("runtime_name") or stored["name"],
                [DType(t) for t in stored["arg_types"]],
                DType(stored["out_type"]),
                stored["body"],
                policy=_formula_policy(FormulaSpec(**stored)),
            )
        except (InvalidOperator, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    requested_universe = req.universe or session["universe"]
    universe_definition = _universe_definition_pin(requested_universe)
    universe = str(universe_definition["name"])
    new_panel = _panel_for_universe(requested_universe, session["as_of"], panel)
    _validate_panel_config(config, new_panel)
    stored_universe_definition = session.get("universe_definition") or {}
    universe_changed = (
        universe != session["universe"]
        or stored_universe_definition.get("fingerprint") != universe_definition["fingerprint"]
    )
    scoring_changed = any(
        getattr(stored_config, field) != getattr(config, field)
        for field in ("parsimony", "ic_method", "min_names", "horizon")
    )
    rescore = universe_changed or scoring_changed

    extra_seeds, _, _, _ = _load_seed_factors(req.seed_factor_ids)
    _validate_seeds(extra_seeds, config)

    effective_operator_names = {
        str(item["name"]) for item in session.get("operators", [])
    } | {spec.name for spec in req.operators}
    allowed = _allowed_operators(
        config,
        {str(item["name"]) for item in pinned_formulas},
        effective_operator_names,
    )

    job_id = uuid.uuid4().hex
    if not sessions.claim_job(
        session_id,
        job_id,
        requested_generations=req.generations,
        config=config.to_dict(),
        resources=requested_resources.to_dict(),
    ):
        raise HTTPException(status_code=409, detail="a segment is already running")

    # Reload the reservation before persisting config/operator changes so a stale request
    # snapshot cannot erase ``active_job_id``.
    session = sessions.load_session(session_id)
    if req.operators:
        by_name = {str(item["name"]): item for item in session.get("operators", [])}
        by_name.update({spec.name: _model_dump(spec) for spec in req.operators})
        session["operators"] = list(by_name.values())
    session["universe"] = universe
    session["universe_definition"] = universe_definition
    session["config"] = merged
    session["resources"] = requested_resources.to_dict()
    sessions.save_session(session)

    progress = RunProgress(
        target_generations=req.generations,
        resources=resources.to_dict(),
    )
    cancel = threading.Event()

    def _task() -> dict[str, Any]:
        sessions.update_job_status(session_id, job_id, "running")
        progress.set_phase("initializing")
        try:
            result = sessions.run_segment(
                session_id,
                job_id=job_id,
                panel=new_panel,
                config=config,
                generations=req.generations,
                extra_seeds=extra_seeds,
                rescore=rescore,
                progress=progress,
                stop=cancel.is_set,
                allowed_operators=allowed,
                resources=resources,
                scheduler=TRAINING_SCHEDULER,
            )
        except (TrainingCancelled, TrainingLeaseCancelled):
            sessions.finish_job(session_id, job_id, "stopped")
            raise
        except Exception as exc:
            sessions.finish_job(session_id, job_id, "failed", error=repr(exc))
            raise
        sessions.finish_job(
            session_id,
            job_id,
            "stopped" if result.get("termination_reason") == "user_stopped" else "done",
        )
        return result

    try:
        _jobs.submit(
            _task,
            job_id=job_id,
            progress=progress,
            metadata={"universe_definition": universe_definition},
            cancel=cancel,
        )
    except Exception as exc:
        sessions.finish_job(session_id, job_id, "failed", error=repr(exc))
        raise
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
        _session_job_view(session)  # reconcile a reservation left by a prior process
        session = sessions.load_session(directory.name)
        segments = session.get("segments") or []
        rounds = sessions.list_rounds(session["id"])
        finalizations = sessions.list_finalizations(session["id"])
        last_job = session.get("last_job") or {}
        current_generation = max(
            (
                int(segment["gen_end"])
                for segment in segments
                if segment.get("gen_end") is not None
            ),
            default=0,
        )
        summaries.append(
            {
                "id": session["id"],
                "name": session["name"],
                "created_at": session.get("created_at", ""),
                "updated_at": session.get("updated_at", session.get("created_at", "")),
                "universe": session["universe"],
                "segments": len(segments),
                "rounds": len(rounds),
                "latest_completed_round": rounds[-1]["index"] if rounds else None,
                "current_generation": current_generation,
                "last_status": last_job.get("status")
                or (segments[-1].get("status") if segments else "created"),
                "has_checkpoint": (
                    sessions.session_dir(session["id"]) / "checkpoint.json"
                ).exists(),
                "has_report": bool(finalizations),
                "finalizations": len(finalizations),
                "requested_generations": session.get("last_requested_generations")
                or session.get("config", {}).get("generations"),
                "cumulative_trials": session.get("cumulative_trials", 0),
                "test_reads": session.get("test_reads", 0),
            }
        )
    return sorted(
        summaries,
        key=lambda item: (item["updated_at"], item["created_at"]),
        reverse=True,
    )


@app.get("/sessions/{session_id}")
def get_session(session_id: str) -> dict[str, Any]:
    if not _session_exists(session_id):
        raise HTTPException(status_code=404, detail="unknown session")
    session = sessions.load_session(session_id)
    job = _session_job_view(session)
    finalization_job = _session_finalization_job_view(session)
    # Re-read after stale-job reconciliation so an application restart never returns
    # an already-cleared active reservation to a reconnecting client.
    session = sessions.load_session(session_id)
    session["rounds"] = sessions.list_rounds(session_id)
    result_path = sessions.session_dir(session_id) / "result.json"
    result = None
    if result_path.exists():
        result = json.loads(result_path.read_text(encoding="utf-8"))
        result = {
            key: value
            for key, value in result.items()
            if key not in {"lineage", "report_trials"}
        }
    return {
        **session,
        "job": job,
        "finalization_job": finalization_job,
        "result": result,
    }


@app.get("/sessions/{session_id}/lineage")
def get_session_lineage(
    session_id: str,
    round: int | None = Query(default=None, ge=0),  # noqa: A002
) -> dict[str, Any]:
    if not _session_exists(session_id):
        raise HTTPException(status_code=404, detail="unknown session")
    if round is not None:
        round_lineage = sessions.lineage_for_round(session_id, round)
        if round_lineage is None:
            raise HTTPException(status_code=404, detail="unknown or unavailable round")
        return round_lineage
    lineage_path = sessions.session_dir(session_id) / "lineage.json"
    if not lineage_path.exists():
        raise HTTPException(status_code=404, detail="no lineage yet")
    lineage: dict[str, Any] = json.loads(lineage_path.read_text(encoding="utf-8"))
    return lineage


@app.get("/sessions/{session_id}/rounds")
def get_session_rounds(session_id: str) -> list[dict[str, Any]]:
    if not _session_exists(session_id):
        raise HTTPException(status_code=404, detail="unknown session")
    return sessions.list_rounds(session_id)


@app.post(
    "/sessions/{session_id}/rounds/{round_index}/strategy-comparisons"
)
def compare_session_round_strategies(
    session_id: str,
    round_index: int,
    req: SessionStrategyComparisonRequest,
    panel: Panel | None = Depends(get_panel),  # noqa: B008
) -> dict[str, Any]:
    """Compare up to four weighting strategies on validation data only."""
    if not _session_exists(session_id):
        raise HTTPException(status_code=404, detail="unknown session")
    round_payload = sessions.load_round(session_id, round_index)
    if round_index < 0 or round_payload is None:
        raise HTTPException(status_code=404, detail="unknown round")
    if round_payload.get("restart_required"):
        raise HTTPException(
            status_code=409,
            detail="Restart with the same setup before comparing this legacy round.",
        )
    try:
        strategy_specs = [item.to_spec() for item in req.strategies]
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    session = sessions.load_session(session_id)
    context = dict(round_payload.get("context") or {})
    universe = str(context.get("universe") or session["universe"])
    as_of = str(context.get("as_of") or session["as_of"])
    resolved_panel = _panel_for_universe(universe, as_of, panel)
    comparison_id = uuid.uuid4().hex
    job_id = uuid.uuid4().hex
    claimed, reason = sessions.claim_strategy_comparison(
        session_id,
        job_id=job_id,
        comparison_id=comparison_id,
        round_index=round_index,
        strategies=strategy_specs,
        confirm_repeat=req.confirm_repeat,
    )
    if not claimed:
        status = 404 if reason == "unknown round" else 409
        raise HTTPException(status_code=status, detail=reason)
    progress = RunProgress(target_generations=0)
    cancel = threading.Event()

    def _task() -> dict[str, Any]:
        sessions.update_strategy_comparison_status(
            session_id, comparison_id, job_id, "running"
        )
        try:
            return sessions.run_strategy_comparison(
                session_id,
                round_index,
                comparison_id=comparison_id,
                job_id=job_id,
                panel=resolved_panel,
                progress=progress,
                stop=cancel.is_set,
            )
        except (TrainingCancelled, TrainingLeaseCancelled):
            sessions.update_strategy_comparison_status(
                session_id, comparison_id, job_id, "stopped"
            )
            raise
        except Exception as exc:
            sessions.update_strategy_comparison_status(
                session_id,
                comparison_id,
                job_id,
                "failed",
                error=repr(exc),
            )
            raise

    try:
        _jobs.submit(
            _task,
            job_id=job_id,
            progress=progress,
            metadata={
                "session_id": session_id,
                "round_index": round_index,
                "comparison_id": comparison_id,
                "kind": "strategy_comparison",
            },
            cancel=cancel,
        )
    except Exception as exc:
        sessions.update_strategy_comparison_status(
            session_id,
            comparison_id,
            job_id,
            "failed",
            error=repr(exc),
        )
        raise
    return {
        "session_id": session_id,
        "round_index": round_index,
        "comparison_id": comparison_id,
        "job_id": job_id,
        "status": "queued",
    }


@app.get(
    "/sessions/{session_id}/rounds/{round_index}/strategy-comparisons"
)
def get_session_round_strategy_comparisons(
    session_id: str,
    round_index: int,
) -> list[dict[str, Any]]:
    if not _session_exists(session_id):
        raise HTTPException(status_code=404, detail="unknown session")
    if sessions.load_round(session_id, round_index) is None:
        raise HTTPException(status_code=404, detail="unknown round")
    return [
        item
        for item in sessions.list_strategy_comparisons(session_id)
        if int(item.get("round_index", -1)) == round_index
    ]


@app.get(
    "/sessions/{session_id}/rounds/{round_index}/strategy-comparisons/{comparison_id}"
)
def get_session_round_strategy_comparison(
    session_id: str,
    round_index: int,
    comparison_id: str,
) -> dict[str, Any]:
    if not _session_exists(session_id):
        raise HTTPException(status_code=404, detail="unknown session")
    try:
        payload = sessions.load_strategy_comparison(session_id, comparison_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail="unknown strategy comparison") from exc
    if payload is None or int(payload.get("round_index", -1)) != round_index:
        raise HTTPException(status_code=404, detail="unknown strategy comparison")
    return payload


# --- overlap with known factors ---------------------------------------------------------------
class SessionOverlapRequest(BaseModel):
    include_saved_results: bool = True
    top_k: int = Field(default=5, ge=0, le=10)


def _overlap_path(session_id: str, round_index: int) -> Path:
    return sessions.session_dir(session_id) / "overlap" / f"round-{round_index:04d}.json"


def _ensure_session_operators(session: dict[str, Any]) -> None:
    """Register the formula store plus the session's own operators so its trees evaluate."""
    _load_persisted_formulas()
    pinned = [
        *(session.get("formula_revisions") or []),
        *(session.get("operators") or []),
    ]
    for stored in pinned:
        try:
            ensure_operator(
                stored.get("runtime_name") or stored["name"],
                [DType(t) for t in stored["arg_types"]],
                DType(stored["out_type"]),
                stored["body"],
                policy=stored.get("policy")
                if "runtime_name" not in stored
                else _formula_policy(FormulaSpec(**stored)),
            )
        except (InvalidOperator, ValueError, KeyError, TypeError):
            # A pinned revision that no longer registers only affects references using it; the
            # candidate's own evaluation reports the failure explicitly.
            continue


def _overlap_references(include_saved_results: bool) -> list[ReferenceFactor]:
    references = references_from_formulas(_load_persisted_formulas())
    if include_saved_results:
        for factor in _factor_store().list():
            references.append(
                ReferenceFactor(
                    key=f"result:{factor.id}",
                    name=factor.id,
                    display_name=factor.name,
                    group="saved_results",
                    family=str(factor.kind or "formula_result"),
                    tree=factor.expanded_tree or factor.tree,
                )
            )
    return references


def _reference_signature(references: list[ReferenceFactor]) -> str:
    digest = hashlib.sha256()
    for reference in sorted(references, key=lambda item: item.key):
        digest.update(reference.key.encode("utf-8"))
        digest.update(tree_to_json(reference.tree).encode("utf-8"))
    return digest.hexdigest()


def _run_round_overlap(
    session_id: str,
    round_index: int,
    *,
    panel: Panel,
    request: SessionOverlapRequest,
    progress: RunProgress,
    stop: Any,
) -> dict[str, Any]:
    session = sessions.load_session(session_id)
    round_payload = sessions.load_round(session_id, round_index)
    if round_payload is None or not isinstance(round_payload.get("best_factor"), str):
        raise ValueError("round has no selected formula")
    metadata = dict(round_payload.get("round_metadata") or {})
    config = GPConfig.from_dict(metadata.get("config") or session["config"])
    boundaries = sessions.Boundaries.from_dict(session["boundaries"])
    train_end = pd.Timestamp(boundaries.train_end)
    keep = pd.DatetimeIndex(panel.dates) <= train_end
    if not bool(keep.any()):
        raise ValueError("the session's training window has no cached dates")
    # Physically truncated: labels near the boundary are missing, never read from validation.
    train_panel = Panel({name: frame.loc[keep] for name, frame in panel.fields.items()})
    train_dates = sessions.split_from_boundaries(panel.dates, boundaries).train

    progress.set_phase("evaluating")
    _ensure_session_operators(session)
    candidate_tree = tree_from_json(str(round_payload["best_factor"]))
    candidate = evaluate(expand_all(candidate_tree), train_panel)
    if not isinstance(candidate, pd.DataFrame):
        raise ValueError("the selected formula did not evaluate to a panel")
    references = _overlap_references(request.include_saved_results)
    evaluated: list[tuple[ReferenceFactor, pd.DataFrame | None, str | None]] = []
    for done, reference in enumerate(references, start=1):
        if stop():
            raise TrainingCancelled("overlap check stopped")
        try:
            value = evaluate(expand_all(reference.tree), train_panel)
            if isinstance(value, pd.DataFrame):
                evaluated.append((reference, value, None))
            else:
                evaluated.append((reference, None, "did not evaluate to a panel"))
        except Exception as exc:  # noqa: BLE001 - one broken reference must not end the check
            evaluated.append((reference, None, str(exc)[:200]))
        progress.set_report_progress(done, len(references) * 2)

    def measured(done: int, total: int) -> None:
        if stop():
            raise TrainingCancelled("overlap check stopped")
        progress.set_report_progress(total + done, total * 2)

    report = overlap_report(
        candidate,
        evaluated,
        forward_returns(train_panel, config.horizon, config.execution),
        train_dates,
        min_names=config.min_names,
        top_k=request.top_k,
        progress=measured,
    )
    payload = {
        **report,
        "session_id": session_id,
        "round_index": round_index,
        "best_factor": round_payload["best_factor"],
        "horizon": config.horizon,
        "execution": config.execution,
        "include_saved_results": request.include_saved_results,
        "reference_signature": _reference_signature(references),
        "computed_at": _now_iso(),
        "disclaimer": DISCLAIMER,
    }
    target = _overlap_path(session_id, round_index)
    target.parent.mkdir(parents=True, exist_ok=True)
    safe = _json_finite(payload)
    atomic_write_text(target, json.dumps(safe, allow_nan=False, sort_keys=True))
    return safe


def _json_finite(value: Any) -> Any:
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {key: _json_finite(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_finite(item) for item in value]
    return value


@app.post("/sessions/{session_id}/rounds/{round_index}/overlap")
def start_round_overlap(
    session_id: str,
    round_index: int,
    req: SessionOverlapRequest | None = None,
    panel: Panel | None = Depends(get_panel),  # noqa: B008
) -> dict[str, Any]:
    """Compare a round's selected formula with known factors on the training window only."""
    request = req or SessionOverlapRequest()
    if not _session_exists(session_id):
        raise HTTPException(status_code=404, detail="unknown session")
    round_payload = sessions.load_round(session_id, round_index)
    if round_index < 0 or round_payload is None:
        raise HTTPException(status_code=404, detail="unknown round")
    if not isinstance(round_payload.get("best_factor"), str):
        raise HTTPException(status_code=409, detail="round has no selected formula")
    session = sessions.load_session(session_id)
    context = dict(round_payload.get("context") or {})
    universe = str(context.get("universe") or session["universe"])
    as_of = str(context.get("as_of") or session["as_of"])
    resolved_panel = _panel_for_universe(universe, as_of, panel)
    for job in _jobs.list():
        if (
            job.metadata.get("kind") == "round_overlap"
            and job.metadata.get("session_id") == session_id
            and job.metadata.get("round_index") == round_index
            and job.status in {"queued", "running"}
        ):
            return {"job_id": job.id, "status": job.status, "reused": True}
    progress = RunProgress(target_generations=0)
    cancel = threading.Event()
    job_id = uuid.uuid4().hex

    def _task() -> dict[str, Any]:
        return _run_round_overlap(
            session_id,
            round_index,
            panel=resolved_panel,
            request=request,
            progress=progress,
            stop=cancel.is_set,
        )

    _jobs.submit(
        _task,
        job_id=job_id,
        progress=progress,
        metadata={"kind": "round_overlap", "session_id": session_id, "round_index": round_index},
        cancel=cancel,
    )
    return {"job_id": job_id, "status": "queued", "reused": False}


@app.get("/sessions/{session_id}/rounds/{round_index}/overlap")
def get_round_overlap(session_id: str, round_index: int) -> dict[str, Any]:
    """The last completed overlap check for a round, flagged stale if its inputs changed."""
    if not _session_exists(session_id):
        raise HTTPException(status_code=404, detail="unknown session")
    round_payload = sessions.load_round(session_id, round_index)
    if round_index < 0 or round_payload is None:
        raise HTTPException(status_code=404, detail="unknown round")
    path = _overlap_path(session_id, round_index)
    if not path.exists():
        raise HTTPException(status_code=404, detail="no overlap check yet")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=404, detail="no readable overlap check") from exc
    stale_reasons: list[str] = []
    if payload.get("overlap_version") != OVERLAP_VERSION:
        stale_reasons.append("computed by an older overlap method")
    if payload.get("best_factor") != round_payload.get("best_factor"):
        stale_reasons.append("the round's selected formula changed")
    try:
        references = _overlap_references(bool(payload.get("include_saved_results", True)))
        if payload.get("reference_signature") != _reference_signature(references):
            stale_reasons.append("formulas or saved results changed since this check")
    except Exception:  # noqa: BLE001 - staleness is advisory; never hide the stored result
        stale_reasons.append("reference factors could not be re-read")
    return {**payload, "stale": bool(stale_reasons), "stale_reasons": stale_reasons}


@app.post("/sessions/{session_id}/rounds/{round_index}/finalization-plans")
def create_session_round_finalization_plan(
    session_id: str,
    round_index: int,
    req: SessionFinalizationPlanRequest,
) -> dict[str, Any]:
    if not _session_exists(session_id):
        raise HTTPException(status_code=404, detail="unknown session")
    if sessions.load_round(session_id, round_index) is None:
        raise HTTPException(status_code=404, detail="unknown round")
    try:
        return sessions.create_finalization_plan(
            session_id,
            round_index,
            comparison_id=req.comparison_id,
            primary_strategy_id=req.primary_strategy_id,
        )
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.get("/sessions/{session_id}/rounds/{round_index}/finalization-plans")
def get_session_round_finalization_plans(
    session_id: str,
    round_index: int,
) -> list[dict[str, Any]]:
    if not _session_exists(session_id):
        raise HTTPException(status_code=404, detail="unknown session")
    if sessions.load_round(session_id, round_index) is None:
        raise HTTPException(status_code=404, detail="unknown round")
    return [
        item
        for item in sessions.list_finalization_plans(session_id)
        if int(item.get("round_index", -1)) == round_index
    ]


@app.get(
    "/sessions/{session_id}/rounds/{round_index}/finalization-plans/{strategy_plan_id}"
)
def get_session_round_finalization_plan(
    session_id: str,
    round_index: int,
    strategy_plan_id: str,
) -> dict[str, Any]:
    if not _session_exists(session_id):
        raise HTTPException(status_code=404, detail="unknown session")
    try:
        payload = sessions.load_finalization_plan(session_id, strategy_plan_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail="unknown finalization plan") from exc
    if payload is None or int(payload.get("round_index", -1)) != round_index:
        raise HTTPException(status_code=404, detail="unknown finalization plan")
    return payload


@app.post("/sessions/{session_id}/rounds/{round_index}/finalize")
def finalize_session_round(
    session_id: str,
    round_index: int,
    req: SessionFinalizeRequest,
    panel: Panel | None = Depends(get_panel),  # noqa: B008
) -> dict[str, Any]:
    if not _session_exists(session_id):
        raise HTTPException(status_code=404, detail="unknown session")
    if round_index < 0:
        raise HTTPException(status_code=404, detail="unknown round")
    round_payload = sessions.load_round(session_id, round_index)
    if round_payload is None:
        raise HTTPException(status_code=404, detail="unknown round")
    if round_payload.get("restart_required"):
        raise HTTPException(
            status_code=409,
            detail="Restart with the same setup before finalizing this legacy round.",
        )
    session = sessions.load_session(session_id)
    context = dict(round_payload.get("context") or {})
    universe = str(context.get("universe") or session["universe"])
    as_of = str(context.get("as_of") or session["as_of"])
    resolved_panel = _panel_for_universe(universe, as_of, panel)
    metadata = dict(round_payload.get("round_metadata") or {})
    try:
        config = GPConfig.from_dict(metadata.get("config") or session["config"])
    except ValueError as exc:
        raise HTTPException(
            status_code=409,
            detail=f"round configuration is invalid: {exc}",
        ) from exc
    boundaries = sessions.Boundaries.from_dict(session["boundaries"])
    fingerprint = sessions.holdout_fingerprint(resolved_panel, boundaries, config)
    evaluation_id = uuid.uuid4().hex
    job_id = uuid.uuid4().hex
    claimed, reason = sessions.claim_finalization(
        session_id,
        job_id=job_id,
        evaluation_id=evaluation_id,
        round_index=round_index,
        holdout_fingerprint_value=fingerprint,
        confirm_repeat=req.confirm_repeat,
        strategy_plan_id=req.strategy_plan_id,
    )
    if not claimed:
        status = 404 if reason == "unknown round" else 409
        raise HTTPException(status_code=status, detail=reason)
    reservation = dict(
        sessions.load_session(session_id).get("active_finalization") or {}
    )
    resolved_strategy_plan_id = str(
        reservation.get("strategy_plan_id") or ""
    ) or None
    pinned_plan = (
        sessions.load_finalization_plan(session_id, resolved_strategy_plan_id)
        if resolved_strategy_plan_id is not None
        else None
    )

    progress = RunProgress(target_generations=0)
    cancel = threading.Event()

    def _task() -> dict[str, Any]:
        sessions.update_finalization_status(session_id, job_id, "running")
        try:
            result = sessions.finalize_round(
                session_id,
                round_index,
                job_id=job_id,
                evaluation_id=evaluation_id,
                panel=resolved_panel,
                strategy_plan_id=resolved_strategy_plan_id,
                progress=progress,
                stop=cancel.is_set,
            )
        except (TrainingCancelled, TrainingLeaseCancelled):
            sessions.finish_finalization_job(session_id, job_id, "stopped")
            raise
        except Exception as exc:
            sessions.finish_finalization_job(
                session_id,
                job_id,
                "failed",
                error=repr(exc),
            )
            raise
        return result

    try:
        _jobs.submit(
            _task,
            job_id=job_id,
            progress=progress,
            metadata={
                "session_id": session_id,
                "round_index": round_index,
                "evaluation_id": evaluation_id,
                "strategy_plan_id": resolved_strategy_plan_id,
                "comparison_id": (
                    pinned_plan.get("comparison_id")
                    if pinned_plan is not None
                    else None
                ),
                "primary_strategy_id": (
                    pinned_plan.get("primary_strategy_id")
                    if pinned_plan is not None
                    else None
                ),
                "kind": "session_finalization",
            },
            cancel=cancel,
        )
    except Exception as exc:
        sessions.finish_finalization_job(
            session_id,
            job_id,
            "failed",
            error=repr(exc),
        )
        raise
    return {
        "session_id": session_id,
        "round_index": round_index,
        "evaluation_id": evaluation_id,
        "strategy_plan_id": resolved_strategy_plan_id,
        "comparison_id": (
            pinned_plan.get("comparison_id") if pinned_plan is not None else None
        ),
        "primary_strategy_id": (
            pinned_plan.get("primary_strategy_id")
            if pinned_plan is not None
            else None
        ),
        "job_id": job_id,
        "status": "queued",
    }


@app.get("/sessions/{session_id}/rounds/{round_index}")
def get_session_round(session_id: str, round_index: int) -> dict[str, Any]:
    if not _session_exists(session_id):
        raise HTTPException(status_code=404, detail="unknown session")
    if round_index < 0:
        raise HTTPException(status_code=404, detail="unknown round")
    result = sessions.load_round(session_id, round_index)
    if result is None:
        raise HTTPException(status_code=404, detail="unknown or unavailable round")
    return {
        key: value
        for key, value in result.items()
        if key != "report_trials"
    }


@app.get("/sessions/{session_id}/finalizations")
def get_session_finalizations(session_id: str) -> list[dict[str, Any]]:
    if not _session_exists(session_id):
        raise HTTPException(status_code=404, detail="unknown session")
    return sessions.list_finalizations(session_id)


@app.get("/sessions/{session_id}/finalizations/{evaluation_id}")
def get_session_finalization(
    session_id: str,
    evaluation_id: str,
) -> dict[str, Any]:
    if not _session_exists(session_id):
        raise HTTPException(status_code=404, detail="unknown session")
    try:
        payload = sessions.load_finalization(session_id, evaluation_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail="unknown finalization") from exc
    if payload is None:
        raise HTTPException(status_code=404, detail="unknown finalization")
    return payload


@app.delete("/sessions/{session_id}")
def delete_session(session_id: str) -> dict[str, Any]:
    """Remove a session and everything that belongs to it.

    The agent conversation, its memory, and its tool-call history live inside the session
    directory precisely so this one removal takes them with it — there is no second cleanup path
    that could be forgotten. A running segment or finalization blocks the delete rather than
    being killed underneath itself.
    """
    if not _session_exists(session_id):
        raise HTTPException(status_code=404, detail="unknown session")
    session = sessions.load_session(session_id)
    if _live_session_job(session) is not None:
        raise HTTPException(status_code=409, detail="stop the running segment before deleting")
    for key in ("active_finalization_job_id", "active_strategy_comparison_job_id"):
        active = _jobs.get(str(session.get(key) or ""))
        if active is not None and active.status in {"queued", "running"}:
            raise HTTPException(
                status_code=409, detail="wait for the running holdout job before deleting"
            )

    directory = sessions.session_dir(session_id)
    had_conversation = (directory / "conversation.json").exists()
    shutil.rmtree(directory, ignore_errors=True)
    if directory.exists():
        raise HTTPException(status_code=500, detail="the session directory could not be removed")
    return {
        "deleted": session_id,
        "conversation_deleted": had_conversation,
    }


@app.post("/sessions/{session_id}/stop")
def stop_session(session_id: str) -> dict[str, bool]:
    if not _session_exists(session_id):
        raise HTTPException(status_code=404, detail="unknown session")
    session = sessions.load_session(session_id)
    job = _live_session_job(session)
    if job is not None and _jobs.cancel(job.id):
        return {"stopping": True}
    finalization_id = str(session.get("active_finalization_job_id") or "")
    if finalization_id and _jobs.cancel(finalization_id):
        return {"stopping": True}
    comparison_id = str(session.get("active_strategy_comparison_job_id") or "")
    if comparison_id and _jobs.cancel(comparison_id):
        return {"stopping": True}
    return {"stopping": False}


# --- P11: the agent ---------------------------------------------------------------
# One conversational surface. The model's reach is defined by the tool registry and nothing else:
# its evaluation tool takes no date argument, and the panel it is handed is truncated at the
# session's frozen train_end, so the validation window and the locked holdout are absent rather
# than merely forbidden (invariant 1). Every mutation stops at an explicit human action —
# promoting a factor, or applying a config patch.
_agent_jobs = JobStore()


class AgentBudgetRequest(BaseModel):
    max_tool_calls: int = Field(default=DEFAULT_MAX_TOOL_CALLS, ge=1, le=MAX_TOOL_CALLS_CEILING)
    max_evaluations: int = Field(
        default=DEFAULT_MAX_EVALUATIONS, ge=1, le=MAX_EVALUATIONS_CEILING
    )
    max_seconds: float = Field(default=DEFAULT_MAX_SECONDS, gt=0, le=MAX_SECONDS_CEILING)


class AgentMessageRequest(BaseModel):
    message: str = Field(min_length=1, max_length=8000)
    round_index: int | None = Field(default=None, ge=0)
    provider: str = ""
    model: str = ""
    base_url: str = ""
    budget: AgentBudgetRequest | None = None


def _conversation_store() -> ConversationStore:
    return ConversationStore()


@app.get("/agent/tools")
def list_agent_tools() -> dict[str, Any]:
    """The complete tool catalog. This is the agent's entire capability, shown before use."""
    return {
        "tools": agent_tools.catalog(),
        "tunable_config_keys": sorted(TUNABLE_KEYS),
        "protected_config_keys": dict(sorted(PROTECTED_KEYS.items())),
        "defaults": {
            "max_tool_calls": DEFAULT_MAX_TOOL_CALLS,
            "max_evaluations": DEFAULT_MAX_EVALUATIONS,
            "max_seconds": DEFAULT_MAX_SECONDS,
        },
        "unavailable_reason": agent_service.unavailable_reason(),
        "disclaimer": DISCLAIMER,
    }


@app.get("/agent/conversations")
def list_conversations() -> list[dict[str, Any]]:
    return _conversation_store().list()


@app.get("/agent/conversations/{session_id}")
def get_conversation(session_id: str) -> dict[str, Any]:
    """The thread for one training session. Empty rather than 404 when nothing has been said."""
    if not _session_exists(session_id):
        raise HTTPException(status_code=404, detail="unknown session")
    session = sessions.load_session(session_id)
    conversation = _conversation_store().load_or_create(
        session_id, str(session.get("name") or session_id)
    )
    return conversation.to_dict()


@app.delete("/agent/conversations/{session_id}")
def clear_conversation(session_id: str) -> dict[str, bool]:
    """Clear the thread without deleting the training session it belongs to."""
    if not _session_exists(session_id):
        raise HTTPException(status_code=404, detail="unknown session")
    return {"cleared": _conversation_store().delete(session_id)}


@app.post("/agent/conversations/{session_id}/messages", response_model=JobResponse)
def send_agent_message(
    session_id: str,
    req: AgentMessageRequest,
    panel: Panel | None = Depends(get_panel),  # noqa: B008
) -> JobResponse:
    """Send one message. The model decides which tools the request needs."""
    _load_persisted_formulas()
    if not _session_exists(session_id):
        raise HTTPException(status_code=404, detail="unknown session")
    session = sessions.load_session(session_id)
    universe = _resolve_universe(str(session.get("universe") or ""))
    as_of = str(session.get("as_of") or _today_iso())
    run_panel = _panel_for_universe(universe.name, as_of, panel)

    budget_request = req.budget or AgentBudgetRequest()
    try:
        context = agent_service.build_context(
            session_id,
            panel=run_panel,
            round_index=req.round_index,
            budget=AgentBudget(
                max_tool_calls=budget_request.max_tool_calls,
                max_evaluations=budget_request.max_evaluations,
                max_seconds=budget_request.max_seconds,
            ),
        )
    except (agent_service.AgentError, GuardViolation, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    config = explain_credentials.load_config(
        provider=req.provider, model=req.model, base_url=req.base_url
    )
    if not config.key.is_set:
        raise HTTPException(
            status_code=400,
            detail=(
                f"no API key configured for provider '{config.provider}'. Add one in Settings "
                "or set the provider's environment variable."
            ),
        )

    def _task() -> dict[str, Any]:
        return agent_service.send_message(
            session_id,
            context,
            message=req.message,
            provider=req.provider,
            model=req.model,
            base_url=req.base_url,
            store=_conversation_store(),
        )

    return JobResponse(job_id=_agent_jobs.submit(_task), status="queued")


@app.get("/agent/jobs/{job_id}")
def get_agent_job(job_id: str) -> dict[str, Any]:
    job = _agent_jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="unknown agent job")
    return {"job_id": job.id, "status": job.status, "error": job.error, "result": job.result}


@app.post("/agent/jobs/{job_id}/stop")
def stop_agent_job(job_id: str) -> dict[str, bool]:
    if _agent_jobs.get(job_id) is None:
        raise HTTPException(status_code=404, detail="unknown agent job")
    return {"stopping": _agent_jobs.cancel(job_id)}


@app.post(
    "/agent/conversations/{session_id}/proposals/{proposal_id}/promote",
    response_model=JobResponse,
)
def promote_agent_proposal(
    session_id: str,
    proposal_id: str,
    panel: Panel | None = Depends(get_panel),  # noqa: B008
) -> JobResponse:
    """Promote a staged factor into a real round.

    Runs the session's actual validation pass first — the agent only ever saw an inner holdout
    inside the training window, and a round carrying only that number would look like a peer of
    rounds that have been properly validated. Async because that validation is real work.
    """
    _load_persisted_formulas()
    if not _session_exists(session_id):
        raise HTTPException(status_code=404, detail="unknown session")
    session = sessions.load_session(session_id)
    universe = _resolve_universe(str(session.get("universe") or ""))
    as_of = str(session.get("as_of") or _today_iso())
    run_panel = _panel_for_universe(universe.name, as_of, panel)

    try:
        plan = agent_service.plan_promotion(session_id, proposal_id, store=_conversation_store())
    except agent_service.AgentError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    progress = RunProgress(target_generations=0)

    def _task() -> dict[str, Any]:
        return agent_service.promote(
            session_id,
            proposal_id,
            panel=run_panel,
            store=_conversation_store(),
            progress=progress,
        )

    del plan  # validated up front so a bad id is a 400 rather than a failed job
    return JobResponse(job_id=_agent_jobs.submit(_task, progress=progress), status="queued")


@app.post("/agent/conversations/{session_id}/proposals/{proposal_id}/apply-config")
def apply_agent_config(session_id: str, proposal_id: str) -> dict[str, Any]:
    """Merge a staged config patch into the session for the next segment."""
    try:
        return agent_service.apply_config(
            session_id, proposal_id, store=_conversation_store()
        )
    except agent_service.AgentError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


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
