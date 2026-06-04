# Evolutionary Alpha Factor Mining Platform — Development Plan

Version v0.1 | 2026-06-03 | Audience: Claude Code (autonomous coding agent) + human maintainer

---

## 0. How to work this plan (read first, agent)

1. Execute phases in order (Phase 0 → 8). Do not start a phase until the previous phase's acceptance gate passes (all listed tests green).
2. Each task has an ID (e.g. `P1-T3`). Treat one task as one branch/PR-sized unit of work. Write the acceptance test before or alongside the implementation.
3. The acceptance criteria are expressed as concrete, runnable checks. "Done" means the named test exists and passes, not "looks done."
4. The invariants in Section 2 are hard constraints. Never weaken a guardrail (for example, never evaluate a factor's reported metric on the locked test set) just to make a test pass. If an invariant blocks progress, stop and surface it rather than working around it.
5. If blocked on external credentials or data (Tiingo API key, network), stub the boundary, mark a `TODO(human)`, and keep building everything testable behind the stub with synthetic data.
6. Keep `CLAUDE.md` (Appendix A) at the repo root and updated. It is the agent's standing context.

This plan is intentionally testable end to end without market data: most logic is validated against synthetic panels. The two load-bearing tests are the synthetic-signal recovery test (`P2-T6`) and the noise-rejection test (`P3-T6`).

---

## 1. Project overview

An evolutionary (genetic-programming) alpha-factor discovery platform. Users define a stock universe and an extensible operator/operand set. The system encodes trading strategies (alpha factors) as strongly-typed expression trees, evolves them with genetic programming (point mutation, subtree mutation, crossover), scores them by information coefficient (IC) and risk-adjusted metrics, and runs a built-in anti-overfitting validation suite. A web UI lets users step through each generation, inspect any factor's tree and parameters, and edit it via drag-and-drop. An optional live layer feeds real-time data and pushes buy/sell prompts.

Lineage and positioning: this is the GP/RL alpha-mining tradition (Allen & Karjalainen 1999 → gplearn → AlphaGen / AlphaForge). The differentiator is interpretable, editable formulaic alphas with rigorous anti-overfitting controls as a first-class feature.

---

## 2. Core invariants (hard constraints)

These govern every phase and outrank feature convenience.

1. **Overfitting is the primary adversary.** The test split is locked for the entire lifecycle of a factor. Any metric shown to a user defaults to out-of-sample / deflated. The number of trials (including search-space growth from user-added primitives) feeds the deflated Sharpe correction.
2. **Encode strategies as strongly-typed expression trees, never as flat numeric vectors.** Useful signals come from cross-series interactions (correlations, spreads, conditionals) that are not additively separable, so a flat per-series weighted-sum encoding cannot represent them. Trees can; keep them typed so every generated/mutated tree is semantically valid.
3. **GP must use a population plus crossover, not single-point hill climbing.** Point mutation alone is a `(1+1)` strategy that gets stuck. Crossover is the main source of search power.
4. **Fitness is IC (or rank IC / IC IR), not raw PnL.** Lower noise, more stable, closer to modern practice.
5. **User-extensible operators go through a constrained vectorized DSL plus a declared type signature.** No arbitrary server-side code execution. Any user code runs client-side or in a strict sandbox and must be vectorized.
6. **Costs are real.** Backtests include transaction-cost and slippage models; a signal that does not survive realistic costs is flagged unusable.
7. **Data integrity.** Universe membership is point-in-time (no survivorship bias); corporate actions are adjusted correctly; free-tier data limitations are surfaced, not hidden.
8. **This is not investment advice and not a brokerage.** Outputs carry disclaimers. No promise of beating the market.

---

## 3. Tech stack

1. Language: Python 3.11+.
2. Packaging/deps: `uv` (fallback `pip`), single `pyproject.toml`.
3. Numerics: `numpy`, `pandas`, `numba` (hot-path vectorized operators), `scipy`/`statsmodels` (stats for validation layer).
4. GP core: custom strongly-typed tree engine (the type registry, vectorized evaluator, lineage tracking, and anti-overfitting hooks are bespoke anyway). `DEAP` may back strongly-typed primitives if helpful; `gplearn` is a reference baseline only (it is weakly typed).
5. Data: Tiingo (`tiingo` Python client or direct `requests`), `pyarrow` for Parquet cache. Optional: Microsoft `qlib` bundled dataset for fast bootstrap; `yfinance`/`stooq` as zero-key prototype fallbacks (with survivorship-bias warnings).
6. Backend (Phase 5+): `FastAPI` + `uvicorn`; job queue `RQ` (or Celery) + Redis; `PostgreSQL` + `SQLAlchemy` + `alembic`.
7. Frontend (Phase 6+): React + Vite + React Flow (tree and node editor) + a charting lib.
8. Quality gates: `pytest` (+ `pytest-cov`), `ruff` (lint + format), `mypy` or `pyright` (types), `hypothesis` (property tests for the tree engine).

---

## 4. Proposed repository layout

```
alpha-forge/
  pyproject.toml
  README.md
  CLAUDE.md
  .env.example
  src/alphaforge/
    data/            # Phase 0
      tiingo_client.py
      cache.py
      adjust.py
      universe.py
    core/            # Phase 1-2
      types.py       # the type system (Series, Scalar, Window, Signal)
      primitives.py  # registry: operators/operands + signatures + impls
      tree.py        # Node, (de)serialization
      generate.py    # ramped half-and-half, strongly-typed random trees
      evaluate.py    # vectorized evaluator
      simplify.py    # algebraic simplification / anti-bloat
      fitness.py     # IC / rank IC / IC IR
      gp.py          # population, selection, crossover, mutations, loop
    validation/      # Phase 3
      splits.py      # train/valid/test + embargo, walk-forward
      purged_cv.py
      deflated_sharpe.py
      pbo.py
      trials.py      # global trial counter
    backtest/        # Phase 4
      portfolio.py   # rank -> long/short, neutralization
      costs.py
      metrics.py     # Sharpe, max drawdown, turnover
    library/         # Phase 5
      store.py       # lineage (parent + diff) persistence
      diversity.py   # correlation-based pruning
      combine.py     # mega-alpha
    api/             # Phase 5
      app.py
      jobs.py
    live/            # Phase 8
      feed.py
      signal.py
      notify.py
  frontend/          # Phase 6 (React + Vite)
  tests/
  scripts/
    download_universe.py
    run_gp.py
  data_cache/        # gitignored Parquet
```

---

## 5. Dev environment and commands

```bash
# setup
uv venv && source .venv/bin/activate
uv pip install -e ".[dev]"

# quality gates (must all pass before any phase gate is considered green)
pytest -q --cov=alphaforge
ruff check . && ruff format --check .
mypy src            # or: pyright

# bootstrap data (Phase 0+, needs TIINGO_API_KEY in .env)
python scripts/download_universe.py --universe sp500-lite --years 15

# run a GP search (Phase 2+)
python scripts/run_gp.py --config configs/dev.yaml
```

Convention: every phase gate is the union of (a) the phase-specific acceptance tests below and (b) the three standing gates above (pytest, ruff, types) passing.

---

## 6. Phased roadmap with acceptance tests

Each phase lists goal, tasks, and acceptance criteria as named tests. MVP is Phases 0 to 4.

### Phase 0 — Data foundation

Goal: reliable, cached, correctly-adjusted, point-in-time universe data.

Tasks:
1. `P0-T1` Tiingo client with retry/backoff and rate-limit awareness.
2. `P0-T2` Parquet local cache; second fetch served from disk.
3. `P0-T3` Corporate-action adjustment (splits/dividends) producing a continuous adjusted series.
4. `P0-T4` Universe builder recording per-symbol entry/exit dates (point-in-time).
5. `P0-T5` Survivorship-bias audit report generator (documents coverage of delisted names).

Acceptance (`tests/test_data.py`):
1. `test_cache_roundtrip` — first call hits API (mocked), second call hits cache, identical frame.
2. `test_split_continuity` — across a known split date, the adjusted close has no discontinuity beyond tolerance.
3. `test_universe_point_in_time` — querying the universe as-of a past date excludes symbols that entered later.
4. `test_survivorship_report` — report enumerates active vs delisted coverage and is non-empty.

### Phase 1 — Expression-tree engine + random generation

Goal: encode strategies as typed trees; generate, evaluate (vectorized), and simplify them.

Tasks:
1. `P1-T1` Type system: `Series`, `Scalar`, `Window(int)`, `Signal`.
2. `P1-T2` Primitive registry: arithmetic, unary/binary time-series (`ts_mean`, `ts_std`, `ts_corr`, `delta`, `delay`), cross-sectional (`rank`, `zscore`), unary math (`log`, `abs`, `sign`); operands OHLCV, vwap, returns, scalar constant, window int. Each primitive declares arity, input types, output type.
3. `P1-T3` `Node` data structure + lossless (de)serialization (JSON).
4. `P1-T4` Strongly-typed random generator with ramped half-and-half; respects max depth/size.
5. `P1-T5` Vectorized evaluator over a panel.
6. `P1-T6` Algebraic simplifier (collapses `x+0`, `x*1`, nested no-ops).

Acceptance (`tests/test_tree_*.py`, use `hypothesis` for fuzzing):
1. `test_random_trees_all_valid` — generate 10k trees; 100% are type-valid and evaluate without error on a synthetic panel.
2. `test_evaluator_matches_reference` — evaluator output equals a hand-computed reference for 3 known factors.
3. `test_depth_size_constraints` — no generated tree exceeds configured depth/node limits.
4. `test_serialization_roundtrip` — tree → JSON → tree is structurally identical and evaluates identically.

### Phase 2 — GP loop + fitness

Goal: full GP search with IC fitness.

Tasks:
1. `P2-T1` Population init (ramped half-and-half).
2. `P2-T2` Fitness: train-set IC / rank IC.
3. `P2-T3` Tournament selection.
4. `P2-T4` Crossover (subtree swap, type-safe).
5. `P2-T5` Point mutation (tweak a constant/window) and subtree mutation.
6. `P2-T6` Anti-bloat (node-count penalty + simplify before display).
7. `P2-T7` Checkpointable, interruptible run loop with a time/generation budget.

Acceptance (`tests/test_gp.py`):
1. `test_synthetic_signal_recovery` (load-bearing) — on a synthetic panel where future returns are a known function of features, GP recovers a factor whose IC against the injected signal exceeds a high threshold.
2. `test_train_ic_improves` — best train IC is non-decreasing across generations and converges.
3. `test_crossover_type_safe` — 10k crossovers never produce an invalid tree.
4. `test_checkpoint_resume` — interrupting and resuming yields the same population state.

### Phase 3 — Anti-overfitting validation (critical gate)

Goal: an honest out-of-sample and statistical-significance verdict for any factor.

Tasks:
1. `P3-T1` train/valid/test split with embargo; walk-forward windows.
2. `P3-T2` Purged k-fold CV.
3. `P3-T3` Deflated Sharpe Ratio.
4. `P3-T4` PBO via combinatorially symmetric cross-validation (CSCV).
5. `P3-T5` Global trial counter wired into the deflation.
6. `P3-T6` Pipeline that re-judges any Phase-2 "best" factor out-of-sample.

Acceptance (`tests/test_validation.py`):
1. `test_noise_rejection` (load-bearing) — feed pure-noise data; the "best" selected factor returns a high PBO and a non-significant deflated Sharpe.
2. `test_deflated_sharpe_known_values` — implementation matches reference values from the López de Prado formulas within tolerance.
3. `test_pbo_bounds` — PBO is in [0,1] and increases as the number of trials grows on noise.
4. `test_test_set_never_touched` — an assertion/guard fires if any reported metric is computed on the locked test split before final reporting.

### Phase 4 — Backtest + portfolio (with costs)

Goal: turn factor scores into a realistic portfolio result.

Tasks:
1. `P4-T1` Rank → long/short portfolio; sector/size neutralization.
2. `P4-T2` Transaction-cost and slippage model.
3. `P4-T3` Metrics: Sharpe, max drawdown, turnover, decay.

Acceptance (`tests/test_backtest.py`):
1. `test_backtest_matches_independent_recompute` — equals an independent vectorized recomputation on a sample.
2. `test_cost_sensitivity` — a signal that is profitable gross but unprofitable net of realistic costs is flagged unusable.
3. `test_turnover_reported` — turnover is computed and within expected bounds.

MVP gate (end of Phase 4): a CLI/notebook single-factor discovery engine where `test_synthetic_signal_recovery` and `test_noise_rejection` both pass and backtests include costs.

### Phase 5 — Backend + factor library (multi-factor)

Goal: service the engine; go from single factor to a synergistic factor set.

Tasks:
1. `P5-T1` FastAPI app; GP runs dispatched to a job queue.
2. `P5-T2` Lineage persistence (parent + diff), replayable.
3. `P5-T3` Diversity pruning by pairwise correlation.
4. `P5-T4` Combination model (mega-alpha).

Acceptance (`tests/test_library.py`, `tests/test_api.py`):
1. `test_lineage_replay` — a stored run reconstructs every generation from the persisted lineage.
2. `test_diversity_threshold` — after pruning, pairwise factor correlation is below the configured threshold.
3. `test_combo_beats_best_single` — the combined factor's deflated metric beats the best single factor's.
4. `test_job_lifecycle` — submit → run → fetch result via the API.

### Phase 6 — Frontend visualization

Goal: step into each generation, open any factor to see its composition and parameters, on honest metrics.

Tasks:
1. `P6-T1` Tree visualization + factor detail (node params).
2. `P6-T2` Evolution genealogy timeline.
3. `P6-T3` Metrics dashboard defaulting to OOS/deflated values.

Acceptance:
1. `test_factor_renders` (component test) — any stored factor renders as a structurally correct tree.
2. `test_genealogy_navigable` — UI can trace parent → operation → child.
3. `test_dashboard_shows_deflated` — the default metric shown is the deflated/OOS one (guardrail visible in UI).
4. One usability pass with notes recorded.

### Phase 7 — User extensibility

Goal: user-defined universe and user-added operators/operands.

Tasks:
1. `P7-T1` Universe-definition UI (point-in-time aware).
2. `P7-T2` Operator-registration UI: a type-signature form composed from the constrained vectorized primitive set (no raw server code).
3. `P7-T3` Search-space / trial accounting wired through to deflated Sharpe.

Acceptance (`tests/test_extensibility.py`):
1. `test_user_operator_valid_immediately` — a user operator with a valid signature is usable in generation/mutation and never yields an invalid tree.
2. `test_no_arbitrary_code_path` — security review check: no code path executes user-supplied server-side code.
3. `test_trial_count_updates` — adding primitives increases the trial count and shows up in the deflation.

### Phase 8 — Live signals + push (optional / stretch)

Goal: real-time fetch, signal generation, notifications.

Tasks:
1. `P8-T1` Scheduled data pull.
2. `P8-T2` Signal pipeline identical to the backtest pipeline.
3. `P8-T3` Mobile/email notifications.
4. `P8-T4` Compliance gate + disclaimers.

Acceptance (`tests/test_live.py`):
1. `test_point_in_time_consistency` (load-bearing) — on historical replay, the live pipeline's signal for a given date exactly matches the backtest signal (no look-ahead).
2. `test_notification_delivery` — a generated signal triggers a delivered notification (mocked transport).
3. Compliance checklist confirmed (personal-use disclaimer; commercial data license required before any public offering).

---

## 7. Data source decision

Recommended: Tiingo free tier. Free access to 30+ years of EOD prices, broad coverage (US + Chinese equities, ETFs/funds/ADRs), a corporate-actions API, and proprietary data cleaning.

Constraints to enforce in code:
1. Free tier is roughly 50 symbols/hour with a monthly unique-symbol cap (historically ~500/month; the vendor states limits are approximate and may change). Usage pattern must be: fetch the universe once → cache locally as Parquet → never call the API during the GP search.
2. Free tier is personal use; commercial/public deployment requires a paid organization plan and possibly a redistribution license.
3. Free fundamentals are ~5 years only.
4. No guarantee of survivorship-bias-free or point-in-time fundamentals; treat free data as prototype-grade.

| Source | Use | Pros | Limits |
|---|---|---|---|
| Tiingo (recommended) | Primary EOD prices | Long history, cleaned, corporate actions | Monthly symbol cap; free is non-commercial |
| Qlib bundled data | Fastest algo bring-up | Ready aligned panel, fits this workflow | Limited coverage/refresh |
| yfinance / stooq | Zero-key prototype | Frictionless | Survivorship bias, unstable quality |
| Paid (EODHD / Polygon / FMP) | Production | Long history, delisted, point-in-time | Costs money |

Data quality checklist (gate before any production use): sufficient history (10+ years), broad cross-section, includes delisted names, point-in-time fundamentals, correct corporate-action adjustment.

---

## 8. Milestones / definition of done

1. MVP (Phases 0 to 4): honestly-validated single-factor engine (CLI/notebook). Done when synthetic-signal recovery and noise-rejection tests pass and backtests include costs.
2. V1 (MVP + Phases 5 to 7): web app + factor library + user-extensible + interactive UI. Done when lineage replays, combination beats the best single factor, and user operators never produce invalid trees with no arbitrary-code path.
3. V2 (V1 + Phase 8): live signals and push, behind compliance and commercial-data-license gates. Done when point-in-time consistency passes and the compliance checklist is confirmed.

---

## 9. References

1. López de Prado, Advances in Financial Machine Learning (2018) — purged CV, PBO, anti-overfitting practice.
2. Allen & Karjalainen (1999), Using Genetic Algorithms to Find Technical Trading Rules, Journal of Financial Economics 51, 245–271.
3. Bailey, Borwein, López de Prado & Zhu (2014), Pseudo-Mathematics and Financial Charlatanism, Notices of the AMS 61(5) (SSRN 2308659).
4. Bailey & López de Prado (2014), The Deflated Sharpe Ratio, Journal of Portfolio Management 40(5) (SSRN 2460551).
5. Yu et al. (2023) AlphaGen (KDD 2023); Shi et al. (2024) AlphaForge (AAAI); Kakushadze (2016) 101 Formulaic Alphas.

Tools: gplearn (reference baseline), DEAP (strongly-typed GP), Microsoft Qlib (data + operators + backtest), Tiingo (`tiingo` client), React Flow / Rete.js (frontend tree editor).

---

## Appendix A — Suggested `CLAUDE.md` (drop at repo root)

```markdown
# CLAUDE.md

## What this is
Evolutionary (genetic-programming) alpha-factor mining platform. Strategies are
strongly-typed expression trees, evolved via GP, scored by IC, validated with a
built-in anti-overfitting suite.

## Invariants (never violate)
1. The test split is locked. Any user-facing metric defaults to OOS/deflated.
2. Encode strategies as typed expression trees, never flat numeric vectors.
3. GP uses a population + crossover, not single-point hill climbing.
4. Fitness is IC / rank IC, not raw PnL.
5. User operators go through a typed, vectorized DSL. No arbitrary server-side code.
6. Backtests include transaction costs; signals must survive them.
7. Universe is point-in-time; corporate actions adjusted; no survivorship bias.
8. Not investment advice, not a brokerage. Disclaimers on outputs.

## Commands
- Setup: `uv venv && uv pip install -e ".[dev]"`
- Test:  `pytest -q --cov=alphaforge`
- Lint:  `ruff check . && ruff format --check .`
- Types: `mypy src`
- Data:  `python scripts/download_universe.py --universe sp500-lite --years 15`
- Run:   `python scripts/run_gp.py --config configs/dev.yaml`

## Conventions
- Python 3.11+. One task = one PR-sized change. Write the acceptance test first.
- Hot paths (tree evaluation) must be vectorized (numpy/pandas/numba).
- Do not call the data API inside the GP loop; read from the Parquet cache.
- The two load-bearing tests are synthetic-signal recovery and noise rejection.
  If either regresses, stop and fix before anything else.

## Phase order
Phases 0 → 8 in DEVELOPMENT_PLAN.md. Do not start a phase until the prior gate is green.
```
