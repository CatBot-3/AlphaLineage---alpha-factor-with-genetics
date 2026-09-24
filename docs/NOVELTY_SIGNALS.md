# Novelty training and formula-to-Signals workflow

Implementation and verification notes, 2026-09-23.

The functional workflow is implemented. **Performance acceptance remains open:** the full
95-formula reference catalog exceeds the requested 15% overhead ceiling on low-overlap searches.
The measurements below include the regressions, as well as the duplicate-heavy improvements.

## Training behavior

- New UI sessions use `novelty_mode: balanced`; omitted configuration and older sessions remain
  `off`. Continuing a session cannot change its novelty mode or frozen reference set.
- References contain expanded expressions, identities, dependency revisions and a fingerprint.
  Active formulas with complete default bindings and saved results are included. Unbindable or
  unevaluable references are recorded as skipped/unavailable.
- Canonical identity expands macros and uses only missing-value-safe simplification. Predictive
  scores are shared across equivalent expressions; each expression retains its complexity cost.
- Behavioral screening uses 128 deterministic training dates. The three closest known references
  above 0.60 absolute rank correlation receive full-training confirmation. Evolutionary family
  prototypes are confirmed separately. Changed name masks are reranked on their intersection.
- The multiplier is `1 - 0.5 * clamp((rho - 0.70) / 0.30, 0, 1)`. Raw IC, complexity cost and
  novelty deduction remain separate. Correlations of at least 0.98 identify a behavioral family.
- Population selection keeps a competitive family representative and allows an additional
  representative through the existing bounded stepping-stone mechanism. Subtrees remain usable.
- Validation sees the training-selected representatives and their frozen novelty multipliers.
  Every label-scored trial remains in statistical accounting, including discarded representatives.
  Similarity uses a physically truncated training panel, never validation or holdout prices.
- Checkpoints retain references, family prototypes, counters and deterministic selection state.
  Cancellation rolls back partially scored batches. Native scoring supplies candidate outputs
  directly; native pairwise rank comparisons have a tested Python fallback.

## Source identity and workflow

`api/formula_sources.py` resolves session rounds, opened evaluations, saved lineage nodes,
Library results and pinned reusable formula revisions into one frozen source representation.
It includes expression, bindings, revisions, execution, universe, horizon, polarity, evidence,
validation outcome, history origin, strategy and provenance.

`POST /formula-results` saves from that authoritative source. Repeated saves of the same identity
are idempotent. Legacy execution context is recovered only from an unambiguous matching round;
otherwise the UI requires the missing execution/direction choices for a portfolio preview.
Saving and applying do not finalize a round or increment holdout reads.

The navigation is Train, Results, Library, Signals, Agent and Build & Data. Existing workspace
tab values still restore. Results groups Overview, Formula and Lineage around the same selected
round. Training progress and completion actions remain visible after navigation without forcing
a page change. Training can prepare recoverable data gaps and start automatically after a fresh
coverage check. Restored progress uses persisted generations and actual checkpoint availability.

Library offers saved results and reusable formulas, source inspection, application, editing a
copy, search and training seeds. The save dialog preserves the selected source while its request
is pending. Saved status reconciles against backend identities. Failures remain distinct from
empty libraries. Signals labels Library copies separately from their original training rounds.

## Signals calculation contract

- Refresh & rank checks requirements, performs a bounded provider synchronization, rechecks
  coverage, evaluates and stores an immutable snapshot. Use cached data and offline settings
  avoid downloads. Navigation and freshness checks never fetch market prices.
- Requirements cover the displayed history and compounded formula warmup, with historical
  universe membership retained for cross-sectional calculations. Training's historical coverage
  rules remain strict. Membership entry/exit dates are never silently rewritten.
- XNYS supplies completed sessions, including holidays and early closes. A wholly stale cache
  cannot silently select an older, better-covered date. Reading a snapshot also checks the
  current completed session; the UI checks again on focus and every minute without recalculating.
- Rankings need at least 90% of active members and the formula's minimum-name count. Above the
  floor, partial coverage and exclusions are visible. Below it, ranking export and portfolio
  preview are blocked while diagnostic stock inspection remains available.
- Every source is evaluated over the relevant universe before a stock is extracted. Recursive
  and expanding formulas use their own recorded origin, independent of comparison formulas and
  chart zoom. Missing origins use a stable 2000-01-01 origin and are labelled approximate.
- Candles and inputs use adjusted prices. Daily candles, volume, raw/percentile comparisons,
  ranks, coverage, rankings and portfolio weights reference the same snapshot and data revision.
  Missing observations remain gaps. Historical inspection is explicitly retrospective.
- Portfolio previews reuse quantile long/short and rank-proportional engines. The pinned primary
  strategy is the default; otherwise the explicit default is a 20% quantile long/short portfolio.
  Notional amounts are indicative. Exports carry source, date, execution, coverage, universe,
  expression and data identities. No order submission or brokerage connection is implemented.

API additions: `POST /signals/resolve`, `/signals/prepare`, `/signals/history`,
`/signals/snapshot`, `/signals/jobs/{id}/stop`; `GET /signals/sources`,
`/signals/snapshots/{id}`, `/signals/snapshots/{id}/series/{symbol}`;
`POST /signals/snapshots/{id}/portfolio`; `GET /signals/snapshots/{id}/export`.
The existing factor endpoints remain available. Errors include a stable code, user message,
retryability, affected symbols and recovery action, with technical details kept separately.

Integration references: [Lightweight Charts attribution and API](https://tradingview.github.io/lightweight-charts/docs)
and [exchange_calendars sessions](https://github.com/gerrymanoim/exchange_calendars/blob/master/README.md).
The chart retains TradingView attribution; the pinned 5.0.9 license and notice ship in
`frontend/public/third-party`.

## Verification

- Full backend regression: **733 tests passed** after the native scoring, source, coverage,
  novelty and checkpoint changes. The subsequent freshness change passed **24 Signals tests**,
  including one new test for a cached snapshot becoming stale after another market close.
- All 13 novelty tests also passed with the evaluator explicitly set to Python; a dedicated
  parity test compares native and Python rank intersections with ties and missing names.
- Full frontend regression: **291 tests passed across 49 files**, including source handoff,
  failed save/retry, automatic prepare/start, delayed submit and stock-response races, and
  read-only market-freshness checks.
- Temporary parquet caches and mocked production providers cover missing-symbol repair,
  unavailable/IPO history, aliases, quotas, provider failures, partial coverage, offline use,
  holidays, early closes and wholly stale caches. No live provider allowance was consumed.
- Numerical checks cover chart/snapshot endpoints, cross-sectional context, independent recursive
  origins, pinned-strategy exports and equality with the existing portfolio engine.
- Browser walkthrough in an isolated 12-stock synthetic workspace verified training, navigation
  during training, completion review, failed-validation labels, save, Library inspection/application,
  cached ranking, stock selection, SMA comparison, percentile view and portfolio notional amounts.
  Reload preserved the snapshot, QA03 selection, 1Y range and percentile setting.
- Edge blocked the CSV download (`ERR_BLOCKED_BY_CLIENT`), so completion of a browser download
  was not verified. The HTTP export response, CSV content, metadata and portfolio weights are
  covered by backend tests. Browser restrictions were not bypassed.
- TypeScript checking and the production frontend build pass. New Python modules and dedicated
  novelty/Signals tests pass targeted Ruff checks. This is not a claim that unrelated pre-existing
  repository formatting issues have been resolved.

## Resource benchmark

Reproduce with `.venv\Scripts\python.exe scripts/benchmark_novelty.py --catalog --output result.json`.
Raw results: [novelty-2026-09-23.json](benchmarks/novelty-2026-09-23.json).

Each case starts a fresh process: 504 synthetic dates, 336 training dates, 100 candidates,
95 starter references, one worker and a 512 MiB training budget. Elapsed time covers GP/reference
initialization, scoring and three-fold validation, excluding import/catalog serialization,
production report generation and provider I/O. Peak RSS is sampled every 10 ms. These are
single-run local measurements, not statistical confidence intervals or production forecasts.

| Symbols | Workload | Off (s) | Balanced (s) | Time change | Validation candidates | Peak RSS MiB, off → balanced |
| --- | --- | ---: | ---: | ---: | --- | --- |
| 100 | duplicate-heavy | 3.79 | 2.17 | -42.8% | 100 → 1 | 165 → 175 |
| 100 | ordinary | 3.68 | 3.57 | -2.9% | 100 → 56 | 169 → 183 |
| 100 | low-overlap | 4.26 | 5.17 | +21.3% | 100 → 100 | 168 → 190 |
| 200 | duplicate-heavy | 5.13 | 3.72 | -27.5% | 100 → 1 | 174 → 194 |
| 200 | ordinary | 5.68 | 6.04 | +6.4% | 100 → 57 | 179 → 213 |
| 200 | low-overlap | 5.55 | 9.03 | +62.6% | 100 → 100 | 179 → 220 |
| 500 | duplicate-heavy | 12.67 | 10.30 | -18.7% | 100 → 1 | 201 → 252 |
| 500 | ordinary | 11.19 | 14.20 | +26.9% | 100 → 56 | 213 → 292 |
| 500 | low-overlap | 12.22 | 18.02 | +47.4% | 100 → 100 | 215 → 314 |

All these candidates have distinct structural identities: all 100 receive a first predictive
evaluation, and structural skips are zero. The duplicate-heavy case uses different monotone
expressions with identical rankings; its savings come from selection and validation. Tests
separately verify that macro/expanded structural equivalents share one predictive trial.

The duplicate-heavy speed goal is met in these cases. The low-overlap overhead ceiling is **not
met**, and the 500-name ordinary case also regresses. Balanced mode is selectable, and the
original `off` mode remains available. Further work should reduce reference preparation and
candidate screening cost without weakening train-only, missing-mask or reproducibility rules.

## Running the changed application

The frontend build, `lightweight-charts` dependency and local ABI 10 native extension have been
prepared. `exchange-calendars` is a backend dependency. Restart the application process to load
the changed backend and native extension, then reload the browser. Existing sessions preserve
their recorded novelty mode; create a new session to use balanced novelty.

For another checkout: install project dependencies, run `npm ci` in `frontend`, run
`npm run typecheck` and `npm run build:app`, and optionally rebuild the accelerator with
`.venv\Scripts\python.exe scripts/build_cpp.py --skip-dependency-install`. The Python fallback
remains supported. Existing user worktree changes, immutable reports and formulas were retained.

The isolated browser fixture is `scripts/workflow_qa_server.py`. Its module docstring gives
the separate frontend build location and fixed fixture date. It uses only synthetic QA symbols
and a separate `.runtime/workflow-verification/workflow-qa-data` store. Verification logs and
temporary artifacts are retained under `.runtime/workflow-verification`; they are not product data.
