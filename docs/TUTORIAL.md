# AlphaLineage — Tutorial

This is the from-scratch guide to AlphaLineage. It explains:

1. **What each piece does** and why it was built that way.
2. **How the pieces fit together** — one end-to-end walk of a single search.
3. **How to do everything from the browser** — define a universe, add a ticker, add a computation operator, train, watch it, continue, save factors, seed new searches, and save/load your work.

> **Not investment advice.** AlphaLineage is a research tool. Nothing it produces is a recommendation to trade.

---

## 0. In General

AlphaLineage looks for **alpha factors**: formulas that, computed across many stocks each day, rank the stocks so that tomorrow's winners tend to score higher than tomorrow's losers. 

Instead of guessing formulas, a **genetic-programming** search breeds thousands of them, keeps the ones that rank well, and recombines them.

AlphaLineage is built to prevent overfitting: data will be sliced in training and testing parts, and the testing section is used to prevent overfitting of the alpha factors. And use Deflated indexes to correct for the time of trial.

---

## 1. The components

### 1.1 Data layer
Source: [`src/alphalineage/data/`](../src/alphalineage/data/)

- **Cache (`cache.py`, `paths.py`).** Prices are downloaded once and stored as Parquet under
  `data_cache/prices/{SYMBOL}.parquet`. The search reads only from this cache and never calls a data
  API inside the loop, which keeps runs fast, reproducible, and offline-capable.
- **Point-in-time universe (`universe.py`).** A *universe* is the set of symbols a search ranges
  over. Each member carries an **entry** and **exit** date, so a query for the constituents *as of*
  a given date returns exactly the symbols listed on that date. This is what avoids **survivorship
  bias** — scoring a strategy only on the names that happened to survive to the present.
- **Adjustment (`adjust.py`).** Splits and dividends are applied so corporate actions do not appear
  as price jumps.
- **Providers (`provider.py`, `tiingo_client.py`, `yfinance_provider.py`).** Tiingo is the primary
  source (free key); yfinance is the fallback. A key is required only to download new data — the
  bundled cache runs without one.

Data is reshaped into a **Panel** ([`core/panel.py`](../src/alphalineage/core/panel.py)): one aligned
matrix per field (`close`, `volume`, …) of shape *dates × symbols*. Factors are evaluated over these
matrices at once (vectorized NumPy/pandas), which is the basis of the engine's throughput.

### 1.2 Typed expression trees
Source: [`core/types.py`](../src/alphalineage/core/types.py),
[`core/primitives.py`](../src/alphalineage/core/primitives.py),
[`core/tree.py`](../src/alphalineage/core/tree.py)

A factor is encoded as a **typed expression tree** — internal nodes are operators, leaves are price
fields or constants — rather than a flat numeric vector. For instance `rank(ts_mean(close, 20))`
denotes the cross-sectional rank of each symbol's 20-day mean close.

The type system has four types: `SERIES` (a value per symbol per day), `SIGNAL` (a rank-ready
series), `SCALAR` (a number), and `WINDOW` (a look-back length). Every primitive declares its
argument types and output type, and `validate()` enforces arity and subtype compatibility at every
node. Consequently the generator and the variation operators can only ever produce **well-typed**
trees — a malformed expression such as `ts_mean(window, window)` is unconstructible.

This typing is also the safety boundary for user extensions (§3.3): a custom operator is submitted
as a **typed body tree** referencing only existing primitives and `$arg` placeholders, type-checked
on registration, and expanded into a built-in-only tree before evaluation. There is no `eval`,
`exec`, or compilation path — a user operator is data, never server-side code.

Trees are immutable and serialize losslessly to JSON, which is what makes lineage persistence, saved
factors, and bit-for-bit run replay possible.

### 1.3 The GP loop
Source: [`core/gp.py`](../src/alphalineage/core/gp.py),
[`core/generate.py`](../src/alphalineage/core/generate.py),
[`core/fitness.py`](../src/alphalineage/core/fitness.py)

The search is a standard generational genetic program:

1. **Initialize** a population by ramped half-and-half generation (or from supplied seed trees).
2. **Score** each individual by its **information coefficient (IC)**. For each date the factor is
   correlated cross-sectionally with the next period's return (Spearman rank correlation by default);
   the per-date series is averaged. Fitness is **mean |rank IC| − λ · tree size**, where the
   parsimony coefficient λ penalizes bloat. IC — not realized PnL — is the objective because PnL is
   substantially easier to overfit (invariant 4). A breadth floor (`min_names`) and a minimum number
   of valid dates reject factors that earn a spurious IC on a near-empty cross-section.
3. **Select** parents by tournament (sample *k*, keep the fittest).
4. **Vary**: type-safe subtree **crossover**, plus **subtree** and **point mutation**; the top
   `elitism` individuals carry over unchanged. Depth and node-count caps are enforced by
   construction.
5. Iterate for *N* generations. The loop is driven by a single seeded RNG (deterministic) and is
   **checkpointable** — a checkpoint captures the RNG state, population, history, and trial count, so
   a run resumes bit-for-bit. This is the mechanism behind iterative *sessions* (§1.7).

### 1.4 Validation — the anti-overfitting suite
Source: [`validation/`](../src/alphalineage/validation/)

This subsystem produces the honest verdict and is non-negotiable.

- **Locked split (`splits.py`, `pipeline.py`).** The date index is partitioned, in time order, into
  **train / validation / test** with an **embargo** gap of *E* trading days between adjacent
  segments — the embargo prevents a trailing-window factor or a forward-return label from straddling
  a boundary and leaking. The **test** segment is wrapped in a `LockedTestSet`: any attempt to read
  it before final reporting raises. It is unlocked exactly once, to compute the out-of-sample IC.
- **Deflated Sharpe ratio (`deflated_sharpe.py`, `trials.py`).** Selecting the best of many trials
  inflates the winner's apparent Sharpe. The DSR discounts the observed Sharpe by the expected
  maximum under the null, given the number of trials *N* and the variance of the trials' Sharpe
  ratios. The trial count is further inflated for the size of the search space:
  `N_eff = round(N · (n_operators / baseline)^2)`, so enlarging the operator palette — including with
  user operators — makes a lucky result strictly harder to certify.
- **PBO (`pbo.py`).** The Probability of Backtest Overfitting, estimated by combinatorially symmetric
  cross-validation: across folds, the frequency with which the in-sample-best configuration
  underperforms out-of-sample.

A factor is reported **significant** only when `DSR > 0.95` **and** `PBO < 0.5`. The dashboard
presents OOS IC, deflated Sharpe, PBO, and a plain-language verdict as the headline; in-sample (train)
figures are demoted to a collapsed "for reference only" section. The ordering is deliberate — the
flattering number is never the default (invariant 1).

### 1.5 Backtest with costs
Source: [`backtest/`](../src/alphalineage/backtest/)

A factor's scores are mapped to a portfolio — long the top quantile, short the bottom — optionally
neutralized, with **transaction costs and slippage** (in basis points) deducted. A signal that
survives only gross of costs is not counted. Reported metrics include Sharpe, maximum drawdown,
turnover, and IC decay.

### 1.6 Factor library & lineage
Source: [`library/store.py`](../src/alphalineage/library/store.py),
[`library/factors.py`](../src/alphalineage/library/factors.py)

- **Lineage (`store.py`).** Each individual the GP produces is recorded with its id, generation, the
  variation operator that created it, its parent ids, and its fitness. The record is replayable
  generation by generation and is the data behind the Genealogy view.
- **Saved factors (`factors.py`).** A kept factor is written as a self-contained JSON document: the
  tree, its metrics, its **provenance** (originating session, generation, and universe, plus the
  cumulative trial and out-of-sample-read counts at save time), and the **specifications of any user
  operators the tree references**. Because the operator definitions travel with the factor, seeding a
  new search from it is reproducible on a fresh machine. The storage directory is configurable
  (§3.13).

### 1.7 Backend: API, jobs, and sessions
Source: [`api/`](../src/alphalineage/api/)

- **FastAPI app (`app.py`).** A lightweight local service — no external servers; state is the Parquet
  cache plus JSON/Parquet files.
- **Job runner (`jobs.py`).** Searches execute on a background thread; the client polls for progress
  and may request a stop, which the GP honors after its current generation.
- **Sessions (`sessions.py`).** A **session** is a search grown over **segments**. After the initial
  segment, a session may be **continued** — warm-starting from the evolved population with changed
  hyperparameters, a changed universe, or newly registered operators — or a fresh session may be
  **seeded** from previously saved factors. Three honesty invariants hold across segments:
  - **The time boundary is frozen at creation.** The split dates `train_end`, `valid_start`,
    `valid_end`, `test_start` (and the embargo) are recorded once. Every later segment rebuilds its
    split by filtering the *current* panel against those frozen dates — `train = {t ≤ train_end}`,
    `valid = [valid_start, valid_end]`, `test = {t ≥ test_start}` — so even a universe change cannot
    relocate the boundary, and the GP is never handed a date ≥ `test_start` (asserted in the runner).
  - **Trial counts are cumulative and monotone.** They are carried through the GP checkpoint and
    summed across segments and across the provenance of any seed factors; the deflation only ever
    tightens.
  - **Each segment is one out-of-sample read.** The session counts reads and surfaces the count, so
    repeated peeking at the locked set is visible rather than silent.

### 1.8 Frontend
Source: [`frontend/src/`](../frontend/src/)

A React + Vite application with two build targets:

- **`app`** communicates with the local backend for real searches. This is what `docker compose up`
  serves and what `npm run dev:app` runs.
- **`demo`** is a static, backend-free snapshot of one finished run (metrics, factor tree,
  genealogy), suitable for deployment to a static host.

The UI is organized into tabs — **Train, Metrics, Best factor, Genealogy, Library, Extend** — which
§3 covers in turn.

### 1.9 Optional C++ accelerator
Source: [`cpp/`](../cpp/), [`core/cpp.py`](../src/alphalineage/core/cpp.py)

The evaluator's hot path has an optional pybind11/CMake backend. Pure Python is the default and the
correctness baseline; when the compiled extension is present it engages automatically and is pinned
identical to Python by a parity test. Set `ALPHALINEAGE_EVALUATOR=python` to force the baseline.

---

## 2. End-to-end: the life of one search

The sequence triggered by **Start training**, traced through the components:

1. **Universe → symbols.** The selected universe is resolved as of the search date to a point-in-time
   symbol list.
2. **Symbols → Panel.** The symbols' cached prices are loaded and aligned into *dates × symbols*
   matrices. No network access; no survivorship bias.
3. **Panel → frozen split.** The session computes the train/validation/test boundaries once and
   records them; the test segment is locked.
4. **Train slice → GP.** A population of typed trees is evolved over the *train* dates only — scored
   by IC, selected, recombined — for the configured number of generations. Every individual is
   written to the lineage with its fitness, and progress streams to the UI.
5. **GP → judgment.** The best factor is scored on the **validation** dates; the **deflated Sharpe**
   is computed against the cumulative (effective) trial count and the **PBO** across folds; then the
   **test** segment is unlocked exactly once and the out-of-sample IC is read.
6. **Judgment → dashboard.** The headline is the out-of-sample / deflated verdict, alongside the
   factor's tree and its genealogy. Nothing observed during the search touched the test segment until
   step 5.
7. **Iterate.** Continue the session from the evolved population, or save the best factor and seed a
   new session from it. Trial and out-of-sample-read counts carry forward, so the honesty accounting
   only tightens.

---

## 3. How to do everything from the browser

Open the app:
- **Docker:** `docker compose up`, then **http://localhost:8000**.
- **Dev:** backend `uvicorn alphalineage.api.app:app --port 8000`, frontend `cd frontend && npm run
  dev:app`, then **http://localhost:5173**.

The **Train / Library / Extend** tabs require the backend and are disabled in the static demo.

### 3.1 Define and store a universe
**Extend** tab → **Define a universe (point-in-time)** panel.
1. Enter a **Name** (e.g. `my-tech`).
2. Each row is one symbol: fill **Symbol** (e.g. `AAPL`), **Entry** (e.g. `2000-01-03`), and
   optionally **Exit** (blank means still active).
3. Use **+ Add symbol** for additional rows; **Remove** deletes one.
4. Click **Define universe**. A confirmation lists the symbols. The universe is persisted (Parquet
   under `data_cache/universe/`) and appears in the Train tab's **Universe** dropdown.

The entry/exit dates make the universe point-in-time, so a search over it cannot reference a symbol
before its listing or after its delisting.

### 3.2 Add a ticker to a universe
In the same panel, add a row (**+ Add symbol**) with the new **Symbol** and its **Entry** date, then
click **Define universe** to re-save the universe with the added member.

> The ticker's price history must be present in the cache. If it is not, download it once with
> `python scripts/download_universe.py --universe sp500-lite --years 15` (requires a Tiingo key in
> `.env`). The bundled cache already covers the `sp500-lite` names.

### 3.3 Add a custom computation operator
**Extend** tab → **Compose an operator** panel. An operator is a reusable, typed macro — submitted as
a graph, never as code.
1. **Name** the operator (e.g. `momentum`).
2. **Args**: a comma-separated list of input types, e.g. `series,window`.
3. **Output**: the result type (`series`, `signal`, `window`, `scalar`).
4. Build the body from the **palette**:
   - **`$arg 0`, `$arg 1`, …** are the argument placeholders (one per declared argument).
   - **const** and **window** are literal leaves.
   - The remaining buttons are existing primitives (e.g. `ts_mean`, `rank`, `sub`).
   Click a palette button to add a node, then drag from one node's handle to another to connect a
   child into a parent. For `momentum(series, window) = rank(ts_mean($arg0, $arg1))`: add `rank`,
   `ts_mean`, `$arg 0`, `$arg 1`; connect `$arg 0` and `$arg 1` into `ts_mean`, and `ts_mean` into
   `rank`.
5. Click **Register**. On success the signature is confirmed and the operator — now type-checked —
   becomes available to the GP and appears in the operator palette. A malformed graph (no single
   output, or a type mismatch) is rejected with an explanatory message and nothing is registered.

### 3.4 Start a training run
**Train** tab.
1. **Session name**.
2. **Universe** — select from the dropdown (custom universes appear here).
3. Set the core hyperparameters: **Population**, **Generations**, **Max depth**, **Max nodes**,
   **Seed**. The defaults are a fast interactive preset; **Advanced GP parameters** exposes
   crossover/mutation rates, parsimony, and tournament size.
4. Optionally select factors under **Seed from saved factors** to initialize from kept formulas.
5. Click **Start training**.

### 3.5 Monitor progress
On start, the Train tab shows a live progress view: a `generation X / Y` bar, a best-fitness
sparkline, and counters for **segments**, **cumulative trials**, and **OOS reads**, refreshed each
second until completion.

### 3.6 Stop a run
**Stop** in the progress view halts the run after the current generation; it completes normally with
fewer generations and the partial result is retained.

### 3.7 Continue training (changed settings, universe, or operators)
When a segment completes, a **Continue training from this generation** panel appears.
1. Set **Additional generations**.
2. Changed universe/parameter selections and any newly registered operators apply to the next
   segment. A changed universe is re-scored against the **frozen** boundary, so the locked test
   segment does not move.
3. Click **Continue**. The evolved population is warm-started rather than reinitialized. The
   cumulative trial count grows and the **OOS reads** counter increments (§3.11).

**Open dashboard** jumps to the metrics for the latest segment.

### 3.8 Read the results
- **Metrics** tab — the headline is **OOS rank IC, Deflated Sharpe, PBO, Verdict** (the
  out-of-sample / deflated figures). Train metrics and the trial count are under the collapsible
  "In-sample (train) metrics — for reference only."
- **Best factor** tab — the winning expression as a tree; click a node for its details, or **Save
  best factor to library**.
- **Genealogy** tab — see §3.9.

### 3.9 Explore the genealogy
The default **Generations** view is a collapsible list, newest generation first. Expanding a
generation groups its individuals by the variation operator that produced them — *crossover, subtree
mutation, point mutation, elite, seed*. Expanding a group lists its members **sorted by fitness (IC),
best first**. Each row can be selected (details appear on the right), **Save**d to the library, or
**Trace**d.

**Trace ancestry** switches to a focused graph containing only the selected factor's ancestor
closure — typically a few dozen nodes — which keeps the graph legible where a full-population DAG
would not be.

### 3.10 Save factors and seed new sessions
- **Save** from the Best factor tab (**Save best factor to library**) or from any Genealogy member
  (**Save**). The factor is stored with its tree, metrics, provenance, and any operators it uses.
- **Library** tab — saved factors with their research IC and source universe; **Rename** or
  **Delete** as needed.
- **Seed a new search** — select one or more factors and click **Start seeded session (N)**. The
  Train tab opens with those factors preselected as seeds; choose a universe and parameters and
  **Start training**. The initial population begins from the selected factors (recorded as `seed` in
  the lineage), which is how earlier results are combined into a new search.

### 3.11 On honest metrics
- The **test segment is locked** for the session's lifetime; changing the universe does not relocate
  the boundary.
- **Trials accumulate** across segments and seeded sessions; the deflated Sharpe cannot be reset by
  continuing.
- Each completed segment is **one out-of-sample read**. The dashboard shows the read count and raises
  a warning once it exceeds one: a result selected on after repeated readings is, in effect,
  in-sample, and the verdict should be treated with corresponding caution.

### 3.12 Save and load your work (workspaces)
The header provides **Save local / Load local** (browser storage) and, in app mode, **Save backend /
Load backend** (server-side workspace files). A workspace captures the current run, drafts, and the
active tab/selection. An in-flight session is also persisted: reloading the page re-attaches the
Train tab to it and resumes progress streaming.

### 3.13 Change where factors are stored
**Library** tab → **Storage folder** → set **Factors directory** and click **Save folder**.
Subsequent saves are written there. Resolution order: the `ALPHALINEAGE_FACTORS_DIR` environment
variable, then this setting, then the default `data_cache/factors/`.

---

## 4. Where to go next

- **Quality gates** (run before changing anything): `pytest -q`, `ruff check .`, `mypy src`, and in
  `frontend/`, `npm run typecheck && npm test`. The two load-bearing tests are synthetic-signal
  recovery and noise rejection; if either regresses, fix it before proceeding.
- **Architecture and invariants:** [`CLAUDE.md`](../CLAUDE.md) and
  [`DEVELOPMENT_PLAN.md`](../DEVELOPMENT_PLAN.md).
- **Static demo:** `cd frontend && npm run build:demo`, then deploy `dist/`. Regenerate the snapshot
  from a real run with `python scripts/export_demo.py --workspace run-<id> --out
  frontend/public/demo-run.json`.
