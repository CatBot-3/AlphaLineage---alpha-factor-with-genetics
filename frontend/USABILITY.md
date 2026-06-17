# Usability pass — Phase 6 frontend

A heuristic review (Nielsen heuristics) of the three views, with findings and what was changed.
Recorded as the Phase 6 acceptance item ("one usability pass with notes recorded").

## Method
Walked the `demo` build (static `demo-run.json`) and the `app` build against a live backend,
exercising: read the metrics → open the best factor → inspect a node → step the genealogy.

## Findings & resolutions

1. **Honest metric must be unmistakably the default (invariant 1).** Risk: users anchor on the
   flattering train IC. *Resolution:* the dashboard's hero block is OOS/deflated (OOS IC, deflated
   Sharpe, PBO, verdict) with a "the honest default" label; train metrics are collapsed inside a
   `<details>` marked "for reference only." Covered by `test_dashboard_shows_deflated`.

2. **"Deflated Sharpe / PBO" are jargon.** *Resolution:* inline hints — the verdict reads
   "plausible / not significant," PBO is labelled a red-flag metric, and the panel header frames the
   block as out-of-sample. (Future: tooltips with one-line definitions.)

3. **Tree readability.** React Flow starts unfit on large trees. *Resolution:* `fitView` on mount +
   pan/zoom `Controls`; nodes show the primitive name and any window/const parameter; clicking a node
   opens a detail panel. (Future: collapse/expand subtrees.)

4. **Genealogy can be dense (40+ nodes/generation).** *Resolution:* a layered layout (generation on
   the y axis) + a navigable detail panel that traces parents → operation → children with clickable
   links, so you can walk lineage without hunting in the graph. Covered by `test_genealogy_navigable`.

5. **Loading / error states.** *Resolution:* explicit "Loading…" and error messages; the `demo` build
   needs no backend so first paint is instant. (Future: a progress indicator while an `app` run is
   queued/running.)

6. **Disclaimer visibility (invariant 8).** *Resolution:* a persistent footer — "Not investment
   advice. Research output only."

## Known gaps (tracked for later phases)
- Tooltips/glossary for the statistical metrics.
- Drag-to-edit the factor tree → Phase 7 (where React Flow's node editor is the right tool).

---

# Usability pass — iterative sessions, library & genealogy redesign (V1 finish)

A second heuristic pass covering the new Train, Library and redesigned Genealogy views, plus the
honesty surfacing. Exercised end-to-end against a live backend on the cached `sp500-lite` data:
configure a run → watch progress → continue with a changed population → save a factor → seed a
new session → trace ancestry.

## Findings & resolutions

1. **The old run button hid a hardcoded config (users couldn't choose anything).** *Resolution:* a
   Train tab with a run launcher — universe dropdown, the GP hyperparameters (core knobs visible, the
   rest behind an "Advanced" disclosure), and an optional seed-from-saved-factors picker. The form's
   values are posted verbatim; nothing is hardcoded. Covered by `TrainPanel.test.tsx`.

2. **A run gave no feedback until it finished.** *Resolution:* a live progress view — generation bar,
   best/mean-fitness sparkline, and a Stop button — fed by the backend's per-generation snapshot.

3. **No way to grow a search or reuse a find.** *Resolution:* after a segment completes, a "Continue
   from this generation" panel (changeable generations/params/universe) warm-starts the population;
   any lineage node or the best factor can be saved to the Library and used to seed a new session.

4. **Genealogy was illegible past a few hundred nodes (the whole-run DAG).** *Resolution (user-chosen
   design):* the default is now a collapsible per-generation list grouped by evolution method
   (crossover / subtree mutation / point mutation / elite / seed), each group expanding to its members
   in fitness-descending order; a "Trace ancestry" mode renders only a chosen node's ancestor closure
   as a small DAG — a graph where a graph actually helps. Covered by `Genealogy.test.tsx`,
   `groupLineage.test.ts`, `ancestry.test.ts`.

5. **Repeated out-of-sample reads silently erode honesty (invariant 1).** *Resolution:* the session
   counts every OOS read; the dashboard shows the read count and a prominent warning badge once it
   exceeds one ("repeated readings you select on become, in effect, in-sample"). Covered by the new
   Dashboard test. Trial counts also accumulate across segments and seeded sessions, so the deflated
   Sharpe only ever gets harder to pass.

6. **Factor storage location must be the user's choice.** *Resolution:* a Library settings field sets
   the factors directory via `PUT /settings`; the backend resolves env var > setting > default.

## Known gaps (tracked for later)
- Tooltips/glossary for the statistical metrics (still outstanding).
- Per-node tree preview inside the genealogy member rows (currently selects into the detail panel).
- Backgrounded runs survive a reload via the persisted session id, but a session browser/history
  list in the UI is not built yet (the backend exposes `GET /sessions`).
