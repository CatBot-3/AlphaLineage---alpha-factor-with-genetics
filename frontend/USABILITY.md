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
- Live run progress UI for the `app` build (currently polls to completion).
