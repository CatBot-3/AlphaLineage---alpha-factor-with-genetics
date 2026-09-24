// The Library tab: the user's kept formula results. List, rename, delete, relocate the
// storage folder, and seed a new training session from one or more formula results.

import { useEffect, useRef, useState } from "react";
import { deleteFormulaResult, listFormulaResults, updateFormulaResult, listFormulas, resolveFormulaSource } from "../api/client";
import type { FactorNode, FormulaResult, FormulaSpec } from "../api/types";
import { FactorTree } from "../factor/FactorTree";

function researchIc(factor: FormulaResult): string {
  const ic = factor.metrics?.oos_ic ?? factor.metrics?.validation_median_ic;
  return typeof ic === "number" ? ic.toFixed(3) : "-";
}

export function LibraryPanel({ onSeed, onApply, onEdit }: {
  onSeed: (ids: string[]) => void;
  onApply?: (source: string, universe?: string | null) => void;
  onEdit?: (tree: FactorNode, name: string) => void;
}) {
  const [factors, setFactors] = useState<FormulaResult[]>([]);
  const [selected, setSelected] = useState<string[]>([]);
  const [status, setStatus] = useState<string | null>(null);
  const [loaded, setLoaded] = useState(false);
  const [search, setSearch] = useState("");
  const [view, setView] = useState<"results" | "formulas">("results");
  const [formulas, setFormulas] = useState<FormulaSpec[] | null>(null);
  const [formulaLoad, setFormulaLoad] = useState(0);
  const inspectionVersion = useRef(0);
  const [inspection, setInspection] = useState<Awaited<ReturnType<typeof resolveFormulaSource>> | null>(null);
  useEffect(() => {
    if (view !== "formulas") return;
    let active = true;
    setFormulas(null); setStatus(null);
    listFormulas().then(items => { if (active) setFormulas(items); })
      .catch(e => { if (active) setStatus(String(e)); });
    return () => { active = false; };
  }, [view, formulaLoad]);
  async function inspect(source: string, copy = false) {
    const version = ++inspectionVersion.current;
    setStatus(null);
    try {
      const resolved = await resolveFormulaSource(source);
      if (version !== inspectionVersion.current) return;
      if (copy) onEdit?.(resolved.tree, resolved.name);
      else setInspection(resolved);
    } catch (e) { if (version === inspectionVersion.current) setStatus(String(e)); }
  }

  function refresh() {
    setStatus(null);
    listFormulaResults().then(items => {setFactors(items); setLoaded(true);}).catch((e) => setStatus(String(e)));
  }

  useEffect(() => {
    refresh();
  }, []);

  function toggle(id: string) {
    setSelected((prev) => (prev.includes(id) ? prev.filter((s) => s !== id) : [...prev, id]));
  }

  async function handleRename(factor: FormulaResult) {
    const next = window.prompt("Rename formula result", factor.name);
    if (!next || next === factor.name) return;
    try { await updateFormulaResult(factor.id, { name: next }); refresh(); }
    catch (e) { setStatus(String(e)); }
  }

  async function handleDelete(id: string) {
    try {
      await deleteFormulaResult(id);
      setSelected((prev) => prev.filter((s) => s !== id));
      refresh();
    } catch (e) { setStatus(String(e)); }
  }

  return (
    <div className="library-panel" data-testid="library-panel">
      <div className="signals-toolbar">
        <button className="ghost" aria-pressed={view === "results"} onClick={() => setView("results")}>Saved results</button>
        <button className="ghost" aria-pressed={view === "formulas"} onClick={() => setView("formulas")}>Reusable formulas</button>
        <input aria-label="Search library" placeholder="Search names or universe" value={search} onChange={e => setSearch(e.target.value)}/>
      </div>
      {view === "formulas" && formulas === null && !status && <p>Loading reusable formulas…</p>}
      {view === "formulas" && formulas?.length === 0 && <p>No reusable formulas yet. Create one in Build &amp; Data.</p>}
      {view === "formulas" && <ul className="factor-rows">{formulas?.filter(item => `${item.name} ${item.display_name}`.toLowerCase().includes(search.toLowerCase())).map(item => <li key={item.runtime_name ?? item.name} className="factor-row">
        <span>{item.display_name || item.name} · revision {item.revision} · unvalidated</span>
        <span className="factor-actions"><button className="ghost" onClick={() => void inspect(`formula:${item.runtime_name ?? item.name}`)}>Inspect</button><button className="ghost" onClick={() => onApply?.(`formula:${item.runtime_name ?? item.name}`)}>Apply in Signals</button><button className="ghost" onClick={() => void inspect(`formula:${item.runtime_name ?? item.name}`, true)}>Edit a copy</button></span>
      </li>)}</ul>}
      {view === "results" && <>
      <section className="library-list">
        <header className="library-head">
          <h3>Formula Results</h3>
          <button
            type="button"
            className="primary-action"
            data-testid="seed-session"
            disabled={selected.length === 0}
            onClick={() => onSeed(selected)}
          >
            Seed training from Formula Results ({selected.length})
          </button>
        </header>

        {loaded && factors.length === 0 && !status && (
          <p className="hint">
            No saved results yet. Start training, then choose Save to Library in Results.
            <button className="ghost" onClick={() => onSeed([])}>Start training</button>
          </p>
        )}

        <ul className="factor-rows">
          {factors.filter(factor => `${factor.name} ${factor.provenance?.universe ?? ""}`.toLowerCase().includes(search.toLowerCase())).map((factor) => (
            <li key={factor.id} className="factor-row" data-testid="factor-row">
              <label className="factor-pick">
                <input
                  type="checkbox"
                  checked={selected.includes(factor.id)}
                  onChange={() => toggle(factor.id)}
                />
                <span className="factor-name">{factor.name}</span>
              </label>
              <span className="factor-meta">
                {factor.provenance?.evidence ?? "Legacy evidence"} · {factor.provenance?.validation_passed === true ? "Validation passed" : factor.provenance?.validation_passed === false ? "Validation not passed" : "Validation outcome unknown"} · IC {researchIc(factor)} · {factor.provenance?.universe ?? "Universe unknown"}
              </span>
              <span className="factor-actions">
                <button className="ghost" onClick={() => void inspect(`result:${factor.id}`)}>Inspect</button>
                <button className="ghost" onClick={() => onApply?.(`result:${factor.id}`, factor.provenance?.universe as string | undefined)}>Apply in Signals</button>
                <button className="ghost" onClick={() => void inspect(`result:${factor.id}`, true)}>Edit a copy</button>
                <button className="ghost" onClick={() => onSeed([factor.id])}>Seed training</button>
                <button type="button" className="ghost" onClick={() => handleRename(factor)}>
                  Rename
                </button>
                <button type="button" className="ghost" onClick={() => handleDelete(factor.id)}>
                  Delete
                </button>
              </span>
            </li>
          ))}
        </ul>
      </section>
      </>}
      {!loaded && !status && view === "results" && <p>Loading saved results…</p>}
      {inspection && <section className="library-inspection"><h3>{inspection.name}</h3><p>{inspection.evidence} · {inspection.universe ?? "Choose a universe on application"} · {inspection.execution ?? "Execution timing required"}{inspection.validation_passed === false ? " · Validation not passed" : inspection.validation_passed ? " · Validation passed" : ""}</p><FactorTree factor={inspection.tree}/><details><summary>Frozen expression and provenance</summary><pre>{JSON.stringify(inspection, null, 2)}</pre></details><button className="primary-action" onClick={() => onApply?.(inspection.id, inspection.universe)}>Apply in Signals</button><button className="ghost" onClick={() => setInspection(null)}>Close inspection</button></section>}

      <p className="hint">Change where formula results are stored in the settings menu.</p>

      {status && <p role="alert" className="surface-message">{status} <button className="ghost" onClick={() => view === "formulas" ? setFormulaLoad(value => value + 1) : refresh()}>Retry loading</button></p>}
      <p className="disclaimer">Not investment advice. Research output only.</p>
    </div>
  );
}
