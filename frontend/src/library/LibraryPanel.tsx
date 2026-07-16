// The Library tab: the user's kept formula results. List, rename, delete, relocate the
// storage folder, and seed a new training session from one or more formula results.

import { useEffect, useState } from "react";
import { deleteFormulaResult, listFormulaResults, updateFormulaResult } from "../api/client";
import type { FormulaResult } from "../api/types";

function researchIc(factor: FormulaResult): string {
  const ic = factor.metrics?.oos_ic;
  return typeof ic === "number" ? ic.toFixed(3) : "-";
}

export function LibraryPanel({ onSeed }: { onSeed: (ids: string[]) => void }) {
  const [factors, setFactors] = useState<FormulaResult[]>([]);
  const [selected, setSelected] = useState<string[]>([]);
  const [status, setStatus] = useState<string | null>(null);

  function refresh() {
    listFormulaResults().then(setFactors).catch((e) => setStatus(String(e)));
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
    await updateFormulaResult(factor.id, { name: next });
    refresh();
  }

  async function handleDelete(id: string) {
    await deleteFormulaResult(id);
    setSelected((prev) => prev.filter((s) => s !== id));
    refresh();
  }

  return (
    <div className="library-panel" data-testid="library-panel">
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

        {factors.length === 0 && (
          <p className="hint">
            No formula results yet. Keep one from a formula backtest or save one from the Best
            Formula Result or Genealogy view.
          </p>
        )}

        <ul className="factor-rows">
          {factors.map((factor) => (
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
                research IC {researchIc(factor)} - {factor.provenance?.universe ?? "?"}
              </span>
              <span className="factor-actions">
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

      <p className="hint">Change where formula results are stored in the settings menu.</p>

      {status && <p className="surface-message">{status}</p>}
      <p className="disclaimer">Not investment advice. Research output only.</p>
    </div>
  );
}
