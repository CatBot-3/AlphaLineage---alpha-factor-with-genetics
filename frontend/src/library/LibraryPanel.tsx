// The Library tab: the user's kept factors. List, rename, delete, relocate the storage
// folder, and seed a new training session from one or more saved factors.

import { useEffect, useState } from "react";
import {
  deleteFactor,
  getSettings,
  listFactors,
  putSettings,
  renameFactor,
} from "../api/client";
import type { SavedFactor } from "../api/types";

function researchIc(factor: SavedFactor): string {
  const ic = factor.metrics?.oos_ic;
  return typeof ic === "number" ? ic.toFixed(3) : "-";
}

export function LibraryPanel({ onSeed }: { onSeed: (ids: string[]) => void }) {
  const [factors, setFactors] = useState<SavedFactor[]>([]);
  const [selected, setSelected] = useState<string[]>([]);
  const [factorsDir, setFactorsDir] = useState("");
  const [status, setStatus] = useState<string | null>(null);

  function refresh() {
    listFactors().then(setFactors).catch((e) => setStatus(String(e)));
  }

  useEffect(() => {
    refresh();
    getSettings()
      .then((s) => setFactorsDir(s.factors_dir))
      .catch(() => undefined);
  }, []);

  function toggle(id: string) {
    setSelected((prev) => (prev.includes(id) ? prev.filter((s) => s !== id) : [...prev, id]));
  }

  async function handleRename(factor: SavedFactor) {
    const next = window.prompt("Rename factor", factor.name);
    if (!next || next === factor.name) return;
    await renameFactor(factor.id, next);
    refresh();
  }

  async function handleDelete(id: string) {
    await deleteFactor(id);
    setSelected((prev) => prev.filter((s) => s !== id));
    refresh();
  }

  async function saveDir() {
    try {
      const s = await putSettings(factorsDir);
      setFactorsDir(s.factors_dir);
      setStatus(`Library folder set to ${s.factors_dir}`);
    } catch (e) {
      setStatus(String(e));
    }
  }

  return (
    <div className="library-panel" data-testid="library-panel">
      <section className="library-list">
        <header className="library-head">
          <h3>Saved factors</h3>
          <button
            type="button"
            className="primary-action"
            data-testid="seed-session"
            disabled={selected.length === 0}
            onClick={() => onSeed(selected)}
          >
            Start seeded session ({selected.length})
          </button>
        </header>

        {factors.length === 0 && <p className="hint">No saved factors yet. Save one from the Best factor or Genealogy view.</p>}

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

      <section className="library-settings">
        <h4>Storage folder</h4>
        <label className="field">
          <span className="field-label">Factors directory</span>
          <input
            value={factorsDir}
            aria-label="Factors directory"
            onChange={(e) => setFactorsDir(e.target.value)}
          />
        </label>
        <button type="button" className="ghost" onClick={saveDir}>
          Save folder
        </button>
      </section>

      {status && <p className="surface-message">{status}</p>}
      <p className="disclaimer">Not investment advice. Research output only.</p>
    </div>
  );
}
