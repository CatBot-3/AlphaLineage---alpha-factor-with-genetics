// P7-T1: define a point-in-time universe (symbols with entry/exit dates).

import { useEffect, useState } from "react";
import { defineUniverse } from "../api/client";
import type { UniverseDraft } from "../api/types";
import { toUniversePayload, type UniverseRow } from "./toUniversePayload";

const EMPTY: UniverseRow = { symbol: "", entry: "", exit: "" };

export function UniverseBuilder({
  draft,
  onDraftChange,
  canSubmit = true,
}: {
  draft?: UniverseDraft;
  onDraftChange?: (draft: UniverseDraft) => void;
  canSubmit?: boolean;
}) {
  const [name, setName] = useState(draft?.name ?? "my-universe");
  const [rows, setRows] = useState<UniverseRow[]>(draft?.rows ?? [{ ...EMPTY }]);
  const [result, setResult] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    onDraftChange?.({ name, rows });
  }, [name, rows, onDraftChange]);

  const update = (i: number, field: keyof UniverseRow, value: string) =>
    setRows((rs) => rs.map((r, j) => (j === i ? { ...r, [field]: value } : r)));

  async function submit() {
    if (!canSubmit) return;
    setError(null);
    setResult(null);
    try {
      const res = await defineUniverse(toUniversePayload(name, rows));
      setResult(`Defined "${res.name}" with ${res.symbols.length} symbols: ${res.symbols.join(", ")}`);
    } catch (e) {
      setError(String(e));
    }
  }

  return (
    <section className="panel" data-testid="universe-builder">
      <h3>Define a universe (point-in-time)</h3>
      {!canSubmit && (
        <p className="panel-note">
          Static demo mode saves this draft locally; connect the backend to define it.
        </p>
      )}
      <label className="field">
        Name <input value={name} onChange={(e) => setName(e.target.value)} />
      </label>
      <table className="rows">
        <thead>
          <tr>
            <th>Symbol</th>
            <th>Entry</th>
            <th>Exit (optional)</th>
            <th />
          </tr>
        </thead>
        <tbody>
          {rows.map((row, i) => (
            <tr key={i}>
              <td>
                <input
                  aria-label={`symbol-${i}`}
                  value={row.symbol}
                  onChange={(e) => update(i, "symbol", e.target.value)}
                />
              </td>
              <td>
                <input
                  aria-label={`entry-${i}`}
                  placeholder="2020-01-01"
                  value={row.entry}
                  onChange={(e) => update(i, "entry", e.target.value)}
                />
              </td>
              <td>
                <input
                  aria-label={`exit-${i}`}
                  placeholder="active"
                  value={row.exit}
                  onChange={(e) => update(i, "exit", e.target.value)}
                />
              </td>
              <td>
                <button onClick={() => setRows((rs) => rs.filter((_, j) => j !== i))}>
                  Remove
                </button>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
      <div className="actions">
        <button onClick={() => setRows((rs) => [...rs, { ...EMPTY }])}>+ Add symbol</button>
        <button onClick={submit} data-testid="define-universe" disabled={!canSubmit}>
          Define universe
        </button>
      </div>
      {result && (
        <p className="ok" data-testid="universe-result">
          {result}
        </p>
      )}
      {error && <p className="error">{error}</p>}
    </section>
  );
}
