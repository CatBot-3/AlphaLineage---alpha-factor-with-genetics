// The run launcher: pick a universe, set GP hyperparameters, optionally seed from saved
// factors, and start a session. Replaces the old hardcoded run config in the client.

import { useEffect, useMemo, useState } from "react";
import { getPrimitives, listFactors, listUniverses } from "../api/client";
import type { GpConfig, PrimitiveInfo, SavedFactor, UniverseInfo } from "../api/types";
import { ADVANCED_FIELDS, CORE_FIELDS, DEFAULT_CONFIG } from "./defaults";

export interface RunRequestForm {
  name: string;
  universe: string;
  config: GpConfig;
  seed_factor_ids: string[];
}

// A stable empty default so the `initialSeedIds` effect dependency doesn't change every render
// (a fresh `[]` literal default would re-fire the effect forever -> infinite re-render).
const NO_SEEDS: string[] = [];

function numberField(key: keyof GpConfig, value: number, onChange: (v: number) => void, label: string) {
  return (
    <label key={String(key)} className="field">
      <span className="field-label">{label}</span>
      <input
        type="number"
        step="any"
        value={value}
        aria-label={label}
        onChange={(e) => onChange(Number(e.target.value))}
      />
    </label>
  );
}

export function RunConfigForm({
  initialSeedIds = NO_SEEDS,
  onStart,
  disabled,
  onEditUniverse,
  onOpenFormulaEditor,
}: {
  initialSeedIds?: string[];
  onStart: (req: RunRequestForm) => void;
  disabled?: boolean;
  onEditUniverse?: (universeName: string) => void;
  onOpenFormulaEditor?: () => void;
}) {
  const [name, setName] = useState("Session");
  const [universe, setUniverse] = useState("sp500-lite");
  const [config, setConfig] = useState<GpConfig>({ ...DEFAULT_CONFIG });
  const [seedIds, setSeedIds] = useState<string[]>(initialSeedIds);
  const [universes, setUniverses] = useState<UniverseInfo[]>([]);
  const [factors, setFactors] = useState<SavedFactor[]>([]);
  const [primitives, setPrimitives] = useState<PrimitiveInfo[]>([]);
  // Operator categories the GP may draw from this run. `condition` (boolean ops) is off by
  // default so the classic numeric search space is unchanged unless the user opts it in.
  const [disabledCats, setDisabledCats] = useState<Set<string>>(new Set(["condition"]));

  useEffect(() => {
    listUniverses().then(setUniverses).catch(() => setUniverses([]));
    listFactors().then(setFactors).catch(() => setFactors([]));
    getPrimitives().then(setPrimitives).catch(() => setPrimitives([]));
  }, []);

  useEffect(() => setSeedIds(initialSeedIds), [initialSeedIds]);

  const set = (key: keyof GpConfig) => (v: number) =>
    setConfig((prev) => ({ ...prev, [key]: v }));

  function toggleSeed(id: string) {
    setSeedIds((prev) => (prev.includes(id) ? prev.filter((s) => s !== id) : [...prev, id]));
  }

  // Functions available to training, grouped by category (operators + user formulas only).
  const functionsByCategory = useMemo(() => {
    const groups = new Map<string, PrimitiveInfo[]>();
    for (const p of primitives) {
      if (p.kind !== "operator") continue;
      const key = p.category ?? "uncategorized";
      (groups.get(key) ?? groups.set(key, []).get(key)!).push(p);
    }
    return groups;
  }, [primitives]);

  function toggleCategory(cat: string) {
    setDisabledCats((prev) => {
      const next = new Set(prev);
      if (next.has(cat)) next.delete(cat);
      else next.add(cat);
      return next;
    });
  }

  return (
    <form
      className="run-form"
      data-testid="run-config-form"
      onSubmit={(e) => {
        e.preventDefault();
        const enabled = [...functionsByCategory.keys()].filter((c) => !disabledCats.has(c));
        onStart({
          name,
          universe,
          config: { ...config, enabled_categories: enabled.length ? enabled : null },
          seed_factor_ids: seedIds,
        });
      }}
    >
      <div className="field-grid">
        <label className="field">
          <span className="field-label">Session name</span>
          <input value={name} aria-label="Session name" onChange={(e) => setName(e.target.value)} />
        </label>
        <label className="field">
          <span className="field-label">Universe</span>
          <span className="universe-field-row">
            <select value={universe} aria-label="Universe" onChange={(e) => setUniverse(e.target.value)}>
              {universes.length === 0 && <option value="sp500-lite">sp500-lite</option>}
              {universes.map((u) => (
                <option key={u.name} value={u.name}>
                  {u.name} ({u.symbols.length})
                </option>
              ))}
            </select>
            {onEditUniverse && (
              <button
                type="button"
                className="edit-universe-btn"
                data-testid="edit-universe"
                onClick={() => onEditUniverse(universe)}
              >
                Edit
              </button>
            )}
          </span>
        </label>
        {CORE_FIELDS.map((f) =>
          numberField(f.key, config[f.key] as number, set(f.key), f.label),
        )}
      </div>

      <details className="advanced">
        <summary>Advanced GP parameters</summary>
        <div className="field-grid">
          {ADVANCED_FIELDS.map((f) =>
            numberField(f.key, config[f.key] as number, set(f.key), f.label),
          )}
        </div>
      </details>

      <details className="advanced" data-testid="function-space">
        <summary>Available functions (current settings)</summary>
        <div className="function-space-head">
          <span className="hint">
            Toggle which operator categories the search may use. Functions are not part of the
            universe; edit them in the Formula Editor.
          </span>
          {onOpenFormulaEditor && (
            <button
              type="button"
              className="edit-universe-btn"
              data-testid="edit-functions"
              onClick={onOpenFormulaEditor}
            >
              Edit functions
            </button>
          )}
        </div>
        {[...functionsByCategory.entries()].map(([cat, prims]) => (
          <fieldset key={cat} className="function-cat" data-testid={`function-cat-${cat}`}>
            <legend>
              <label className="seed-option">
                <input
                  type="checkbox"
                  aria-label={`enable ${cat}`}
                  checked={!disabledCats.has(cat)}
                  onChange={() => toggleCategory(cat)}
                />
                <span>{cat}</span>
              </label>
            </legend>
            <span className="function-cat-names">{prims.map((p) => p.name).join(", ")}</span>
          </fieldset>
        ))}
      </details>

      {factors.length > 0 && (
        <fieldset className="seed-picker" data-testid="seed-picker">
          <legend>Seed from saved factors (optional)</legend>
          {factors.map((factor) => (
            <label key={factor.id} className="seed-option">
              <input
                type="checkbox"
                checked={seedIds.includes(factor.id)}
                onChange={() => toggleSeed(factor.id)}
              />
              <span>{factor.name}</span>
            </label>
          ))}
        </fieldset>
      )}

      <button type="submit" className="primary-action" disabled={disabled}>
        Start training
      </button>
    </form>
  );
}
