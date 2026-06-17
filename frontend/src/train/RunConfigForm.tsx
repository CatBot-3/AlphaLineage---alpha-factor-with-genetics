// The run launcher: pick a universe, set GP hyperparameters, optionally seed from saved
// factors, and start a session. Replaces the old hardcoded run config in the client.

import { useEffect, useState } from "react";
import { listFactors, listUniverses } from "../api/client";
import type { GpConfig, SavedFactor, UniverseInfo } from "../api/types";
import { ADVANCED_FIELDS, CORE_FIELDS, DEFAULT_CONFIG } from "./defaults";

export interface RunRequestForm {
  name: string;
  universe: string;
  config: GpConfig;
  seed_factor_ids: string[];
}

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
  initialSeedIds = [],
  onStart,
  disabled,
}: {
  initialSeedIds?: string[];
  onStart: (req: RunRequestForm) => void;
  disabled?: boolean;
}) {
  const [name, setName] = useState("Session");
  const [universe, setUniverse] = useState("sp500-lite");
  const [config, setConfig] = useState<GpConfig>({ ...DEFAULT_CONFIG });
  const [seedIds, setSeedIds] = useState<string[]>(initialSeedIds);
  const [universes, setUniverses] = useState<UniverseInfo[]>([]);
  const [factors, setFactors] = useState<SavedFactor[]>([]);

  useEffect(() => {
    listUniverses().then(setUniverses).catch(() => setUniverses([]));
    listFactors().then(setFactors).catch(() => setFactors([]));
  }, []);

  useEffect(() => setSeedIds(initialSeedIds), [initialSeedIds]);

  const set = (key: keyof GpConfig) => (v: number) =>
    setConfig((prev) => ({ ...prev, [key]: v }));

  function toggleSeed(id: string) {
    setSeedIds((prev) => (prev.includes(id) ? prev.filter((s) => s !== id) : [...prev, id]));
  }

  return (
    <form
      className="run-form"
      data-testid="run-config-form"
      onSubmit={(e) => {
        e.preventDefault();
        onStart({ name, universe, config, seed_factor_ids: seedIds });
      }}
    >
      <div className="field-grid">
        <label className="field">
          <span className="field-label">Session name</span>
          <input value={name} aria-label="Session name" onChange={(e) => setName(e.target.value)} />
        </label>
        <label className="field">
          <span className="field-label">Universe</span>
          <select value={universe} aria-label="Universe" onChange={(e) => setUniverse(e.target.value)}>
            {universes.length === 0 && <option value="sp500-lite">sp500-lite</option>}
            {universes.map((u) => (
              <option key={u.name} value={u.name}>
                {u.name} ({u.symbols.length})
              </option>
            ))}
          </select>
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
