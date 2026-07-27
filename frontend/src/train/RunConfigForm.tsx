// The run launcher: pick a universe, set GP hyperparameters, optionally seed from kept
// formula results, and start a session. Replaces the old hardcoded run config in the client.

import { useEffect, useMemo, useState } from "react";
import {
  getPrimitives,
  getUniverseCoverage,
  listFormulas,
  listFormulaResults,
  listUniverses,
} from "../api/client";
import type {
  FormulaSpec,
  GpConfig,
  PrimitiveInfo,
  SavedFactor,
  TrainingResourcesRequest,
  UniverseCacheCoverage,
  UniverseInfo,
} from "../api/types";
import { ADVANCED_FIELDS, CORE_FIELDS, DEFAULT_CONFIG } from "./defaults";
import { TrainingResourcePicker } from "./TrainingResourcePicker";

export interface RunRequestForm {
  name: string;
  universe: string;
  as_of: string;
  config: GpConfig;
  resources: TrainingResourcesRequest;
  seed_factor_ids: string[];
}

// A stable empty default so the `initialSeedIds` effect dependency doesn't change every render
// (a fresh `[]` literal default would re-fire the effect forever -> infinite re-render).
const NO_SEEDS: string[] = [];

function localToday(): string {
  const now = new Date();
  const offset = now.getTimezoneOffset() * 60_000;
  return new Date(now.getTime() - offset).toISOString().slice(0, 10);
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
  initialSeedIds = NO_SEEDS,
  onStart,
  disabled,
  onEditUniverse,
  onOpenDataSync,
  onOpenFormulaEditor,
}: {
  initialSeedIds?: string[];
  onStart: (req: RunRequestForm) => void;
  disabled?: boolean;
  onEditUniverse?: (universeName: string) => void;
  onOpenDataSync?: (universeName: string) => void;
  onOpenFormulaEditor?: () => void;
}) {
  const [name, setName] = useState("Session");
  const [universe, setUniverse] = useState("sp500-lite");
  const [asOf, setAsOf] = useState(localToday);
  const [config, setConfig] = useState<GpConfig>({ ...DEFAULT_CONFIG });
  const [resources, setResources] = useState<TrainingResourcesRequest>({
    profile: "auto",
    cpu_budget_percent: null,
  });
  const [seedIds, setSeedIds] = useState<string[]>(initialSeedIds);
  const [universes, setUniverses] = useState<UniverseInfo[]>([]);
  const [coverage, setCoverage] = useState<UniverseCacheCoverage | null>(null);
  const [coverageLoading, setCoverageLoading] = useState(true);
  const [coverageError, setCoverageError] = useState<string | null>(null);
  const [factors, setFactors] = useState<SavedFactor[]>([]);
  const [primitives, setPrimitives] = useState<PrimitiveInfo[]>([]);
  const [formulas, setFormulas] = useState<FormulaSpec[] | null>(null);
  const [disabledFormulaNames, setDisabledFormulaNames] = useState<Set<string>>(new Set());
  // Operator categories the GP may draw from this run. `condition` (boolean ops) is off by
  // default so the classic numeric search space is unchanged unless the user opts it in.
  const [disabledCats, setDisabledCats] = useState<Set<string>>(
    new Set(["condition", "technical_indicators"]),
  );

  useEffect(() => {
    listUniverses({ summary: true }).then(setUniverses).catch(() => setUniverses([]));
    listFormulaResults().then(setFactors).catch(() => setFactors([]));
    getPrimitives().then(setPrimitives).catch(() => setPrimitives([]));
    listFormulas().then(setFormulas).catch(() => setFormulas(null));
  }, []);

  useEffect(() => {
    let cancelled = false;
    setCoverageLoading(true);
    setCoverageError(null);
    getUniverseCoverage(universe, asOf)
      .then((next) => {
        if (!cancelled) setCoverage(next);
      })
      .catch((error) => {
        if (!cancelled) {
          setCoverage(null);
          setCoverageError(error instanceof Error ? error.message : String(error));
        }
      })
      .finally(() => {
        if (!cancelled) setCoverageLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [universe, asOf]);

  useEffect(() => setSeedIds(initialSeedIds), [initialSeedIds]);

  const set = (key: keyof GpConfig) => (v: number) =>
    setConfig((prev) => ({ ...prev, [key]: v }));

  function toggleSeed(id: string) {
    setSeedIds((prev) => (prev.includes(id) ? prev.filter((s) => s !== id) : [...prev, id]));
  }

  // Functions available to training, grouped by category (operators + user formulas only).
  const functionsByCategory = useMemo(() => {
    const groups = new Map<string, PrimitiveInfo[]>();
    const latestFormulaRuntimes = formulas === null
      ? null
      : new Set(
        formulas
          .filter((formula) => formula.status !== "retired" && formula.registered !== false && !formula.error)
          .map((formula) => formula.runtime_name ?? formula.name),
      );
    for (const p of primitives) {
      if (p.kind !== "operator") continue;
      if (
        latestFormulaRuntimes !== null &&
        (p.origin === "catalog_formula" || p.origin === "user_formula") &&
        !latestFormulaRuntimes.has(p.runtime_name ?? p.name)
      ) continue;
      const key = p.category ?? "uncategorized";
      (groups.get(key) ?? groups.set(key, []).get(key)!).push(p);
    }
    return groups;
  }, [formulas, primitives]);

  function toggleCategory(cat: string) {
    setDisabledCats((prev) => {
      const next = new Set(prev);
      if (next.has(cat)) next.delete(cat);
      else next.add(cat);
      return next;
    });
  }

  function toggleFormula(formulaName: string) {
    setDisabledFormulaNames((previous) => {
      const next = new Set(previous);
      if (next.has(formulaName)) next.delete(formulaName);
      else next.add(formulaName);
      return next;
    });
  }

  function setCategoryFormulaSelection(items: FormulaSpec[], enabled: boolean) {
    setDisabledFormulaNames((previous) => {
      const next = new Set(previous);
      for (const formula of items) {
        if (enabled) next.delete(formula.name);
        else next.add(formula.name);
      }
      return next;
    });
  }

  const selectableFormulas = useMemo(() => (formulas ?? []).filter((formula) => (
    formula.status !== "retired" && formula.registered !== false && !formula.error
  )), [formulas]);

  const formulasByCategory = useMemo(() => {
    const groups = new Map<string, FormulaSpec[]>();
    for (const formula of selectableFormulas) {
      const key = formula.category ?? "custom";
      (groups.get(key) ?? groups.set(key, []).get(key)!).push(formula);
    }
    for (const items of groups.values()) {
      items.sort((left, right) => (
        (left.family_order ?? 0) - (right.family_order ?? 0) ||
        (left.display_name || left.name).localeCompare(right.display_name || right.name)
      ));
    }
    return groups;
  }, [selectableFormulas]);

  const selectedUniverse = universes.find((item) => item.name === universe);
  const trainingReady = coverage?.complete === true;

  return (
    <form
      className="run-form"
      data-testid="run-config-form"
      onSubmit={(e) => {
        e.preventDefault();
        if (disabled || coverageLoading || !trainingReady) return;
        const enabled = [...functionsByCategory.keys()].filter((c) => !disabledCats.has(c));
        const enabledFormulaNames = formulas === null
          ? null
          : selectableFormulas
            .filter((formula) => !disabledFormulaNames.has(formula.name))
            .map((formula) => formula.name);
        onStart({
          name,
          universe,
          as_of: asOf,
          config: {
            ...config,
            enabled_categories: enabled.length ? enabled : null,
            enabled_formula_names: enabledFormulaNames,
          },
          resources,
          seed_factor_ids: seedIds,
        });
      }}
    >
      <section className="run-form-section" aria-labelledby="run-setup-heading">
        <div className="run-form-section__head">
          <h3 id="run-setup-heading">Run setup</h3>
          <span>Choose the research context before tuning the search.</span>
        </div>
        <div className="run-setup-grid">
          <label className="field">
            <span className="field-label">Session name</span>
            <input value={name} aria-label="Session name" onChange={(e) => setName(e.target.value)} />
          </label>
          <label className="field run-universe-field">
            <span className="field-label">Universe</span>
            <span className="universe-field-row">
              <select value={universe} aria-label="Universe" onChange={(e) => setUniverse(e.target.value)}>
                {universes.length === 0 && <option value="sp500-lite">sp500-lite</option>}
                {universes.map((u) => (
                  <option key={u.name} value={u.name}>
                    {u.display_name ?? u.name} ({u.symbol_count ?? u.definition?.member_count ?? u.symbols.length})
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
          <label className="field run-date-field">
            <span className="field-label">As of date</span>
            <input
              type="date"
              value={asOf}
              max={localToday()}
              required
              aria-label="As of date"
              onChange={(event) => setAsOf(event.target.value)}
            />
            <small className="hint">Membership and available data are evaluated on this date.</small>
          </label>
        </div>
        <div
          className={`run-readiness ${trainingReady ? "run-readiness--ready" : "run-readiness--attention"}`}
          role="status"
          data-testid="run-readiness"
        >
          <div>
            <strong>
              {coverageLoading
                ? "Checking price readiness…"
                : trainingReady
                  ? "Price history ready"
                  : "Price history needs attention"}
            </strong>
            <span>
              {coverageLoading
                ? "Training unlocks after cached coverage is checked."
                : trainingReady
                  ? `${selectedUniverse?.symbol_count ?? selectedUniverse?.definition?.member_count ?? coverage?.eligible_symbols.length ?? 0} symbols are covered through ${asOf}.`
                  : coverageError ?? `${coverage?.incomplete_symbols?.length ?? 0} symbols are missing or incomplete through ${asOf}.`}
            </span>
          </div>
          {!coverageLoading && !trainingReady && onOpenDataSync && (
            <button type="button" className="ghost" onClick={() => onOpenDataSync(universe)}>
              Data Sync
            </button>
          )}
        </div>
      </section>

      <section className="run-form-section" aria-labelledby="search-budget-heading">
        <div className="run-form-section__head">
          <h3 id="search-budget-heading">Search budget</h3>
          <span>Control breadth, duration, and expression complexity.</span>
        </div>
        <div className="search-budget-grid">
          {CORE_FIELDS.map((f) =>
            numberField(f.key, config[f.key] as number, set(f.key), f.label),
          )}
        </div>
      </section>

      <TrainingResourcePicker value={resources} onChange={setResources} disabled={disabled} />

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
            universe; edit them in the Formula Builder.
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
        {[...functionsByCategory.entries()].map(([cat, prims]) => {
          const categoryFormulas = formulasByCategory.get(cat) ?? [];
          const categoryDisabled = disabledCats.has(cat);
          return (
          <fieldset key={cat} className="function-cat" data-testid={`function-cat-${cat}`}>
            <legend>
              <label className="seed-option">
                <input
                  type="checkbox"
                  aria-label={`enable ${cat}`}
                  checked={!categoryDisabled}
                  onChange={() => toggleCategory(cat)}
                />
                <span>{cat}</span>
              </label>
            </legend>
            <span className="function-cat-names">
              {prims
                .filter((primitive) => primitive.origin !== "catalog_formula" && primitive.origin !== "user_formula")
                .map((primitive) => primitive.name)
                .join(", ")}
              {categoryFormulas.length > 0 && (
                `${prims.some((primitive) => primitive.origin !== "catalog_formula" && primitive.origin !== "user_formula") ? " · " : ""}${categoryFormulas.length} selectable formula${categoryFormulas.length === 1 ? "" : "s"}`
              )}
            </span>
            {categoryFormulas.length > 0 && (
              <div className="training-formula-picker">
                <div className="training-formula-picker__head">
                  <p className="hint">
                    Choose the formulas training may combine. Only the declared numeric parameters
                    below are locally tuned; canonical market inputs stay fixed.
                  </p>
                  <span className="inline-tools">
                    <button
                      type="button"
                      className="ghost"
                      disabled={categoryDisabled}
                      onClick={() => setCategoryFormulaSelection(categoryFormulas, true)}
                    >Select all</button>
                    <button
                      type="button"
                      className="ghost"
                      disabled={categoryDisabled}
                      onClick={() => setCategoryFormulaSelection(categoryFormulas, false)}
                    >Clear</button>
                  </span>
                </div>
                <div className="training-formula-picker__grid">
                  {categoryFormulas.map((formula) => (
                    <label key={formula.name} className="training-formula-option">
                      <input
                        type="checkbox"
                        aria-label={`enable formula ${formula.name}`}
                        checked={!disabledFormulaNames.has(formula.name)}
                        disabled={categoryDisabled}
                        onChange={() => toggleFormula(formula.name)}
                      />
                      <span>
                        <strong>{formula.display_name || formula.name}</strong>
                        <small>
                          {(formula.inputs ?? []).length
                            ? (formula.inputs ?? []).map((input) => {
                              if (input.tuning?.enabled) {
                                return `${input.name} ${input.default} (${input.tuning.min}–${input.tuning.max}, step ${input.tuning.step})`;
                              }
                              return typeof input.default === "number"
                                ? `${input.name} ${input.default}`
                                : input.name;
                            }).join(" · ")
                            : "Fixed canonical market inputs"}
                          {(formula.constraints ?? []).map((constraint) => (
                            ` · ${constraint.left} ${constraint.operator === "lt" ? "<" : constraint.operator} ${constraint.right}`
                          )).join("")}
                        </small>
                      </span>
                    </label>
                  ))}
                </div>
              </div>
            )}
          </fieldset>
          );
        })}
      </details>

      {factors.length > 0 && (
        <fieldset className="seed-picker" data-testid="seed-picker">
          <legend>Seed training from Formula Results (optional)</legend>
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

      <button
        type="submit"
        className="primary-action"
        disabled={disabled || coverageLoading || !trainingReady}
      >
        Start training
      </button>
    </form>
  );
}
