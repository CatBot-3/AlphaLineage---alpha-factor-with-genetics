// The run launcher: pick a universe, set GP hyperparameters, optionally seed from kept
// formula results, and start a session. Replaces the old hardcoded run config in the client.

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  getDataSync,
  getPrimitives,
  getUniverseCoverage,
  getUniverseFillPlan,
  listFormulas,
  listFormulaResults,
  listUniverses,
  startUniverseFill,
} from "../api/client";
import type {
  FormulaSpec,
  GpConfig,
  PrimitiveInfo,
  SavedFactor,
  SessionState,
  TrainingResourcesRequest,
  UniverseCacheCoverage,
  UniverseFillPlan,
  UniverseInfo,
} from "../api/types";
import { UniversePicker } from "../extend/UniverseTree";
import {
  ADVANCED_FIELDS,
  CORE_FIELDS,
  DEFAULT_CONFIG,
  EXECUTION_TIMING_OPTIONS,
} from "./defaults";
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

/**
 * The GP seed with a re-roll button.
 *
 * Deliberately manual. Re-rolling the seed to find a luckier trajectory is seed-hacking — it is
 * on the agent's refused list for exactly that reason — so this makes it one click for a human
 * making the choice knowingly, and never something that happens on its own.
 */
function seedField(value: number, onChange: (v: number) => void) {
  return (
    <label key="seed" className="field field--seed">
      <span className="field-label">Seed</span>
      <div className="field-seed-row">
        <input
          type="number"
          step="any"
          value={value}
          aria-label="Seed"
          onChange={(event) => onChange(Number(event.target.value))}
        />
        <button
          type="button"
          className="seed-dice"
          title="Pick a new random seed"
          aria-label="Randomize seed"
          onClick={() => onChange(Math.floor(Math.random() * 1_000_000))}
          data-testid="randomize-seed"
        >
          ⚄
        </button>
      </div>
    </label>
  );
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
  initialSession,
  onStart,
  disabled,
  onEditUniverse,
  onOpenDataSync,
  onOpenFormulaEditor,
}: {
  initialSeedIds?: string[];
  initialSession?: SessionState | null;
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
  const [fillPlan, setFillPlan] = useState<UniverseFillPlan | null>(null);
  const [filling, setFilling] = useState(false);
  const [fillNotice, setFillNotice] = useState<string | null>(null);
  const preparationKey = useRef("");
  preparationKey.current = JSON.stringify({name, universe, asOf, config, seedIds});
  const [coverageLoading, setCoverageLoading] = useState(true);
  const [coverageError, setCoverageError] = useState<string | null>(null);
  const [factors, setFactors] = useState<SavedFactor[]>([]);
  const [primitives, setPrimitives] = useState<PrimitiveInfo[]>([]);
  const [formulas, setFormulas] = useState<FormulaSpec[] | null>(null);
  const [disabledFormulaNames, setDisabledFormulaNames] = useState<Set<string>>(new Set());
  // Operator categories the GP may draw from this run. `condition` (boolean ops) is off by
  // default so the classic numeric search space is unchanged unless the user opts it in.
  const [disabledCats, setDisabledCats] = useState<Set<string>>(
    new Set(["condition", "technical_indicators", "classic_alphas"]),
  );
  const [previousEnabledCategories, setPreviousEnabledCategories] =
    useState<string[] | null | undefined>(undefined);
  const [previousEnabledFormulaNames, setPreviousEnabledFormulaNames] =
    useState<string[] | null | undefined>(undefined);

  useEffect(() => {
    listUniverses({ summary: true }).then(setUniverses).catch(() => setUniverses([]));
    listFormulaResults().then(setFactors).catch(() => setFactors([]));
    getPrimitives().then(setPrimitives).catch(() => setPrimitives([]));
    listFormulas().then(setFormulas).catch(() => setFormulas(null));
  }, []);

  const loadCoverage = useCallback(() => {
    setCoverageLoading(true);
    setCoverageError(null);
    return getUniverseCoverage(universe, asOf)
      .then((next) => {
        setCoverage(next);
        return next;
      })
      .finally(() => setCoverageLoading(false));
  }, [asOf, universe]);

  /** Fill this universe's gaps, then re-read coverage so the banner reflects the result. */
  const fillGaps = useCallback(
    async (scope: "top_up" | "full") => {
      setFilling(true);
      setFillNotice(null);
      try {
        const job = await startUniverseFill(universe, { as_of: asOf, scope });
        if (job.job_id) {
          // The global data-pull bar already shows per-symbol progress; this only waits.
          for (let attempt = 0; attempt < 600; attempt += 1) {
            const status = await getDataSync(job.job_id);
            if (status.status !== "queued" && status.status !== "running") {
              if (status.status === "failed" || status.status === "stopped") throw new Error(status.error ?? "Data preparation did not finish.");
              const quota = status.result?.quota;
              if (quota) {
                setFillNotice(
                  `The pull stopped early: ${quota.provider} reported the ${quota.scope} ` +
                    "allowance used. What was already fetched has been kept.",
                );
              }
              break;
            }
            await new Promise((resolve) => setTimeout(resolve, 1_000));
          }
        }
        const checked = await getUniverseCoverage(universe, asOf);
        setCoverage(checked);
        return checked.complete;
      } catch (error) {
        setFillNotice(error instanceof Error ? error.message : String(error));
      } finally {
        setFilling(false);
      }
    },
    [asOf, loadCoverage, universe],
  );

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

  // Readiness checks plan the required repair without downloading prices. The user's
  // Prepare data & start action owns the fill and continues only after coverage is rechecked.
  useEffect(() => {
    if (coverageLoading || coverage === null || coverage.complete) {
      setFillPlan(null);
      return;
    }
    let cancelled = false;
    getUniverseFillPlan(universe, asOf, "full")
      .then((plan) => {
        if (cancelled) return;
        setFillPlan(plan);

      })
      .catch(() => {
        if (!cancelled) setFillPlan(null);
      });
    return () => {
      cancelled = true;
    };
  }, [asOf, coverage, coverageLoading, disabled, fillGaps, universe]);

  useEffect(() => setSeedIds(initialSeedIds), [initialSeedIds]);

  useEffect(() => {
    if (!initialSession) return;
    if (initialSession.name) setName(`${initialSession.name} copy`);
    if (initialSession.universe) setUniverse(initialSession.universe);
    if (initialSession.as_of) setAsOf(initialSession.as_of);
    const storedConfig = initialSession.config ?? {};
    const previousConfig = {
      ...DEFAULT_CONFIG,
      ...storedConfig,
      // A stored config without the key predates the setting and meant the same-close timing;
      // "same setup" must reproduce that rather than silently switching to the new default.
      execution: storedConfig.execution ?? "close",
      novelty_mode: storedConfig.novelty_mode ?? "off",
    } as GpConfig;
    setConfig(previousConfig);
    setResources(initialSession.resources ?? { profile: "auto", cpu_budget_percent: null });
    setSeedIds(initialSession.seed_factor_ids ?? []);
    setPreviousEnabledCategories(storedConfig.enabled_categories);
    setPreviousEnabledFormulaNames(storedConfig.enabled_formula_names);
  }, [initialSession?.id]);

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

  useEffect(() => {
    if (previousEnabledCategories === undefined || functionsByCategory.size === 0) return;
    setDisabledCats(
      previousEnabledCategories === null
        ? new Set()
        : new Set(
          [...functionsByCategory.keys()].filter(
            (category) => !previousEnabledCategories.includes(category),
          ),
        ),
    );
  }, [functionsByCategory, previousEnabledCategories]);

  useEffect(() => {
    if (previousEnabledFormulaNames === undefined || formulas === null) return;
    setDisabledFormulaNames(
      previousEnabledFormulaNames === null
        ? new Set()
        : new Set(
          selectableFormulas
            .map((formula) => formula.name)
            .filter((name) => !previousEnabledFormulaNames.includes(name)),
        ),
    );
  }, [formulas, previousEnabledFormulaNames, selectableFormulas]);

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
  const missingSeedIds = seedIds.filter((id) => !factors.some((factor) => factor.id === id));
  const knownFormulaNames = new Set(selectableFormulas.map((formula) => formula.name));
  const missingFormulaNames = (previousEnabledFormulaNames ?? []).filter(
    (name) => !knownFormulaNames.has(name),
  );
  const missingDependencies = missingSeedIds.length > 0 || missingFormulaNames.length > 0;

  return (
    <form
      className="run-form"
      data-testid="run-config-form"
      onSubmit={async (e) => {
        e.preventDefault();
        if (disabled || filling || coverageLoading || missingDependencies) return;
        const setup = preparationKey.current;
        if (!trainingReady && (!(await fillGaps("full")) || preparationKey.current !== setup)) return;
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
          <label className="field"><span className="field-label">Novelty control</span><select aria-label="Novelty control" value={config.novelty_mode ?? "off"} onChange={e => setConfig(prev => ({...prev, novelty_mode: e.target.value as "balanced" | "off"}))}><option value="balanced">Balanced · reward different ideas</option><option value="off">Off · classic scoring</option></select><small>Known formulas remain useful building blocks. References stay frozen for this session.</small></label>
          <label className="field">
            <span className="field-label">Session name</span>
            <input value={name} aria-label="Session name" onChange={(e) => setName(e.target.value)} />
          </label>
          <div className="field run-universe-field">
            <span className="field-label">Universe</span>
            <span className="universe-field-row">
              <UniversePicker universes={universes} value={universe} onChange={setUniverse} />
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
          </div>
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
          <label className="field run-execution-field">
            <span className="field-label">Execution timing</span>
            <select
              aria-label="Execution timing"
              data-testid="execution-timing"
              value={config.execution ?? "close"}
              onChange={(event) =>
                setConfig((previous) => ({
                  ...previous,
                  execution: event.target.value as GpConfig["execution"],
                }))
              }
            >
              {EXECUTION_TIMING_OPTIONS.map((option) => (
                <option key={option.value} value={option.value}>{option.label}</option>
              ))}
            </select>
            <small
              className={config.execution === "close" ? "hint oos-warning" : "hint"}
              data-testid="execution-timing-hint"
            >
              {EXECUTION_TIMING_OPTIONS.find((option) => option.value === (config.execution ?? "close"))?.description}{" "}
              Fixed for the whole session: fitness, validation, backtests and the holdout all use it.
            </small>
          </label>
        </div>
        {missingDependencies && (
          <p className="error surface-message" role="alert" data-testid="missing-run-dependencies">
            The previous setup cannot start until these dependencies are restored:
            {missingSeedIds.length > 0 ? ` Formula Results ${missingSeedIds.join(", ")}.` : ""}
            {missingFormulaNames.length > 0 ? ` Formulas ${missingFormulaNames.join(", ")}.` : ""}
          </p>
        )}
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
          {!coverageLoading && !trainingReady && (
            <div className="run-readiness__actions">
              {fillPlan && fillPlan.symbols.length > 0 && !disabled && (
                <button
                  type="button"
                  className="primary-action"
                  data-testid="fill-price-gaps"
                  disabled={filling}
                  onClick={() => void fillGaps("full")}
                >
                  {filling
                    ? "Pulling prices…"
                    : fillPlan.first_time_count > 0
                      ? `Pull ${fillPlan.symbols.length} symbols (${fillPlan.first_time_count} new)`
                      : `Pull ${fillPlan.symbols.length} symbols`}
                </button>
              )}
              {onOpenDataSync && (
                <button
                  type="button"
                  className="ghost"
                  onClick={() => onOpenDataSync(universe)}
                >
                  Universe Editor
                </button>
              )}
            </div>
          )}
        </div>
        {fillPlan && fillPlan.first_time_count > 0 && (
          <p className="hint" data-testid="fill-plan-note">
            {fillPlan.cached_gap_count > 0
              ? `${fillPlan.cached_gap_count} cached symbol(s) are being refreshed automatically. `
              : ""}
            {fillPlan.first_time_count} symbol(s) have never been downloaded, which spends your
            provider's monthly unique-symbol allowance, so they wait for the button above.
          </p>
        )}
        {fillNotice && (
          <p className="oos-warning" data-testid="fill-notice" role="status">
            {fillNotice}
          </p>
        )}
      </section>

      <section className="run-form-section" aria-labelledby="search-budget-heading">
        <div className="run-form-section__head">
          <h3 id="search-budget-heading">Search budget</h3>
          <span>Control breadth, duration, and expression complexity.</span>
        </div>
        <div className="search-budget-grid">
          {CORE_FIELDS.map((f) =>
            f.key === "seed"
              ? seedField(config.seed as number, set("seed"))
              : numberField(f.key, config[f.key] as number, set(f.key), f.label),
          )}
        </div>
      </section>

      <TrainingResourcePicker value={resources} onChange={setResources} disabled={disabled} />

      <details className="advanced">
        <summary>Advanced GP parameters</summary>
        <div className="field-grid">
          <label className="field">
            <span className="field-label">Exploration profile</span>
            <select
              aria-label="Exploration profile"
              value={config.exploration_profile ?? "aggressive"}
              onChange={(event) =>
                setConfig((previous) => ({
                  ...previous,
                  exploration_profile: event.target.value as
                    | "classic"
                    | "balanced"
                    | "aggressive",
                }))
              }
            >
              <option value="aggressive">Aggressive valley crossing</option>
              <option value="balanced">Balanced exploration</option>
              <option value="classic">Classic tournament search</option>
            </select>
            <small className="hint">
              Aggressive mode protects structurally novel stepping stones and evaluates
              multi-edit endpoints without scoring weak intermediate formulas.
            </small>
          </label>
          <label className="field">
            <span className="field-label">Complexity penalty</span>
            <select
              aria-label="Complexity penalty"
              value={config.complexity_penalty_mode ?? "normalized_budget"}
              onChange={(event) =>
                setConfig((previous) => ({
                  ...previous,
                  complexity_penalty_mode: event.target.value as
                    | "per_node"
                    | "normalized_budget",
                }))
              }
            >
              <option value="normalized_budget">Normalized to max nodes</option>
              <option value="per_node">Per node (legacy)</option>
            </select>
            <small className="hint">
              Normalized mode caps the total deduction at the configured value when an
              expression reaches the node budget.
            </small>
          </label>
        </div>
        <div className="field-grid">
          {ADVANCED_FIELDS.map((f) => f.key === "seed" ? seedField(config.seed, set("seed")) :
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
        disabled={disabled || filling || coverageLoading || (!trainingReady && (!fillPlan || fillPlan.mode === "off")) || missingDependencies}
      >
        {filling ? "Preparing data…" : trainingReady ? "Start training" : "Prepare data & start"}
      </button>
    </form>
  );
}
