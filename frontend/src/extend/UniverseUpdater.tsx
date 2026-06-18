import { useEffect, useMemo, useState } from "react";
import {
  addFormula,
  defineUniverse,
  deleteFormula,
  deleteUniverse,
  getDataCoverage,
  getDataSync,
  getUniverse,
  listFormulas,
  listUniverses,
  searchSymbols,
  startDataSync,
  updateUniverse,
  validateSymbol,
} from "../api/client";
import type {
  DataCoverage,
  DataSyncJob,
  FactorNode,
  FormulaDraft,
  FormulaSpec,
  SymbolCandidate,
  SymbolValidation,
  UniverseDraft,
  UniverseInfo,
} from "../api/types";
import { CompactSection } from "../app/CompactSection";
import { toUniversePayload, type UniverseRow } from "./toUniversePayload";

const EMPTY: UniverseRow = { symbol: "", entry: "", exit: "" };
const DEFAULT_SYNC_START = "2020-01-01";

interface FormulaTemplate {
  id: string;
  label: string;
  name: string;
  display_name: string;
  description: string;
  arg_types: string[];
  out_type: string;
  body: FactorNode;
}

const arg = (index: number): FactorNode => ({ name: "$arg", value: index });
const windowNode = (value: number): FactorNode => ({ name: "window", value });
const op = (name: string, children: FactorNode[]): FactorNode => ({ name, children });

const FORMULA_TEMPLATES: FormulaTemplate[] = [
  {
    id: "moving_average",
    label: "Moving average",
    name: "moving_average",
    display_name: "Moving average",
    description: "Trailing mean for a series with a caller-provided window.",
    arg_types: ["series", "window"],
    out_type: "series",
    body: op("ts_mean", [arg(0), arg(1)]),
  },
  {
    id: "momentum",
    label: "Momentum",
    name: "momentum",
    display_name: "Momentum",
    description: "Current value minus a delayed value.",
    arg_types: ["series", "window"],
    out_type: "series",
    body: op("sub", [arg(0), op("delay", [arg(0), arg(1)])]),
  },
  {
    id: "macd_spread",
    label: "Macd-style spread",
    name: "macd_spread",
    display_name: "Macd-style spread",
    description: "Twelve-day mean minus twenty-six-day mean.",
    arg_types: ["series"],
    out_type: "series",
    body: op("sub", [
      op("ts_mean", [arg(0), windowNode(12)]),
      op("ts_mean", [arg(0), windowNode(26)]),
    ]),
  },
  {
    id: "bollinger_zscore",
    label: "Bollinger z-score",
    name: "bollinger_zscore",
    display_name: "Bollinger z-score",
    description: "Distance from a twenty-day mean, scaled by twenty-day volatility.",
    arg_types: ["series"],
    out_type: "series",
    body: op("div", [
      op("sub", [arg(0), op("ts_mean", [arg(0), windowNode(20)])]),
      op("ts_std", [arg(0), windowNode(20)]),
    ]),
  },
];

function rowsFromUniverse(universe: UniverseInfo): UniverseRow[] {
  const rows = universe.memberships.map((membership) => ({
    symbol: membership.symbol,
    entry: membership.entry,
    exit: membership.exit ?? "",
  }));
  return rows.length > 0 ? rows : [{ ...EMPTY }];
}

function uniqueSymbols(rows: UniverseRow[]): string[] {
  return Array.from(
    new Set(rows.map((row) => row.symbol.trim().toUpperCase()).filter(Boolean)),
  );
}

function delay(ms: number): Promise<void> {
  return new Promise((resolve) => window.setTimeout(resolve, ms));
}

function formulaTemplate(id: string | undefined): FormulaTemplate {
  return FORMULA_TEMPLATES.find((template) => template.id === id) ?? FORMULA_TEMPLATES[0];
}

export function UniverseUpdater({
  draft,
  formulaDraft,
  onDraftChange,
  onFormulaDraftChange,
  canSubmit = true,
}: {
  draft?: UniverseDraft;
  formulaDraft?: FormulaDraft;
  onDraftChange?: (draft: UniverseDraft) => void;
  onFormulaDraftChange?: (draft: FormulaDraft) => void;
  canSubmit?: boolean;
}) {
  const initialFormula = formulaTemplate(formulaDraft?.template);
  const [name, setName] = useState(draft?.name ?? "my-universe");
  const [rows, setRows] = useState<UniverseRow[]>(
    draft?.rows?.length ? draft.rows : [{ ...EMPTY }],
  );
  const [selectedUniverse, setSelectedUniverse] = useState(draft?.selectedUniverse ?? "");
  const [syncStart, setSyncStart] = useState(draft?.syncStart ?? DEFAULT_SYNC_START);
  const [syncEnd, setSyncEnd] = useState(draft?.syncEnd ?? "");
  const [syncMode, setSyncMode] = useState<"incremental" | "refresh">(
    draft?.syncMode ?? "incremental",
  );
  const [universeOptions, setUniverseOptions] = useState<UniverseInfo[]>([]);
  const [universeMessage, setUniverseMessage] = useState<string | null>(null);
  const [universeError, setUniverseError] = useState<string | null>(null);

  const [symbolQuery, setSymbolQuery] = useState("");
  const [candidates, setCandidates] = useState<SymbolCandidate[]>([]);
  const [candidateValidation, setCandidateValidation] = useState<SymbolValidation | null>(null);
  const [symbolError, setSymbolError] = useState<string | null>(null);
  const [symbolBusy, setSymbolBusy] = useState(false);

  const [coverage, setCoverage] = useState<DataCoverage[]>([]);
  const [syncJob, setSyncJob] = useState<DataSyncJob | null>(null);
  const [syncError, setSyncError] = useState<string | null>(null);
  const [syncBusy, setSyncBusy] = useState(false);

  const [formulas, setFormulas] = useState<FormulaSpec[]>([]);
  const [formulaTemplateId, setFormulaTemplateId] = useState(initialFormula.id);
  const [formulaName, setFormulaName] = useState(formulaDraft?.name ?? initialFormula.name);
  const [formulaDisplay, setFormulaDisplay] = useState(
    formulaDraft?.display_name ?? initialFormula.display_name,
  );
  const [formulaDescription, setFormulaDescription] = useState(
    formulaDraft?.description ?? initialFormula.description,
  );
  const [formulaArgTypes, setFormulaArgTypes] = useState<string[]>(
    formulaDraft?.arg_types ?? initialFormula.arg_types,
  );
  const [formulaOutType, setFormulaOutType] = useState(
    formulaDraft?.out_type ?? initialFormula.out_type,
  );
  const [formulaBody, setFormulaBody] = useState<FactorNode>(
    formulaDraft?.body ?? initialFormula.body,
  );
  const [formulaMessage, setFormulaMessage] = useState<string | null>(null);
  const [formulaError, setFormulaError] = useState<string | null>(null);

  const symbols = useMemo(() => uniqueSymbols(rows), [rows]);
  const selectedInfo = universeOptions.find((universe) => universe.name === selectedUniverse);
  const staleCount = coverage.filter((item) => item.needs_sync).length;
  const currentCount = coverage.length - staleCount;
  const syncSummary =
    syncJob?.status === "running" || syncJob?.status === "queued"
      ? "Sync running"
      : coverage.length > 0
        ? `${staleCount} stale, ${currentCount} current`
        : `${symbols.length} symbols`;

  useEffect(() => {
    onDraftChange?.({ name, rows, selectedUniverse, syncStart, syncEnd, syncMode });
  }, [name, onDraftChange, rows, selectedUniverse, syncEnd, syncMode, syncStart]);

  useEffect(() => {
    onFormulaDraftChange?.({
      name: formulaName,
      display_name: formulaDisplay,
      description: formulaDescription,
      template: formulaTemplateId,
      arg_types: formulaArgTypes,
      out_type: formulaOutType,
      body: formulaBody,
    });
  }, [
    formulaArgTypes,
    formulaBody,
    formulaDescription,
    formulaDisplay,
    formulaName,
    formulaOutType,
    formulaTemplateId,
    onFormulaDraftChange,
  ]);

  useEffect(() => {
    if (!canSubmit) {
      setUniverseOptions([]);
      setFormulas([]);
      return;
    }
    let cancelled = false;
    listUniverses()
      .then((items) => {
        if (!cancelled) setUniverseOptions(items);
      })
      .catch(() => {
        if (!cancelled) setUniverseOptions([]);
      });
    listFormulas()
      .then((items) => {
        if (!cancelled) setFormulas(items);
      })
      .catch(() => {
        if (!cancelled) setFormulas([]);
      });
    return () => {
      cancelled = true;
    };
  }, [canSubmit]);

  function updateRow(index: number, field: keyof UniverseRow, value: string) {
    setRows((current) =>
      current.map((row, rowIndex) => (rowIndex === index ? { ...row, [field]: value } : row)),
    );
  }

  function removeRow(index: number) {
    setRows((current) => {
      const next = current.filter((_, rowIndex) => rowIndex !== index);
      return next.length > 0 ? next : [{ ...EMPTY }];
    });
  }

  async function refreshUniverses() {
    if (!canSubmit) return;
    setUniverseOptions(await listUniverses());
  }

  async function loadUniverse(nameToLoad: string) {
    setSelectedUniverse(nameToLoad);
    if (!canSubmit || !nameToLoad) return;
    setUniverseError(null);
    setUniverseMessage(null);
    try {
      const universe = await getUniverse(nameToLoad);
      setName(universe.name);
      setRows(rowsFromUniverse(universe));
      setUniverseMessage(`Loaded ${universe.name} with ${universe.symbols.length} symbols`);
    } catch (error) {
      setUniverseError(String(error));
    }
  }

  async function saveUniverse() {
    if (!canSubmit) return;
    setUniverseError(null);
    setUniverseMessage(null);
    const payload = toUniversePayload(name, rows);
    if (!payload.name || payload.memberships.length === 0) {
      setUniverseError("Add a universe name and at least one symbol with an entry date.");
      return;
    }
    if (selectedInfo?.source === "sample" && payload.name === selectedInfo.name) {
      setUniverseError("Rename the sample universe before saving a custom copy.");
      return;
    }
    try {
      const response =
        selectedInfo?.source === "custom" && selectedUniverse === payload.name
          ? await updateUniverse(payload.name, payload)
          : await defineUniverse(payload);
      setSelectedUniverse(response.name);
      setUniverseMessage(`Saved ${response.name} with ${response.symbols.length} symbols`);
      await refreshUniverses();
    } catch (error) {
      setUniverseError(String(error));
    }
  }

  async function removeUniverse() {
    if (!canSubmit || selectedInfo?.source !== "custom") return;
    setUniverseError(null);
    setUniverseMessage(null);
    try {
      await deleteUniverse(selectedInfo.name);
      setSelectedUniverse("");
      setUniverseMessage(`Deleted ${selectedInfo.name}`);
      await refreshUniverses();
    } catch (error) {
      setUniverseError(String(error));
    }
  }

  async function searchTicker() {
    if (!canSubmit || !symbolQuery.trim()) return;
    setSymbolBusy(true);
    setSymbolError(null);
    setCandidateValidation(null);
    try {
      const results = await searchSymbols(symbolQuery);
      setCandidates(results);
      if (results.length === 0) setSymbolError("No ticker candidates found.");
    } catch (error) {
      setSymbolError(String(error));
    } finally {
      setSymbolBusy(false);
    }
  }

  async function selectCandidate(candidate: SymbolCandidate) {
    if (!canSubmit) return;
    setSymbolBusy(true);
    setSymbolError(null);
    setCandidateValidation(null);
    try {
      const validation = await validateSymbol({
        symbol: candidate.symbol,
        start: syncStart,
        end: syncEnd || undefined,
      });
      setCandidateValidation(validation);
      if (!validation.valid) setSymbolError(validation.error ?? "Ticker did not validate.");
    } catch (error) {
      setSymbolError(String(error));
    } finally {
      setSymbolBusy(false);
    }
  }

  function addValidatedSymbol() {
    if (!candidateValidation?.valid) return;
    const symbol = candidateValidation.symbol.toUpperCase();
    if (symbols.includes(symbol)) {
      setSymbolError(`${symbol} is already in this universe draft.`);
      return;
    }
    setRows((current) => [
      ...current.filter((row) => row.symbol.trim() || row.entry.trim() || row.exit.trim()),
      { symbol, entry: syncStart, exit: "" },
    ]);
    setUniverseMessage(`Added ${symbol} to the draft universe`);
  }

  async function checkCoverage() {
    if (!canSubmit || symbols.length === 0) return;
    setSyncError(null);
    try {
      setCoverage(await getDataCoverage(symbols, syncStart, syncEnd || undefined));
    } catch (error) {
      setSyncError(String(error));
    }
  }

  async function syncData() {
    if (!canSubmit || symbols.length === 0) return;
    setSyncBusy(true);
    setSyncError(null);
    setSyncJob(null);
    try {
      const started = await startDataSync({
        symbols,
        start: syncStart,
        end: syncEnd || undefined,
        mode: syncMode,
      });
      let job = await getDataSync(started.job_id);
      setSyncJob(job);
      for (let i = 0; i < 120 && job.status !== "done" && job.status !== "failed"; i += 1) {
        await delay(500);
        job = await getDataSync(started.job_id);
        setSyncJob(job);
      }
      if (job.status === "done") {
        setCoverage(await getDataCoverage(symbols, syncStart, syncEnd || undefined));
      } else if (job.status === "failed") {
        setSyncError(job.error ?? "Data sync failed.");
      }
    } catch (error) {
      setSyncError(String(error));
    } finally {
      setSyncBusy(false);
    }
  }

  function applyTemplate(templateId: string) {
    const template = formulaTemplate(templateId);
    setFormulaTemplateId(template.id);
    setFormulaName(template.name);
    setFormulaDisplay(template.display_name);
    setFormulaDescription(template.description);
    setFormulaArgTypes(template.arg_types);
    setFormulaOutType(template.out_type);
    setFormulaBody(template.body);
    setFormulaError(null);
    setFormulaMessage(null);
  }

  async function saveFormula() {
    if (!canSubmit) return;
    setFormulaError(null);
    setFormulaMessage(null);
    const spec: FormulaSpec = {
      name: formulaName,
      display_name: formulaDisplay,
      description: formulaDescription,
      arg_types: formulaArgTypes,
      out_type: formulaOutType,
      body: formulaBody,
    };
    try {
      const saved = await addFormula(spec);
      setFormulas((current) => [...current.filter((item) => item.name !== saved.name), saved]);
      setFormulaMessage(`Saved formula ${saved.display_name || saved.name}`);
    } catch (error) {
      setFormulaError(String(error));
    }
  }

  async function removeFormula(nameToRemove: string) {
    if (!canSubmit) return;
    setFormulaError(null);
    setFormulaMessage(null);
    try {
      await deleteFormula(nameToRemove);
      setFormulas((current) => current.filter((item) => item.name !== nameToRemove));
      setFormulaMessage(`Deleted formula ${nameToRemove}`);
    } catch (error) {
      setFormulaError(String(error));
    }
  }

  return (
    <section className="panel universe-updater" data-testid="universe-updater">
      <header className="panel-head">
        <div>
          <h3>Update universe</h3>
          <p className="panel-note">
            Resolve tickers, maintain memberships, sync cached prices, and save reusable formulas.
          </p>
        </div>
        <span className="mode-chip">{canSubmit ? "Local backend" : "Static demo"}</span>
      </header>

      {!canSubmit && (
        <p className="panel-note">
          Static demo mode keeps universe and formula drafts locally. Search, validation, save, and
          data sync unlock when the backend is running.
        </p>
      )}

      <CompactSection title="Universe editor" summary={`${symbols.length} symbols`} defaultOpen>
        {canSubmit && (
          <label className="field">
            <span className="field-label">Load universe</span>
            <select
              aria-label="Load universe"
              value={selectedUniverse}
              onChange={(event) => loadUniverse(event.target.value)}
            >
              <option value="">Draft only</option>
              {universeOptions.map((universe) => (
                <option key={universe.name} value={universe.name}>
                  {universe.name} ({universe.source})
                </option>
              ))}
            </select>
          </label>
        )}

        <label className="field">
          <span className="field-label">Universe name</span>
          <input value={name} onChange={(event) => setName(event.target.value)} />
        </label>

        <table className="rows universe-rows">
          <thead>
            <tr>
              <th>Symbol</th>
              <th>Entry</th>
              <th>Exit</th>
              <th>Action</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((row, index) => (
              <tr key={`${row.symbol}-${index}`}>
                <td>
                  <input
                    aria-label={`symbol-${index}`}
                    value={row.symbol}
                    onChange={(event) => updateRow(index, "symbol", event.target.value)}
                  />
                </td>
                <td>
                  <input
                    aria-label={`entry-${index}`}
                    placeholder="2020-01-01"
                    value={row.entry}
                    onChange={(event) => updateRow(index, "entry", event.target.value)}
                  />
                </td>
                <td>
                  <input
                    aria-label={`exit-${index}`}
                    placeholder="active"
                    value={row.exit}
                    onChange={(event) => updateRow(index, "exit", event.target.value)}
                  />
                </td>
                <td>
                  <button type="button" onClick={() => removeRow(index)}>
                    Remove
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>

        <div className="actions">
          <button type="button" onClick={() => setRows((current) => [...current, { ...EMPTY }])}>
            Add row
          </button>
          <button
            type="button"
            data-testid="save-universe"
            onClick={saveUniverse}
            disabled={!canSubmit}
          >
            {selectedInfo?.source === "custom" && selectedUniverse === name
              ? "Update universe"
              : "Save universe"}
          </button>
          <button
            type="button"
            className="ghost"
            onClick={removeUniverse}
            disabled={!canSubmit || selectedInfo?.source !== "custom"}
          >
            Delete custom universe
          </button>
        </div>
      </CompactSection>

      <CompactSection title="Add ticker" summary={candidateValidation?.symbol ?? symbolQuery}>
        <div className="inline-tools">
          <label className="field">
            <span className="field-label">Ticker query</span>
            <input
              aria-label="Ticker query"
              placeholder="AAPL"
              value={symbolQuery}
              onChange={(event) => setSymbolQuery(event.target.value)}
            />
          </label>
          <button
            type="button"
            data-testid="search-symbols"
            onClick={searchTicker}
            disabled={!canSubmit || symbolBusy || !symbolQuery.trim()}
          >
            {symbolBusy ? "Searching..." : "Search"}
          </button>
          <button
            type="button"
            className="ghost"
            onClick={addValidatedSymbol}
            disabled={!candidateValidation?.valid}
          >
            Add selected ticker
          </button>
        </div>

        {candidates.length > 0 && (
          <ul className="candidate-list" data-testid="symbol-candidates">
            {candidates.map((candidate) => (
              <li key={candidate.symbol}>
                <button type="button" onClick={() => selectCandidate(candidate)}>
                  <strong>{candidate.symbol}</strong>
                  <span>{candidate.name || "Unnamed security"}</span>
                  <span>
                    {[candidate.exchange, candidate.quote_type, candidate.currency]
                      .filter(Boolean)
                      .join(" - ")}
                  </span>
                </button>
              </li>
            ))}
          </ul>
        )}

        {candidateValidation && (
          <p className={candidateValidation.valid ? "ok" : "error"} data-testid="symbol-validation">
            {candidateValidation.valid
              ? `${candidateValidation.symbol} validated with ${candidateValidation.rows} rows from ${candidateValidation.provider}`
              : `${candidateValidation.symbol} could not be validated: ${candidateValidation.error}`}
          </p>
        )}
        {symbolError && <p className="error">{symbolError}</p>}
      </CompactSection>

      <CompactSection title="Sync data" summary={syncSummary}>
        <div className="inline-tools">
          <label className="field">
            <span className="field-label">Start date</span>
            <input
              aria-label="Sync start date"
              value={syncStart}
              onChange={(event) => setSyncStart(event.target.value)}
            />
          </label>
          <label className="field">
            <span className="field-label">End date</span>
            <input
              aria-label="Sync end date"
              placeholder="today"
              value={syncEnd}
              onChange={(event) => setSyncEnd(event.target.value)}
            />
          </label>
          <label className="check-row">
            <input
              type="checkbox"
              checked={syncMode === "incremental"}
              onChange={(event) => setSyncMode(event.target.checked ? "incremental" : "refresh")}
            />
            Incremental merge
          </label>
        </div>
        <div className="actions">
          <button
            type="button"
            onClick={checkCoverage}
            disabled={!canSubmit || symbols.length === 0}
          >
            Check coverage
          </button>
          <button
            type="button"
            data-testid="sync-universe"
            onClick={syncData}
            disabled={!canSubmit || syncBusy || symbols.length === 0}
          >
            {syncBusy ? "Syncing..." : "Sync universe"}
          </button>
        </div>

        {coverage.length > 0 && (
          <ul className="coverage-list" data-testid="coverage-list">
            {coverage.map((item) => (
              <li key={item.symbol}>
                <strong>{item.symbol}</strong>
                <span>{item.cached ? `${item.rows} cached rows` : "No cached rows"}</span>
                <span>{item.needs_sync ? "Needs sync" : "Current"}</span>
                <span>
                  {item.first_date && item.last_date
                    ? `${item.first_date} to ${item.last_date}`
                    : "No date range"}
                </span>
              </li>
            ))}
          </ul>
        )}

        {syncJob?.result && (
          <ul className="coverage-list" data-testid="sync-results">
            {syncJob.result.results.map((result) => (
              <li key={result.symbol}>
                <strong>{result.symbol}</strong>
                <span>{result.status}</span>
                <span>{result.rows_fetched} fetched</span>
                <span>{result.error ?? `${result.rows_cached} cached`}</span>
              </li>
            ))}
          </ul>
        )}
        {syncError && <p className="error">{syncError}</p>}
      </CompactSection>

      <CompactSection
        title="Formula library"
        summary={canSubmit ? `${formulas.length} saved formulas` : "Local draft"}
      >
        <div className="template-grid">
          {FORMULA_TEMPLATES.map((template) => (
            <button
              key={template.id}
              type="button"
              aria-pressed={formulaTemplateId === template.id}
              onClick={() => applyTemplate(template.id)}
            >
              <strong>{template.label}</strong>
              <span>{template.description}</span>
            </button>
          ))}
        </div>

        <div className="formula-editor">
          <label className="field">
            <span className="field-label">Internal name</span>
            <input
              aria-label="Formula name"
              value={formulaName}
              onChange={(event) => setFormulaName(event.target.value)}
            />
          </label>
          <label className="field">
            <span className="field-label">Display name</span>
            <input
              aria-label="Formula display name"
              value={formulaDisplay}
              onChange={(event) => setFormulaDisplay(event.target.value)}
            />
          </label>
          <label className="field formula-description">
            <span className="field-label">Description</span>
            <input
              aria-label="Formula description"
              value={formulaDescription}
              onChange={(event) => setFormulaDescription(event.target.value)}
            />
          </label>
          <div className="formula-signature">
            <span>Args: {formulaArgTypes.join(", ")}</span>
            <span>Output: {formulaOutType}</span>
            <span>Body root: {formulaBody.name}</span>
          </div>
        </div>

        <div className="actions">
          <button
            type="button"
            data-testid="save-formula"
            onClick={saveFormula}
            disabled={!canSubmit}
          >
            Save formula
          </button>
        </div>

        {formulas.length > 0 && (
          <ul className="formula-list" data-testid="formula-list">
            {formulas.map((formula) => (
              <li key={formula.name}>
                <div>
                  <strong>{formula.display_name || formula.name}</strong>
                  <span>
                    {formula.name}({formula.arg_types.join(", ")}) {"->"} {formula.out_type}
                  </span>
                  {formula.error && <span className="error">{formula.error}</span>}
                </div>
                <button
                  type="button"
                  className="ghost"
                  onClick={() => removeFormula(formula.name)}
                  disabled={!canSubmit}
                >
                  Delete
                </button>
              </li>
            ))}
          </ul>
        )}
        {formulaMessage && <p className="ok">{formulaMessage}</p>}
        {formulaError && <p className="error">{formulaError}</p>}
      </CompactSection>

      {universeMessage && (
        <p className="ok" data-testid="universe-result">
          {universeMessage}
        </p>
      )}
      {universeError && <p className="error">{universeError}</p>}
    </section>
  );
}
