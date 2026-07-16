// Extend > Universe Editor: define/edit a point-in-time universe and resolve tickers into it.

import { useEffect, useMemo, useState } from "react";
import {
  defineUniverse,
  deleteUniverse,
  getDataSync,
  getMembershipSync,
  getUniverse,
  listUniversePresets,
  listUniverses,
  searchSymbols,
  startDataSync,
  startMembershipSync,
  updateUniverse,
  validateSymbol,
} from "../api/client";
import type {
  MembershipSyncJob,
  SymbolCandidate,
  SymbolValidation,
  SyncProgressSnapshot,
  UniverseDraft,
  UniverseInfo,
  UniversePreset,
} from "../api/types";
import { CompactSection } from "../app/CompactSection";
import { rowsFromUniverse, toUniversePayload, uniqueSymbols, type UniverseRow } from "./toUniversePayload";
import { parseMembershipImport } from "./membershipImport";

const EMPTY: UniverseRow = { symbol: "", entry: "", exit: "" };
const DEFAULT_EXPECTED_START = "2020-01-01";
const SEARCH_LIMIT = 15;
const VISIBLE_CANDIDATES = 5;

function delay(ms: number): Promise<void> {
  return new Promise((resolve) => window.setTimeout(resolve, ms));
}

export function UniverseEditorPage({
  draft,
  onDraftChange,
  canSubmit = true,
  onPullProgress,
}: {
  draft?: UniverseDraft;
  onDraftChange?: (draft: UniverseDraft) => void;
  canSubmit?: boolean;
  onPullProgress?: (snapshot: SyncProgressSnapshot | null) => void;
}) {
  const [name, setName] = useState(draft?.name ?? "my-universe");
  const [rows, setRows] = useState<UniverseRow[]>(draft?.rows?.length ? draft.rows : [{ ...EMPTY }]);
  const [selectedUniverse, setSelectedUniverse] = useState(draft?.selectedUniverse ?? "");
  const [expectedStart, setExpectedStart] = useState(draft?.expectedStart ?? DEFAULT_EXPECTED_START);
  const [universeOptions, setUniverseOptions] = useState<UniverseInfo[]>([]);
  const [loadedUniverse, setLoadedUniverse] = useState<UniverseInfo | null>(null);
  const [presets, setPresets] = useState<UniversePreset[]>([]);
  const [presetError, setPresetError] = useState<string | null>(null);
  const [preparedPreset, setPreparedPreset] = useState<string | null>(null);
  const [universeMessage, setUniverseMessage] = useState<string | null>(null);
  const [universeError, setUniverseError] = useState<string | null>(null);
  const [membershipText, setMembershipText] = useState("");

  const [symbolQuery, setSymbolQuery] = useState("");
  const [candidates, setCandidates] = useState<SymbolCandidate[]>([]);
  const [candidatesExpanded, setCandidatesExpanded] = useState(false);
  const [selectedSymbol, setSelectedSymbol] = useState<string | null>(null);
  const [verifiedSymbols, setVerifiedSymbols] = useState<Set<string>>(new Set());
  const [candidateValidation, setCandidateValidation] = useState<SymbolValidation | null>(null);
  const [symbolError, setSymbolError] = useState<string | null>(null);
  const [symbolBusy, setSymbolBusy] = useState(false);

  const [membershipSyncJob, setMembershipSyncJob] = useState<MembershipSyncJob | null>(null);
  const [membershipSyncBusy, setMembershipSyncBusy] = useState(false);
  const [membershipSyncError, setMembershipSyncError] = useState<string | null>(null);

  const symbols = useMemo(() => uniqueSymbols(rows), [rows]);
  const selectedInfo = universeOptions.find((universe) => universe.name === selectedUniverse);
  const selectedDetail = loadedUniverse?.name === selectedUniverse ? loadedUniverse : selectedInfo;
  const cacheCoverage = selectedDetail?.cache_coverage;
  const cacheProblems = cacheCoverage?.incomplete_symbols ?? cacheCoverage?.missing_symbols ?? [];
  const importPreview = useMemo(() => parseMembershipImport(membershipText), [membershipText]);
  const delistedResults = (membershipSyncJob?.result?.results ?? []).filter(
    (result) => result.delisted && symbols.includes(result.symbol),
  );

  useEffect(() => {
    onDraftChange?.({ name, rows, selectedUniverse, expectedStart });
  }, [expectedStart, name, onDraftChange, rows, selectedUniverse]);

  useEffect(() => {
    if (!canSubmit) {
      setUniverseOptions([]);
      return;
    }
    let cancelled = false;
    listUniverses({ summary: true })
      .then((items) => {
        if (!cancelled) setUniverseOptions(items);
      })
      .catch(() => {
        if (!cancelled) setUniverseOptions([]);
      });
    listUniversePresets()
      .then((items) => {
        if (!cancelled) {
          setPresets(items);
          setPresetError(null);
        }
      })
      .catch((error) => {
        if (!cancelled) {
          setPresets([]);
          setPresetError(String(error));
        }
      });
    return () => {
      cancelled = true;
    };
  }, [canSubmit]);

  // Leaving this page mid-pull stops the poll loops below - clear the bottom bar rather than
  // leaving it frozen on a stale snapshot.
  useEffect(() => () => onPullProgress?.(null), [onPullProgress]);

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

  function removeSymbolRow(symbol: string) {
    setRows((current) => {
      const next = current.filter((row) => row.symbol.trim().toUpperCase() !== symbol);
      return next.length > 0 ? next : [{ ...EMPTY }];
    });
  }

  async function refreshUniverses() {
    if (!canSubmit) return;
    const [universesResult, presetsResult] = await Promise.allSettled([
      listUniverses({ summary: true }),
      listUniversePresets(),
    ]);
    if (presetsResult.status === "fulfilled") {
      setPresets(presetsResult.value);
      setPresetError(null);
    } else {
      setPresetError(String(presetsResult.reason));
    }
    if (universesResult.status === "rejected") throw universesResult.reason;
    setUniverseOptions(universesResult.value);
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
      setLoadedUniverse(universe);
      setPreparedPreset(null);
      setUniverseMessage(
        `Loaded ${universe.display_name ?? universe.name} with ${universe.symbols.length} symbols`,
      );
    } catch (error) {
      setUniverseError(String(error));
    }
  }

  function newUniverse() {
    setName("my-universe");
    setRows([{ ...EMPTY }]);
    setSelectedUniverse("");
    setLoadedUniverse(null);
    setUniverseMessage(null);
    setUniverseError(null);
    setPreparedPreset(null);
    setMembershipText("");
  }

  function loadPresetSnapshot(preset: UniversePreset) {
    setUniverseError(null);
    setUniverseMessage(null);
    const snapshot = preset.snapshot_universe ?? (preset.available ? preset.id : null);
    if (!snapshot) {
      setUniverseError(`${preset.display_name} does not include a bundled static snapshot.`);
      return;
    }
    void loadUniverse(snapshot);
  }

  function preparePresetImport(preset: UniversePreset) {
    setUniverseError(null);
    setUniverseMessage(null);
    const importName = preset.pit_import_name ?? preset.id;
    setName(importName);
    setRows([{ ...EMPTY }]);
    setSelectedUniverse("");
    setLoadedUniverse(null);
    setPreparedPreset(preset.id);
    setMembershipText("");
    setUniverseMessage(
      `Prepared a separate ${preset.display_name} point-in-time draft named ${importName}. No memberships were generated; import dated intervals below.`,
    );
  }

  function applyMembershipImport(mode: "replace" | "append") {
    setUniverseError(null);
    if (importPreview.rows.length === 0) {
      setUniverseError("Paste at least one valid symbol and entry date before importing.");
      return;
    }
    const currentRows = rows.filter((row) => row.symbol.trim() && row.entry.trim());
    const combined = mode === "replace" ? importPreview.rows : [...currentRows, ...importPreview.rows];
    const combinedPreview = parseMembershipImport(combined
      .map((row) => [row.symbol, row.entry, row.exit].join(","))
      .join("\n"));
    if (combinedPreview.errors.length > 0) {
      setUniverseError(`The imported rows conflict with the current draft: ${combinedPreview.errors[0].message}`);
      return;
    }
    setRows(combinedPreview.rows);
    setUniverseMessage(
      `Imported ${combinedPreview.rows.length} membership interval(s)${importPreview.errors.length ? `; skipped ${importPreview.errors.length} invalid line(s)` : ""}. Review the dates, then save the universe.`,
    );
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
    if (selectedInfo?.source !== "custom" && payload.name === selectedInfo?.name) {
      setUniverseError("Rename the bundled universe before saving a custom copy.");
      return;
    }
    const isUpdatingLoaded = selectedInfo?.source === "custom" && selectedUniverse === payload.name;
    if (!isUpdatingLoaded) {
      const collision = universeOptions.find(
        (universe) => universe.source === "custom" && universe.name === payload.name,
      );
      if (collision && !window.confirm(`A universe named "${payload.name}" already exists. Overwrite it?`)) {
        return;
      }
    }
    try {
      const response = isUpdatingLoaded
        ? await updateUniverse(payload.name, payload)
        : await defineUniverse(payload);
      setSelectedUniverse(response.name);
      setLoadedUniverse(null);
      setPreparedPreset(null);
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
      setLoadedUniverse(null);
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
    setCandidatesExpanded(false);
    try {
      const results = await searchSymbols(symbolQuery, SEARCH_LIMIT);
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
    setSelectedSymbol(candidate.symbol);
    setSymbolBusy(true);
    setSymbolError(null);
    setCandidateValidation(null);
    try {
      const validation = await validateSymbol({ symbol: candidate.symbol, start: expectedStart });
      setCandidateValidation(validation);
      if (validation.valid) {
        setVerifiedSymbols((current) => new Set(current).add(validation.symbol));
      } else {
        setSymbolError(validation.error ?? "Ticker did not validate.");
      }
    } catch (error) {
      setSymbolError(String(error));
    } finally {
      setSymbolBusy(false);
    }
  }

  async function pullSymbolNow(symbol: string) {
    try {
      const started = await startDataSync({ symbols: [symbol], start: expectedStart, mode: "incremental" });
      let job = await getDataSync(started.job_id);
      onPullProgress?.(job.progress ?? null);
      while (job.status === "queued" || job.status === "running" || job.status === "stopping") {
        await delay(500);
        job = await getDataSync(started.job_id);
        onPullProgress?.(job.progress ?? null);
      }
    } catch (error) {
      setSymbolError(String(error));
    } finally {
      onPullProgress?.(null);
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
      { symbol, entry: expectedStart, exit: "" },
    ]);
    setUniverseMessage(`Added ${symbol} to the draft universe`);
    if (canSubmit && window.confirm(`Pull price data for ${symbol} now?`)) {
      void pullSymbolNow(symbol);
    }
  }

  async function syncMembershipDates() {
    if (!canSubmit || symbols.length === 0) return;
    setMembershipSyncBusy(true);
    setMembershipSyncError(null);
    try {
      const started = await startMembershipSync({ symbols, expected_start: expectedStart });
      let job = await getMembershipSync(started.job_id);
      setMembershipSyncJob(job);
      onPullProgress?.(job.progress ?? null);
      while (job.status === "queued" || job.status === "running") {
        await delay(500);
        job = await getMembershipSync(started.job_id);
        setMembershipSyncJob(job);
        onPullProgress?.(job.progress ?? null);
      }
      if (job.status === "done" && job.result) {
        const resolved = new Map(
          job.result.results.filter((r) => r.status === "resolved").map((r) => [r.symbol, r]),
        );
        setRows((current) =>
          current.map((row) => {
            const result = resolved.get(row.symbol.trim().toUpperCase());
            if (!result) return row;
            return {
              ...row,
              entry: result.entry ?? row.entry,
              exit: result.delisted ? result.exit ?? row.exit : "",
            };
          }),
        );
      } else if (job.status === "failed") {
        setMembershipSyncError(job.error ?? "Date sync failed.");
      }
    } catch (error) {
      setMembershipSyncError(String(error));
    } finally {
      setMembershipSyncBusy(false);
      onPullProgress?.(null);
    }
  }

  return (
    <section className="panel universe-editor-page" data-testid="universe-editor-page">
      <header className="panel-head">
        <div>
          <h3>Universe editor</h3>
          <p className="panel-note">
            Resolve tickers, maintain point-in-time memberships, and keep entry/exit dates honest.
          </p>
        </div>
        <span className="mode-chip">{canSubmit ? "Local backend" : "Static demo"}</span>
      </header>

      {!canSubmit && (
        <p className="panel-note">
          Static demo mode keeps the universe draft locally. Search, validation, save, and date
          sync unlock when the backend is running.
        </p>
      )}

      {canSubmit && presets.length > 0 && (
        <CompactSection
          title="Common index templates"
          summary={`${presets.length} templates · snapshots and PIT`}
          defaultOpen
        >
          <p className="hint">
            A bundled static snapshot is convenient for current-universe research, but it is not
            historical membership. Prepare a separate point-in-time import for survivorship-safe
            research; the two definitions never overwrite one another.
          </p>
          <div className="universe-preset-grid" data-testid="universe-presets">
            {presets.map((preset) => {
              const snapshot = preset.snapshot_universe ?? (preset.available ? preset.id : null);
              const definition = preset.definition;
              const provenance = preset.provenance_detail;
              return (
                <article
                  key={preset.id}
                  className={`universe-preset${preparedPreset === preset.id ? " is-prepared" : ""}`}
                >
                  <div className="universe-preset__head">
                    <strong>{preset.display_name}</strong>
                    <span className="mode-chip">{preset.status.replace(/_/g, " ")}</span>
                  </div>
                  <span>{preset.coverage}</span>
                  <small>
                    {preset.mode === "static_snapshot" ? "Static definition" : "Bundled definition"}: {definition?.snapshot_date ?? "packaged with the app"}
                    {definition?.member_count != null ? ` · ${definition.member_count} members` : ""}
                  </small>
                  <small>Provenance: {provenance?.provider ?? preset.provenance}</small>
                  {provenance?.attribution && (
                    <small>
                      Attribution: {provenance.attribution}
                      {provenance.license ? ` · ${provenance.license}` : ""}
                    </small>
                  )}
                  {preset.fingerprint && (
                    <code title={preset.fingerprint}>Fingerprint: {preset.fingerprint}</code>
                  )}
                  <p className={preset.readiness?.research_ready ? "ok" : "oos-warning"}>
                    {preset.warning}
                  </p>
                  <div className="actions">
                    <button
                      type="button"
                      disabled={!snapshot}
                      onClick={() => loadPresetSnapshot(preset)}
                    >
                      {preset.mode === "static_snapshot"
                        ? "Load current snapshot"
                        : "Load bundled sample"}
                    </button>
                    <button
                      type="button"
                      onClick={() => preset.pit_universe
                        ? void loadUniverse(preset.pit_universe)
                        : preparePresetImport(preset)}
                    >
                      {preset.pit_universe
                        ? "Load point-in-time history"
                        : "Import point-in-time history"}
                    </button>
                  </div>
                </article>
              );
            })}
          </div>
        </CompactSection>
      )}
      {presetError && <p className="error">Could not load index templates: {presetError}</p>}

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
                {universe.display_name ?? universe.name} ({universe.source})
              </option>
            ))}
          </select>
        </label>
      )}

      {selectedDetail && (
        <aside className="universe-integrity" data-testid="universe-integrity">
          <strong>
            Mode: {(selectedDetail.mode ?? "point_in_time").replace(/_/g, " ")}
          </strong>
          <span>Source: {selectedDetail.source.replace(/_/g, " ")}</span>
          <span>
            Definition: {selectedDetail.definition?.display_name ?? selectedDetail.display_name ?? selectedDetail.name}
            {selectedDetail.definition?.snapshot_date
              ? ` as of ${selectedDetail.definition.snapshot_date}`
              : ""}
          </span>
          <span>
            {selectedDetail.definition?.interval_semantics ??
              selectedDetail.integrity?.membership_history.replace(/_/g, " ") ??
              "Membership definition supplied by the backend"}
          </span>
          <span>
            Members: {selectedDetail.definition?.member_count ?? selectedDetail.symbols.length}
          </span>
          <code title={selectedDetail.fingerprint ?? "Unavailable on this backend"}>
            Fingerprint: {selectedDetail.fingerprint ?? "unavailable"}
          </code>
          <small>
            Provenance: {selectedDetail.provenance?.provider ?? selectedDetail.integrity?.provenance ?? "unknown"}
            {selectedDetail.provenance?.retrieved_at
              ? ` · retrieved ${selectedDetail.provenance.retrieved_at}`
              : ""}
          </small>
          {selectedDetail.provenance?.source_url && (
            <a href={selectedDetail.provenance.source_url} target="_blank" rel="noreferrer">
              View definition source
            </a>
          )}
          {selectedDetail.provenance?.attribution && (
            <small>
              Attribution: {selectedDetail.provenance.attribution}
              {selectedDetail.provenance.license ? ` · ${selectedDetail.provenance.license}` : ""}
              {selectedDetail.provenance.license_url && (
                <>
                  {" · "}
                  <a
                    href={selectedDetail.provenance.license_url}
                    target="_blank"
                    rel="noreferrer"
                  >
                    License terms
                  </a>
                </>
              )}
              {selectedDetail.provenance.terms_url && (
                <>
                  {" · "}
                  <a
                    href={selectedDetail.provenance.terms_url}
                    target="_blank"
                    rel="noreferrer"
                  >
                    Wikimedia terms
                  </a>
                </>
              )}
            </small>
          )}
          <span>
            Readiness: membership {selectedDetail.readiness?.membership_ready ? "ready" : "not ready"}
            {" · "}prices {(selectedDetail.readiness?.price_ready ?? cacheCoverage?.complete) ? "ready" : "not ready"}
            {" · "}research {selectedDetail.readiness?.research_ready ? "ready" : "exploratory"}
          </span>
          {Object.keys(selectedDetail.aliases ?? {}).length > 0 && (
            <small>
              Symbol aliases: {Object.entries(selectedDetail.aliases ?? {}).slice(0, 4)
                .map(([symbol, alias]) => `${symbol} → ${alias}`).join(", ")}
            </small>
          )}
          <p className={selectedDetail.readiness?.research_ready ? "ok" : "oos-warning"}>
            {selectedDetail.readiness?.issues.join(" · ") ||
              selectedDetail.integrity?.warning ||
              "Review membership provenance and cached prices before research use."}
          </p>
        </aside>
      )}

      {cacheCoverage && (
        <aside className="universe-integrity" data-testid="universe-cache-coverage">
          <strong>Price history: {cacheCoverage.complete ? "complete" : "needs attention"}</strong>
          <span>
            {cacheCoverage.cached_symbols.length} of {cacheCoverage.eligible_symbols.length} historical
            symbols have cache files through {cacheCoverage.as_of}.
          </span>
          <p className={cacheCoverage.complete ? "ok" : "oos-warning"}>
            {cacheCoverage.complete
              ? "Cached dates cover every declared membership interval."
              : `Missing or incomplete history: ${cacheProblems.slice(0, 8).join(", ") || "inspect coverage details before training"}${cacheProblems.length > 8 ? "…" : ""}`}
          </p>
        </aside>
      )}

      <label className="field">
        <span className="field-label">Universe name</span>
        <input value={name} onChange={(event) => setName(event.target.value)} />
      </label>

      <label className="field">
        <span className="field-label">Expected start date</span>
        <input
          aria-label="Expected start date"
          placeholder="2020-01-01"
          value={expectedStart}
          onChange={(event) => setExpectedStart(event.target.value)}
        />
      </label>

      <CompactSection
        key={`membership-import-${preparedPreset ?? "draft"}`}
        title="Import membership history"
        summary={membershipText.trim()
          ? `${importPreview.rows.length} valid / ${importPreview.errors.length} errors`
          : "CSV or TSV"}
        defaultOpen={Boolean(preparedPreset)}
      >
        <p className="hint">
          Paste <code>symbol,entry,exit</code> rows. Entry is required; blank exit means the
          membership remains active. Exit is the first excluded date. A header row is optional.
        </p>
        <label className="field membership-import-field">
          <span className="field-label">Point-in-time membership CSV or TSV</span>
          <textarea
            aria-label="Point-in-time membership CSV or TSV"
            rows={8}
            value={membershipText}
            placeholder={"symbol,entry,exit\nAAPL,2000-01-03,\nLEH,2000-01-03,2008-09-15"}
            onChange={(event) => setMembershipText(event.target.value)}
            spellCheck={false}
          />
        </label>
        {membershipText.trim() && (
          <>
            <p className={importPreview.errors.length ? "error" : "ok"} data-testid="membership-import-summary">
              {importPreview.rows.length} valid interval(s); {importPreview.errors.length} error(s).
              {importPreview.headerSkipped ? " Header recognized." : ""}
            </p>
            {importPreview.errors.length > 0 && (
              <ul className="membership-import-errors" data-testid="membership-import-errors">
                {importPreview.errors.slice(0, 6).map((error) => (
                  <li key={`${error.line}-${error.value}`}>Line {error.line}: {error.message}</li>
                ))}
              </ul>
            )}
            {importPreview.rows.length > 0 && (
              <div className="membership-import-preview">
                <table className="rows">
                  <thead><tr><th>Symbol</th><th>Entry</th><th>Exit</th></tr></thead>
                  <tbody>{importPreview.rows.slice(0, 8).map((row, index) => (
                    <tr key={`${row.symbol}-${row.entry}-${index}`}><td>{row.symbol}</td><td>{row.entry}</td><td>{row.exit || "Active"}</td></tr>
                  ))}</tbody>
                </table>
                {importPreview.rows.length > 8 && <p className="hint">Previewing 8 of {importPreview.rows.length} valid rows.</p>}
              </div>
            )}
          </>
        )}
        <div className="actions">
          <button type="button" onClick={() => applyMembershipImport("replace")} disabled={importPreview.rows.length === 0}>
            Replace draft with valid rows
          </button>
          <button type="button" onClick={() => applyMembershipImport("append")} disabled={importPreview.rows.length === 0}>
            Append valid rows
          </button>
        </div>
      </CompactSection>

      <div className="universe-rows-scroll">
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
      </div>

      <div className="actions">
        <button type="button" data-testid="new-universe" onClick={newUniverse}>
          New universe
        </button>
        <button type="button" onClick={() => setRows((current) => [...current, { ...EMPTY }])}>
          Add row
        </button>
        <button type="button" data-testid="save-universe" onClick={saveUniverse} disabled={!canSubmit}>
          {selectedInfo?.source === "custom" && selectedUniverse === name
            ? "Update universe"
            : "Save universe"}
        </button>
        <button
          type="button"
          data-testid="delete-universe"
          className="ghost danger-navy"
          onClick={removeUniverse}
          disabled={!canSubmit || selectedInfo?.source !== "custom"}
        >
          Delete this universe
        </button>
      </div>

      <div className="actions">
        <button
          type="button"
          data-testid="sync-membership-dates"
          onClick={syncMembershipDates}
          disabled={!canSubmit || membershipSyncBusy || symbols.length === 0}
        >
          {membershipSyncBusy ? "Syncing dates..." : "Sync entry/exit dates"}
        </button>
      </div>
      {membershipSyncError && <p className="error">{membershipSyncError}</p>}

      {delistedResults.length > 0 && (
        <CompactSection
          title="Delisted symbols detected"
          summary={`${delistedResults.length} found`}
          defaultOpen
        >
          <ul className="coverage-list" data-testid="delisted-list">
            {delistedResults.map((result) => (
              <li key={result.symbol}>
                <strong>{result.symbol}</strong>
                <span>Exited {result.exit}</span>
                <span />
                <button type="button" className="ghost" onClick={() => removeSymbolRow(result.symbol)}>
                  Remove
                </button>
              </li>
            ))}
          </ul>
        </CompactSection>
      )}

      <CompactSection title="Add ticker" summary={candidateValidation?.symbol ?? symbolQuery}>
        <div className="inline-tools">
          <label className="field">
            <span className="field-label">Ticker query</span>
            <input
              aria-label="Ticker query"
              placeholder="AAPL"
              value={symbolQuery}
              onChange={(event) => setSymbolQuery(event.target.value)}
              onKeyDown={(event) => {
                if (event.key === "Enter") {
                  event.preventDefault();
                  searchTicker();
                }
              }}
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
          <>
            <ul className="candidate-list" data-testid="symbol-candidates">
              {candidates.slice(0, candidatesExpanded ? candidates.length : VISIBLE_CANDIDATES).map((candidate) => (
                <li key={candidate.symbol}>
                  <button
                    type="button"
                    aria-pressed={selectedSymbol === candidate.symbol}
                    onClick={() => selectCandidate(candidate)}
                  >
                    <strong>{candidate.symbol}</strong>
                    <span>{candidate.name || "Unnamed security"}</span>
                    <span>
                      {[candidate.exchange, candidate.quote_type, candidate.currency]
                        .filter(Boolean)
                        .join(" - ")}
                      {verifiedSymbols.has(candidate.symbol) && (
                        <span className="verified-badge" data-testid="verified-badge">
                          Verified
                        </span>
                      )}
                    </span>
                  </button>
                </li>
              ))}
            </ul>
            {!candidatesExpanded && candidates.length > VISIBLE_CANDIDATES && (
              <button
                type="button"
                className="show-more"
                data-testid="show-more-candidates"
                onClick={() => setCandidatesExpanded(true)}
              >
                ▼▼ Show {candidates.length - VISIBLE_CANDIDATES} more
              </button>
            )}
          </>
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

      {universeMessage && (
        <p className="ok" data-testid="universe-result">
          {universeMessage}
        </p>
      )}
      {universeError && <p className="error">{universeError}</p>}
    </section>
  );
}
