// Extend > Universe Editor: define current snapshots or point-in-time memberships and prices.

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
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
import { UniverseDataSync } from "./SyncDataPage";
import { rowsFromUniverse, toUniversePayload, uniqueSymbols, type UniverseRow } from "./toUniversePayload";
import { parseMembershipImport } from "./membershipImport";

const EMPTY: UniverseRow = { symbol: "", entry: "", exit: "" };
const DEFAULT_EXPECTED_START = "2020-01-01";
const SEARCH_LIMIT = 15;
const VISIBLE_CANDIDATES = 5;

function delay(ms: number): Promise<void> {
  return new Promise((resolve) => window.setTimeout(resolve, ms));
}

function requiredPriceStart(
  rows: UniverseRow[],
  expectedStart: string,
  mode: UniverseInfo["mode"],
  backendRequiredStart?: string | null,
): string {
  const candidates = [expectedStart];
  // A point-in-time universe needs prices for every declared membership interval. This is the
  // important distinction between "a cache file exists" and "the universe is trainable".
  if ((mode ?? "point_in_time") === "point_in_time") {
    if (backendRequiredStart && /^\d{4}-\d{2}-\d{2}$/.test(backendRequiredStart)) {
      return backendRequiredStart;
    }
    candidates.push(...rows.map((row) => row.entry));
  }
  return candidates
    .map((value) => value.trim())
    .filter((value) => /^\d{4}-\d{2}-\d{2}$/.test(value))
    .sort()[0] ?? DEFAULT_EXPECTED_START;
}

function normalizedMembershipRows(rows: UniverseRow[]): string[] {
  return rows
    .filter((row) => row.symbol.trim() || row.entry.trim() || row.exit.trim())
    .map((row) => [
      row.symbol.trim().toUpperCase(),
      row.entry.trim(),
      row.exit.trim(),
    ].join("|"))
    .sort();
}

function savedMembershipRows(universe: UniverseInfo): string[] {
  return universe.memberships
    .map((membership) => [
      membership.symbol.trim().toUpperCase(),
      membership.entry,
      membership.exit ?? "",
    ].join("|"))
    .sort();
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
  const loadUniverseRequest = useRef(0);
  const mountedRef = useRef(true);
  const directOperationEpoch = useRef(0);
  const selectedUniverseRef = useRef(selectedUniverse);
  selectedUniverseRef.current = selectedUniverse;

  const selectUniverseIdentity = useCallback((universeName: string) => {
    if (selectedUniverseRef.current !== universeName) {
      directOperationEpoch.current += 1;
      setMembershipSyncBusy(false);
      setMembershipSyncJob(null);
      setMembershipSyncError(null);
      onPullProgress?.(null);
    }
    selectedUniverseRef.current = universeName;
    setSelectedUniverse(universeName);
  }, [onPullProgress]);

  const symbols = useMemo(() => uniqueSymbols(rows), [rows]);
  const selectedInfo = universeOptions.find((universe) => universe.name === selectedUniverse);
  const selectedDetail = loadedUniverse?.name === selectedUniverse ? loadedUniverse : selectedInfo;
  const loadedSavedUniverse = loadedUniverse?.name === selectedUniverse ? loadedUniverse : null;
  const cacheCoverage = selectedDetail?.cache_coverage;
  const cacheProblems = cacheCoverage?.incomplete_symbols ?? cacheCoverage?.missing_symbols ?? [];
  const savedDefinitionDirty = Boolean(
    selectedUniverse && loadedSavedUniverse && (
      name.trim() !== loadedSavedUniverse.name ||
      normalizedMembershipRows(rows).join("\n") !== savedMembershipRows(loadedSavedUniverse).join("\n")
    ),
  );
  const recommendedPriceStart = useMemo(
    () => requiredPriceStart(
      rows,
      expectedStart,
      selectedDetail?.mode,
      cacheCoverage?.required_start,
    ),
    [cacheCoverage?.required_start, expectedStart, rows, selectedDetail?.mode],
  );
  const importPreview = useMemo(() => parseMembershipImport(membershipText), [membershipText]);
  const membershipDiagnostics = (membershipSyncJob?.result?.results ?? []).filter(
    (result) => symbols.includes(result.symbol),
  );

  const acceptRefreshedUniverse = useCallback((universe: UniverseInfo) => {
    // Sync, save, and hydration requests may finish after the user selects something else.
    // Never let an old detail response replace the active universe's readiness/membership data.
    if (universe.name !== selectedUniverseRef.current) return;
    setLoadedUniverse(universe);
    setUniverseOptions((current) => {
      const existing = current.findIndex((item) => item.name === universe.name);
      if (existing < 0) return [universe, ...current];
      return current.map((item, index) => index === existing ? universe : item);
    });
  }, []);

  useEffect(() => {
    onDraftChange?.({ name, rows, selectedUniverse, expectedStart });
  }, [expectedStart, name, onDraftChange, rows, selectedUniverse]);

  useEffect(() => {
    mountedRef.current = true;
    return () => {
      mountedRef.current = false;
      loadUniverseRequest.current += 1;
      directOperationEpoch.current += 1;
    };
  }, []);

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

  // A workspace can reopen directly on a saved universe. Summaries intentionally omit full
  // memberships/coverage, so hydrate the selected identity without replacing the saved draft.
  useEffect(() => {
    const savedName = draft?.selectedUniverse;
    if (
      !canSubmit ||
      !savedName ||
      selectedUniverse !== savedName ||
      loadedUniverse?.name === savedName
    ) return;
    const request = ++loadUniverseRequest.current;
    let cancelled = false;
    getUniverse(savedName)
      .then((universe) => {
        if (
          !cancelled &&
          request === loadUniverseRequest.current &&
          selectedUniverseRef.current === savedName
        ) {
          acceptRefreshedUniverse(universe);
        }
      })
      .catch(() => undefined);
    return () => {
      cancelled = true;
    };
  }, [
    acceptRefreshedUniverse,
    canSubmit,
    draft?.selectedUniverse,
    loadedUniverse?.name,
    selectedUniverse,
  ]);

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
    const request = ++loadUniverseRequest.current;
    selectUniverseIdentity(nameToLoad);
    if (!nameToLoad) {
      setLoadedUniverse(null);
      return;
    }
    if (!canSubmit) return;
    setUniverseError(null);
    setUniverseMessage(null);
    try {
      const universe = await getUniverse(nameToLoad);
      if (request !== loadUniverseRequest.current) return;
      setName(universe.name);
      setRows(rowsFromUniverse(universe));
      acceptRefreshedUniverse(universe);
      setPreparedPreset(null);
      setUniverseMessage(
        `Loaded ${universe.display_name ?? universe.name} with ${universe.symbols.length} symbols`,
      );
    } catch (error) {
      if (request !== loadUniverseRequest.current) return;
      setUniverseError(String(error));
    }
  }

  function newUniverse() {
    loadUniverseRequest.current += 1;
    setName("my-universe");
    setRows([{ ...EMPTY }]);
    selectUniverseIdentity("");
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
    loadUniverseRequest.current += 1;
    setUniverseError(null);
    setUniverseMessage(null);
    const importName = preset.pit_import_name ?? preset.id;
    setName(importName);
    setRows([{ ...EMPTY }]);
    selectUniverseIdentity("");
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
    const request = ++loadUniverseRequest.current;
    try {
      const response = isUpdatingLoaded
        ? await updateUniverse(payload.name, payload)
        : await defineUniverse(payload);
      if (request !== loadUniverseRequest.current) return;
      selectUniverseIdentity(response.name);
      setLoadedUniverse(null);
      setPreparedPreset(null);
      setUniverseMessage(`Saved ${response.name} with ${response.symbols.length} symbols`);
      await refreshUniverses();
      if (request !== loadUniverseRequest.current) return;
      try {
        const refreshed = await getUniverse(response.name);
        if (request === loadUniverseRequest.current) acceptRefreshedUniverse(refreshed);
      } catch {
        // The save succeeded. A later coverage refresh can hydrate older compatible backends.
      }
    } catch (error) {
      setUniverseError(String(error));
    }
  }

  async function removeUniverse() {
    if (!canSubmit || selectedInfo?.source !== "custom") return;
    const removing = selectedInfo.name;
    const request = ++loadUniverseRequest.current;
    setUniverseError(null);
    setUniverseMessage(null);
    try {
      await deleteUniverse(removing);
      if (
        request === loadUniverseRequest.current &&
        selectedUniverseRef.current === removing
      ) {
        selectUniverseIdentity("");
        setLoadedUniverse(null);
      }
      setUniverseMessage(`Deleted ${removing}`);
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
    const universeAtStart = selectedUniverseRef.current;
    const operation = ++directOperationEpoch.current;
    const isCurrent = () => (
      mountedRef.current &&
      operation === directOperationEpoch.current &&
      selectedUniverseRef.current === universeAtStart
    );
    try {
      const started = await startDataSync({ symbols: [symbol], start: expectedStart, mode: "incremental" });
      if (!isCurrent()) return;
      let job = await getDataSync(started.job_id);
      if (!isCurrent()) return;
      onPullProgress?.(job.progress ?? null);
      while (job.status === "queued" || job.status === "running" || job.status === "stopping") {
        await delay(500);
        if (!isCurrent()) return;
        job = await getDataSync(started.job_id);
        if (!isCurrent()) return;
        onPullProgress?.(job.progress ?? null);
      }
      if (job.status === "done" && universeAtStart) {
        const refreshed = await getUniverse(universeAtStart);
        if (isCurrent()) acceptRefreshedUniverse(refreshed);
      }
    } catch (error) {
      if (isCurrent()) setSymbolError(String(error));
    } finally {
      if (isCurrent()) onPullProgress?.(null);
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
    const universeAtStart = selectedUniverseRef.current;
    const operation = ++directOperationEpoch.current;
    const isCurrent = () => (
      mountedRef.current &&
      operation === directOperationEpoch.current &&
      selectedUniverseRef.current === universeAtStart
    );
    setMembershipSyncBusy(true);
    setMembershipSyncError(null);
    try {
      const started = await startMembershipSync({ symbols, expected_start: expectedStart });
      if (!isCurrent()) return;
      let job = await getMembershipSync(started.job_id);
      if (!isCurrent()) return;
      setMembershipSyncJob(job);
      onPullProgress?.(job.progress ?? null);
      while (job.status === "queued" || job.status === "running") {
        await delay(500);
        if (!isCurrent()) return;
        job = await getMembershipSync(started.job_id);
        if (!isCurrent()) return;
        setMembershipSyncJob(job);
        onPullProgress?.(job.progress ?? null);
      }
      if (job.status === "failed") {
        setMembershipSyncError(job.error ?? "Date sync failed.");
      }
    } catch (error) {
      if (isCurrent()) setMembershipSyncError(String(error));
    } finally {
      if (isCurrent()) {
        setMembershipSyncBusy(false);
        onPullProgress?.(null);
      }
    }
  }

  return (
    <section className="panel universe-editor-page" data-testid="universe-editor-page">
      <header className="panel-head">
        <div>
          <h1>Universe Editor</h1>
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
          title="Prepared universes"
          summary={`${presets.length} offline definitions`}
          defaultOpen
        >
          <details className="universe-preset-guidance">
            <summary>Current snapshot or point-in-time history?</summary>
            <p className="hint">
              A current snapshot answers who is included now. Applying that same list to earlier
              dates omits past removals, so use point-in-time history when historical membership
              matters. The two definitions remain separate.
            </p>
          </details>
          <div className="universe-preset-grid" data-testid="universe-presets">
            {presets.map((preset) => {
              const snapshot = preset.snapshot_universe ?? (preset.available ? preset.id : null);
              const definition = preset.definition;
              const provenance = preset.provenance_detail;
              const isSample = preset.status === "bundled_sample";
              return (
                <article
                  key={preset.id}
                  className={`universe-preset${preparedPreset === preset.id ? " is-prepared" : ""}`}
                >
                  <div className="universe-preset__head">
                    <strong>{preset.display_name}</strong>
                    <span
                      className="universe-preset__scope"
                      title={preset.warning}
                      aria-label={`${preset.warning} More detail is available below.`}
                    >
                      {isSample ? "Demo basket" : "Current members"} ⓘ
                    </span>
                  </div>
                  <p className="universe-preset__summary">{preset.coverage}</p>
                  <p className="universe-preset__facts">
                    <span>
                      {definition?.member_count != null
                        ? `${definition.member_count} symbols`
                        : "Bundled definition"}
                    </span>
                    <span>{definition?.snapshot_date ?? "Offline"}</span>
                  </p>
                  <div className="actions">
                    <button
                      type="button"
                      disabled={!snapshot}
                      onClick={() => loadPresetSnapshot(preset)}
                    >
                      {isSample ? "Load sample" : "Load current snapshot"}
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
                  <details className="universe-preset__details">
                    <summary>Source and research notes</summary>
                    <p>{preset.warning}</p>
                    <small>Source: {provenance?.provider ?? preset.provenance}</small>
                    {provenance?.attribution && (
                      <small>
                        Attribution: {provenance.attribution}
                        {provenance.license ? ` · ${provenance.license}` : ""}
                      </small>
                    )}
                    {preset.fingerprint && (
                      <code title={preset.fingerprint}>
                        Fingerprint: {preset.fingerprint}
                      </code>
                    )}
                  </details>
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
          {(cacheCoverage.exited_symbols?.length ?? 0) > 0 && (
            <small>
              {cacheCoverage.exited_symbols?.length} exited membership
              {cacheCoverage.exited_symbols?.length === 1 ? " requires" : "s require"} prices only through
              their declared exit dates. Missing provider data never creates an exit automatically.
            </small>
          )}
          <p className={cacheCoverage.complete ? "ok" : "oos-warning"}>
            {cacheCoverage.complete
              ? "Cached dates cover every declared membership interval."
              : `Missing or incomplete history: ${cacheProblems.slice(0, 8).join(", ") || "inspect coverage details before training"}${cacheProblems.length > 8 ? "…" : ""}`}
          </p>
        </aside>
      )}

      <UniverseDataSync
        key={`price-data-${selectedUniverse || "draft"}-${loadedSavedUniverse ? "loaded" : "loading"}`}
        rows={rows}
        universeName={selectedUniverse}
        universe={loadedSavedUniverse}
        savedDefinitionDirty={savedDefinitionDirty}
        recommendedStart={recommendedPriceStart}
        canSubmit={canSubmit}
        onPullProgress={onPullProgress}
        onUniverseRefresh={acceptRefreshedUniverse}
      />

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
          {membershipSyncBusy ? "Inspecting prices..." : "Inspect price dates"}
        </button>
      </div>
      <p className="hint">
        Price coverage is diagnostic only: its first row may reflect provider coverage, and a
        missing tail never proves a delisting. A membership exit means that a symbol left this
        universe; it does not necessarily mean the security stopped trading. Enter or import
        membership dates only from an authoritative constituent source.
      </p>
      {membershipSyncError && <p className="error">{membershipSyncError}</p>}

      {membershipDiagnostics.length > 0 && (
        <CompactSection
          title="Price-date diagnostics"
          summary={`${membershipDiagnostics.length} result${membershipDiagnostics.length === 1 ? "" : "s"}`}
          defaultOpen
        >
          <ul className="coverage-list" data-testid="stale-membership-list">
            {membershipDiagnostics.map((result) => (
              <li key={result.symbol}>
                <strong>{result.symbol}</strong>
                <span>
                  {result.status === "failed"
                    ? "Price coverage unavailable"
                    : `${result.first_date ?? result.list_date ?? "unknown"} to ${result.last_date ?? "unknown"}`}
                </span>
                <span>
                  {result.error ?? result.note ?? "Membership dates were left unchanged."}
                </span>
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
