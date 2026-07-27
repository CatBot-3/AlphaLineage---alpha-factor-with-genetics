// Embedded price-data sync controls for the Universe Editor.

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import * as apiClient from "../api/client";
import {
  getDataCoverage,
  getDataSync,
  getUniverse,
  startDataSync,
  stopDataSync,
} from "../api/client";
import type {
  DataCoverage,
  DataSyncJob,
  SyncProgressSnapshot,
  UniverseInfo,
} from "../api/types";
import { CompactSection } from "../app/CompactSection";
import { uniqueSymbols, type UniverseRow } from "./toUniversePayload";

const DEFAULT_SYNC_START = "2020-01-01";
const COVERAGE_CHUNK_SIZE = 75;

function delay(ms: number): Promise<void> {
  return new Promise((resolve) => window.setTimeout(resolve, ms));
}

function listAvailableDataSyncs(): Promise<DataSyncJob[]> {
  // Older embedded/demo clients may not expose the collection route. Starting and polling a
  // newly-created job still works on those clients.
  if (!("listDataSyncs" in apiClient)) return Promise.resolve([]);
  return apiClient.listDataSyncs({ activeOnly: true });
}

function isActive(job: DataSyncJob | null | undefined): boolean {
  return job?.status === "queued" || job?.status === "running" || job?.status === "stopping";
}

function upsertJob(current: DataSyncJob[], next: DataSyncJob): DataSyncJob[] {
  const existing = current.findIndex((item) => item.job_id === next.job_id);
  if (existing < 0) return [next, ...current];
  return current.map((item, index) => index === existing ? next : item);
}

function coverageFromUniverse(universe: UniverseInfo): DataCoverage[] {
  const coverage = universe.cache_coverage;
  if (!coverage) return [];
  const incomplete = new Set(
    coverage.incomplete_symbols ?? [
      ...coverage.missing_symbols,
      ...(coverage.invalid_symbols ?? []),
      ...(coverage.uncovered_symbols ?? []),
      ...(coverage.late_start_symbols ?? []),
      ...(coverage.stale_symbols ?? []),
    ],
  );
  const cached = new Set(coverage.cached_symbols);
  return coverage.eligible_symbols.map((symbol) => {
    const detail = coverage.symbol_coverage?.[symbol];
    return {
      symbol,
      cached: cached.has(symbol),
      rows: 0,
      first_date: detail?.first_date,
      last_date: detail?.last_date,
      requested_start: null,
      requested_end: coverage.as_of,
      needs_sync: incomplete.has(symbol),
    };
  });
}

async function getDraftCoverage(
  symbols: string[],
  start: string,
  end?: string,
): Promise<DataCoverage[]> {
  // Keep the compatibility GET endpoint, but bound each query URL for large unsaved drafts.
  const rows: DataCoverage[] = [];
  for (let offset = 0; offset < symbols.length; offset += COVERAGE_CHUNK_SIZE) {
    rows.push(...await getDataCoverage(
      symbols.slice(offset, offset + COVERAGE_CHUNK_SIZE),
      start,
      end,
    ));
  }
  return rows;
}

function normalizedSymbols(symbols: string[]): string[] {
  return [...new Set(symbols.map((symbol) => symbol.trim().toUpperCase()).filter(Boolean))].sort();
}

function sameSymbols(left: string[], right: string[]): boolean {
  const a = normalizedSymbols(left);
  const b = normalizedSymbols(right);
  return a.length === b.length && a.every((symbol, index) => symbol === b[index]);
}

function jobMatchesTarget(
  job: DataSyncJob,
  universeName: string,
  universeFingerprint: string | undefined,
  symbols: string[],
): boolean {
  const jobUniverse = job.request?.universe ?? job.result?.universe;
  if (universeName) {
    const jobFingerprint = job.universe_definition?.fingerprint ??
      job.result?.universe_definition?.fingerprint;
    if (universeFingerprint && jobFingerprint) return universeFingerprint === jobFingerprint;
    return jobUniverse === universeName;
  }
  if (jobUniverse) return false;
  const requested = job.request?.resolved_symbols ?? job.request?.symbols ??
    job.result?.resolved_symbols ?? job.result?.results.map((item) => item.symbol) ?? [];
  return sameSymbols(requested, symbols);
}

function jobTarget(job: DataSyncJob): string {
  const universe = job.request?.universe ?? job.result?.universe;
  if (universe) return universe;
  const count = job.request?.resolved_symbols?.length ?? job.request?.symbols?.length ??
    job.result?.resolved_symbols?.length ?? job.result?.results.length ?? 0;
  return `${count} symbol${count === 1 ? "" : "s"}`;
}

export function UniverseDataSync({
  rows,
  universeName = "",
  universe = null,
  savedDefinitionDirty = false,
  recommendedStart = DEFAULT_SYNC_START,
  canSubmit = true,
  onPullProgress,
  onUniverseRefresh,
}: {
  rows: UniverseRow[];
  universeName?: string;
  universe?: UniverseInfo | null;
  savedDefinitionDirty?: boolean;
  recommendedStart?: string;
  canSubmit?: boolean;
  onPullProgress?: (snapshot: SyncProgressSnapshot | null) => void;
  onUniverseRefresh?: (universe: UniverseInfo) => void;
}) {
  const draftSymbols = useMemo(() => uniqueSymbols(rows), [rows]);
  const targetIsSaved = Boolean(universeName);
  const targetSymbols = targetIsSaved ? universe?.symbols ?? draftSymbols : draftSymbols;
  const targetKey = targetIsSaved ? `universe:${universeName}` : `draft:${draftSymbols.join(",")}`;
  const targetLabel = targetIsSaved
    ? universe?.display_name ?? universeName
    : `Current draft (${draftSymbols.length} symbol${draftSymbols.length === 1 ? "" : "s"})`;
  const canSyncTarget = targetIsSaved
    ? universe?.name === universeName && !savedDefinitionDirty
    : draftSymbols.length > 0;

  const [syncStart, setSyncStart] = useState(recommendedStart || DEFAULT_SYNC_START);
  const [syncEnd, setSyncEnd] = useState("");
  const [syncMode, setSyncMode] = useState<"incremental" | "refresh">("incremental");
  const [coverage, setCoverage] = useState<DataCoverage[]>(() =>
    universe ? coverageFromUniverse(universe) : [],
  );
  const [jobs, setJobs] = useState<DataSyncJob[]>([]);
  const [syncJob, setSyncJob] = useState<DataSyncJob | null>(null);
  const [syncError, setSyncError] = useState<string | null>(null);
  const [syncNotice, setSyncNotice] = useState<string | null>(null);
  const [expanded, setExpanded] = useState(() => universe?.cache_coverage?.complete !== true);
  const mountedRef = useRef(true);
  const targetKeyRef = useRef(targetKey);
  targetKeyRef.current = targetKey;

  const targetIsCurrent = useCallback(
    () => mountedRef.current && targetKeyRef.current === targetKey,
    [targetKey],
  );

  const latestFailedCount = syncJob?.result?.failed_count ??
    syncJob?.result?.results.filter((item) => item.status === "failed").length ?? 0;

  const staleCount = coverage.filter((item) => item.needs_sync).length;
  const currentCount = coverage.length - staleCount;
  const activeJobs = jobs.filter(isActive);
  const syncSummary = isActive(syncJob)
    ? "Sync running"
    : syncJob?.status === "done" && latestFailedCount > 0
      ? `Completed with ${latestFailedCount} failure${latestFailedCount === 1 ? "" : "s"}`
    : coverage.length > 0
      ? `${staleCount} need sync, ${currentCount} current`
      : `${targetSymbols.length} symbol${targetSymbols.length === 1 ? "" : "s"}`;

  useEffect(() => {
    if (activeJobs.length > 0 || universe?.cache_coverage?.complete === false) {
      setExpanded(true);
    }
  }, [activeJobs.length, universe?.cache_coverage?.complete]);

  const refreshJobs = useCallback(async () => {
    if (!canSubmit) return;
    try {
      setJobs(await listAvailableDataSyncs());
    } catch {
      // Starting and polling a new job remains available on older compatible backends.
    }
  }, [canSubmit]);

  useEffect(() => {
    setSyncStart(recommendedStart || DEFAULT_SYNC_START);
    setSyncEnd("");
    setSyncError(null);
    setSyncNotice(null);
    setSyncJob((current) => current && jobMatchesTarget(
      current,
      universeName,
      universe?.fingerprint,
      draftSymbols,
    )
      ? current
      : null);
  }, [draftSymbols, recommendedStart, targetKey, universe?.fingerprint, universeName]);

  useEffect(() => {
    setCoverage(universeName && universe?.name === universeName
      ? coverageFromUniverse(universe)
      : []);
  }, [universe, universeName]);

  useEffect(() => {
    if (!canSubmit) {
      setJobs([]);
      return;
    }
    let cancelled = false;
    listAvailableDataSyncs()
      .then((items) => {
        if (!cancelled) setJobs(items);
      })
      .catch(() => undefined);
    return () => {
      cancelled = true;
    };
  }, [canSubmit]);

  // Automatically reattach to a job for the universe/draft currently being inspected.
  useEffect(() => {
    if (isActive(syncJob)) return;
    const matching = jobs.find((job) =>
      isActive(job) && jobMatchesTarget(job, universeName, universe?.fingerprint, draftSymbols),
    );
    if (matching) {
      setSyncJob(matching);
      onPullProgress?.(matching.progress ?? null);
    }
  }, [draftSymbols, jobs, onPullProgress, syncJob, universe?.fingerprint, universeName]);

  // Clear the global progress footer when Universe Editor disappears.
  useEffect(() => {
    mountedRef.current = true;
    return () => {
      mountedRef.current = false;
      onPullProgress?.(null);
    };
  }, [onPullProgress]);

  const checkCoverage = useCallback(async () => {
    if (!canSubmit || !canSyncTarget) return;
    setSyncError(null);
    setSyncNotice(null);
    try {
      if (targetIsSaved) {
        const detail = await getUniverse(universeName);
        if (!targetIsCurrent()) return;
        setCoverage(coverageFromUniverse(detail));
        onUniverseRefresh?.(detail);
      } else {
        const nextCoverage = await getDraftCoverage(
          draftSymbols,
          syncStart,
          syncEnd || undefined,
        );
        if (!targetIsCurrent()) return;
        setCoverage(nextCoverage);
      }
    } catch (error) {
      if (targetIsCurrent()) setSyncError(String(error));
    }
  }, [
    canSubmit,
    canSyncTarget,
    draftSymbols,
    onUniverseRefresh,
    syncEnd,
    syncStart,
    targetIsSaved,
    targetIsCurrent,
    universeName,
  ]);

  // Poll without an elapsed-time ceiling: full-index pulls may legitimately take hours.
  useEffect(() => {
    if (!syncJob || !isActive(syncJob)) return;
    let cancelled = false;
    const jobId = syncJob.job_id;

    async function poll() {
      try {
        await delay(500);
        if (cancelled) return;
        const next = await getDataSync(jobId);
        if (cancelled) return;
        setSyncJob(next);
        setJobs((current) => upsertJob(current, next));
        onPullProgress?.(next.progress ?? null);
        if (isActive(next)) {
          void poll();
        } else {
          onPullProgress?.(null);
          if (next.status === "done") await checkCoverage();
          if (next.status === "failed") setSyncError(next.error ?? "Data sync failed.");
        }
      } catch (error) {
        if (!cancelled) setSyncError(String(error));
      }
    }

    void poll();
    return () => {
      cancelled = true;
    };
  }, [checkCoverage, onPullProgress, syncJob?.job_id]);

  async function syncData() {
    if (!canSubmit || !canSyncTarget || !syncStart || isActive(syncJob)) return;
    setSyncError(null);
    setSyncJob(null);
    try {
      const started = await startDataSync({
        ...(targetIsSaved ? { universe: universeName } : { symbols: draftSymbols }),
        start: syncStart,
        end: syncEnd || undefined,
        mode: syncMode,
      });
      if (!targetIsCurrent()) return;
      if (started.reused) {
        setSyncNotice("Reattached to the existing matching sync job.");
      }
      const job = await getDataSync(started.job_id);
      if (!targetIsCurrent()) return;
      setSyncJob(job);
      setJobs((current) => upsertJob(current, job));
      onPullProgress?.(job.progress ?? null);
      if (!isActive(job)) {
        onPullProgress?.(null);
        if (job.status === "done") await checkCoverage();
        if (job.status === "failed") setSyncError(job.error ?? "Data sync failed.");
      }
    } catch (error) {
      if (targetIsCurrent()) {
        setSyncError(String(error));
        onPullProgress?.(null);
      }
    }
  }

  function reattach(job: DataSyncJob) {
    setSyncError(null);
    setSyncJob(job);
    onPullProgress?.(job.progress ?? null);
  }

  async function stopJob(job: DataSyncJob) {
    setSyncError(null);
    try {
      await stopDataSync(job.job_id);
      const next = await getDataSync(job.job_id);
      setJobs((current) => upsertJob(current, next));
      if (syncJob?.job_id === job.job_id) setSyncJob(next);
    } catch (error) {
      setSyncError(String(error));
    }
  }

  return (
    <details
      className="compact-section universe-price-section"
      data-testid="universe-data-sync"
      open={expanded}
      onToggle={(event) => setExpanded(event.currentTarget.open)}
    >
      <summary className="compact-section__trigger">
        <span className="compact-section__chevron" aria-hidden="true">{expanded ? "v" : ">"}</span>
        <span className="compact-section__title">Price data</span>
        <span className="compact-section__summary">{syncSummary}</span>
      </summary>
      <div className="compact-section__body universe-data-sync">
      <div className="universe-data-sync__head">
        <div>
          <h4>Price data sync</h4>
          <p className="hint">
            Target: <strong>{targetLabel}</strong>. Incremental sync keeps existing cached rows and
            resumes missing date ranges.
          </p>
        </div>
      </div>

      {!canSubmit && (
        <p className="panel-note">Price data sync unlocks when the local backend is running.</p>
      )}

      {targetIsSaved && !universe && (
        <p className="hint">Loading the saved universe definition before price sync…</p>
      )}
      {targetIsSaved && universe && savedDefinitionDirty && (
        <p className="error" role="status">
          Save this universe before syncing. The displayed name or memberships differ from the
          loaded saved definition.
        </p>
      )}
      {!targetIsSaved && (
        <p className="hint">
          Draft sync caches these symbols only. Save and load the universe definition to verify
          point-in-time membership readiness.
        </p>
      )}

      <div className="inline-tools universe-data-sync__controls">
        <label className="field">
          <span className="field-label">Required start date</span>
          <input
            type="date"
            aria-label="Sync start date"
            value={syncStart}
            onChange={(event) => setSyncStart(event.target.value)}
          />
        </label>
        <label className="field">
          <span className="field-label">End date</span>
          <input
            type="date"
            aria-label="Sync end date"
            value={syncEnd}
            onChange={(event) => setSyncEnd(event.target.value)}
          />
        </label>
        <label className="check-row">
          <input
            type="checkbox"
            aria-label="Incremental merge"
            checked={syncMode === "incremental"}
            onChange={(event) => setSyncMode(event.target.checked ? "incremental" : "refresh")}
          />
          Incremental merge
        </label>
        <button type="button" onClick={checkCoverage} disabled={!canSubmit || !canSyncTarget}>
          Refresh coverage
        </button>
        <button
          type="button"
          data-testid="sync-universe"
          onClick={syncData}
          disabled={!canSubmit || !canSyncTarget || !syncStart || isActive(syncJob)}
        >
          {isActive(syncJob)
            ? "Syncing..."
            : targetIsSaved && savedDefinitionDirty
              ? "Save universe before syncing"
            : targetIsSaved
              ? "Sync this universe"
              : "Sync draft symbols"}
        </button>
      </div>

      {targetIsSaved && universe && (
        <p className="hint" data-testid="sync-universe-definition">
          {(universe.mode ?? "point_in_time").replace(/_/g, " ")} definition
          {universe.fingerprint ? ` · ${universe.fingerprint}` : ""}
          {Object.keys(universe.aliases ?? {}).length > 0
            ? ` · ${Object.keys(universe.aliases ?? {}).length} provider alias(es)`
            : ""}
          {universe.cache_coverage?.required_start
            ? ` · required ${universe.cache_coverage.required_start} through ${universe.cache_coverage.required_end ?? universe.cache_coverage.as_of}`
            : ""}
        </p>
      )}

      {syncNotice && <p className="ok" role="status">{syncNotice}</p>}
      {syncJob?.status === "done" && latestFailedCount > 0 && (
        <p className="error" role="status">
          Sync completed with {latestFailedCount} failed symbol{latestFailedCount === 1 ? "" : "s"}.
          Coverage remains incomplete until those symbols succeed.
        </p>
      )}

      {activeJobs.length > 0 && (
        <section data-testid="active-sync-jobs">
          <div className="universe-data-sync__subhead">
            <div>
              <h5>Active sync jobs</h5>
              <p className="hint">Jobs survive navigation. Reattach or stop without losing cached rows.</p>
            </div>
            <button type="button" className="ghost" onClick={refreshJobs}>Refresh jobs</button>
          </div>
          <ul className="coverage-list universe-data-sync__list">
            {activeJobs.map((job) => (
              <li key={job.job_id}>
                <strong>{jobTarget(job)}</strong>
                <span>{job.stopping ? "stopping" : job.status}</span>
                <span>{job.progress ? `${job.progress.done} / ${job.progress.total}` : "Waiting"}</span>
                <div className="actions">
                  <button type="button" onClick={() => reattach(job)}>
                    {syncJob?.job_id === job.job_id ? "Attached" : "Reattach"}
                  </button>
                  <button
                    type="button"
                    className="danger-navy"
                    disabled={job.stopping}
                    onClick={() => stopJob(job)}
                  >
                    {job.stopping ? "Stopping..." : "Stop"}
                  </button>
                </div>
              </li>
            ))}
          </ul>
        </section>
      )}

      {coverage.length > 0 && (
        <CompactSection
          title="Symbol coverage"
          summary={`${staleCount} need sync · ${currentCount} current`}
          defaultOpen={staleCount > 0 && coverage.length <= 20}
        >
          <ul className="coverage-list universe-data-sync__list" data-testid="coverage-list">
            {[...coverage]
              .sort((left, right) => Number(right.needs_sync) - Number(left.needs_sync))
              .map((item) => (
                <li key={item.symbol}>
                  <strong>{item.symbol}</strong>
                  <span>{item.cached ? `${item.rows || "Some"} cached rows` : "No cached rows"}</span>
                  <span>{item.needs_sync ? "Needs sync" : "Current"}</span>
                  <span>
                    {item.first_date && item.last_date
                      ? `${item.first_date} to ${item.last_date}`
                      : "No date range"}
                  </span>
                </li>
              ))}
          </ul>
        </CompactSection>
      )}

      {syncJob?.result && (
        <CompactSection
          title="Latest sync result"
          summary={`${syncJob.result.results.filter((item) => item.status !== "failed").length} completed · ${syncJob.result.results.filter((item) => item.status === "failed").length} failed`}
          defaultOpen={syncJob.result.results.some((item) => item.status === "failed")}
        >
          <ul className="coverage-list universe-data-sync__list" data-testid="sync-results">
            {syncJob.result.results.map((result) => (
              <li key={result.symbol}>
                <strong>{result.symbol}</strong>
                <span>{result.status}</span>
                <span>{result.rows_fetched} fetched</span>
                <span>{result.error ?? `${result.rows_cached} cached`}</span>
                {result.provider_symbol && result.provider_symbol !== result.symbol && (
                  <span>Provider symbol: {result.provider_symbol}</span>
                )}
              </li>
            ))}
          </ul>
        </CompactSection>
      )}
      {syncError && <p className="error">{syncError}</p>}
      </div>
    </details>
  );
}
