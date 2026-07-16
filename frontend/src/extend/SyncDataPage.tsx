// Extend > Sync Data: inspect price readiness and manage resumable cache-sync jobs.

import { useCallback, useEffect, useMemo, useState } from "react";
import * as apiClient from "../api/client";
import {
  getDataCoverage,
  getDataSync,
  getUniverse,
  listUniverses,
  startDataSync,
  stopDataSync,
} from "../api/client";
import type {
  DataCoverage,
  DataSyncJob,
  SyncProgressSnapshot,
  UniverseInfo,
} from "../api/types";
import { uniqueSymbols, type UniverseRow } from "./toUniversePayload";

const DEFAULT_SYNC_START = "2020-01-01";
const DRAFT_TARGET = "__draft__";

function delay(ms: number): Promise<void> {
  return new Promise((resolve) => window.setTimeout(resolve, ms));
}

function listAvailableDataSyncs(): Promise<DataSyncJob[]> {
  // Keeps the relocated page compatible with older embedded/demo clients and partial test hosts.
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

function jobTarget(job: DataSyncJob): string {
  const universe = job.request?.universe ?? job.result?.universe;
  if (universe) return universe;
  const count = job.request?.resolved_symbols?.length ?? job.request?.symbols?.length ??
    job.result?.resolved_symbols?.length ?? job.result?.results.length ?? 0;
  return `${count} symbol${count === 1 ? "" : "s"}`;
}

export function SyncDataPage({
  rows,
  canSubmit = true,
  onPullProgress,
}: {
  rows: UniverseRow[];
  canSubmit?: boolean;
  onPullProgress?: (snapshot: SyncProgressSnapshot | null) => void;
}) {
  const draftSymbols = useMemo(() => uniqueSymbols(rows), [rows]);
  const [syncTarget, setSyncTarget] = useState(DRAFT_TARGET);
  const [universes, setUniverses] = useState<UniverseInfo[]>([]);
  const [universeDetail, setUniverseDetail] = useState<UniverseInfo | null>(null);
  const [syncStart, setSyncStart] = useState(DEFAULT_SYNC_START);
  const [syncEnd, setSyncEnd] = useState("");
  const [syncMode, setSyncMode] = useState<"incremental" | "refresh">("incremental");
  const [coverage, setCoverage] = useState<DataCoverage[]>([]);
  const [jobs, setJobs] = useState<DataSyncJob[]>([]);
  const [syncJob, setSyncJob] = useState<DataSyncJob | null>(null);
  const [syncError, setSyncError] = useState<string | null>(null);

  const selectedSummary = universes.find((item) => item.name === syncTarget);
  const selectedUniverse = universeDetail?.name === syncTarget ? universeDetail : selectedSummary;
  const targetSymbols = syncTarget === DRAFT_TARGET
    ? draftSymbols
    : selectedUniverse?.symbols ?? [];
  const canSyncTarget = syncTarget === DRAFT_TARGET ? targetSymbols.length > 0 : Boolean(syncTarget);
  const staleCount = coverage.filter((item) => item.needs_sync).length;
  const currentCount = coverage.length - staleCount;
  const activeJobs = jobs.filter(isActive);
  const syncSummary = isActive(syncJob)
    ? "Sync running"
    : coverage.length > 0
      ? `${staleCount} stale, ${currentCount} current`
      : syncTarget === DRAFT_TARGET
        ? `${targetSymbols.length} draft symbols`
        : `${selectedUniverse?.definition?.member_count ?? targetSymbols.length} universe symbols`;

  const refreshJobs = useCallback(async () => {
    if (!canSubmit) return;
    try {
      setJobs(await listAvailableDataSyncs());
    } catch {
      // Older compatible backends do not expose the collection route. New jobs still work.
    }
  }, [canSubmit]);

  useEffect(() => {
    if (!canSubmit) {
      setUniverses([]);
      return;
    }
    let cancelled = false;
    Promise.allSettled([
      listUniverses({ summary: true }),
      listAvailableDataSyncs(),
    ]).then(([universeResult, jobsResult]) => {
      if (cancelled) return;
      if (universeResult.status === "fulfilled") setUniverses(universeResult.value);
      if (jobsResult.status === "fulfilled") setJobs(jobsResult.value);
    });
    return () => {
      cancelled = true;
    };
  }, [canSubmit]);

  useEffect(() => {
    if (!canSubmit || syncTarget === DRAFT_TARGET) {
      setUniverseDetail(null);
      return;
    }
    let cancelled = false;
    getUniverse(syncTarget)
      .then((detail) => {
        if (!cancelled) setUniverseDetail(detail);
      })
      .catch((error) => {
        if (!cancelled) setSyncError(String(error));
      });
    return () => {
      cancelled = true;
    };
  }, [canSubmit, syncTarget]);

  // Clear the global progress footer when this page disappears.
  useEffect(() => () => onPullProgress?.(null), [onPullProgress]);

  const checkCoverage = useCallback(async () => {
    if (!canSubmit || !canSyncTarget) return;
    setSyncError(null);
    try {
      if (syncTarget === DRAFT_TARGET) {
        setCoverage(await getDataCoverage(targetSymbols, syncStart, syncEnd || undefined));
      } else {
        const detail = await getUniverse(syncTarget);
        setUniverseDetail(detail);
        setCoverage(coverageFromUniverse(detail));
      }
    } catch (error) {
      setSyncError(String(error));
    }
  }, [canSubmit, canSyncTarget, syncEnd, syncStart, syncTarget, targetSymbols]);

  // Poll an attached job until the backend reports a terminal status. There is deliberately no
  // elapsed-time cap: large index pulls can run for hours and remain reattachable after navigation.
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
          if (next.status === "done") void checkCoverage();
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
    if (!canSubmit || !canSyncTarget || isActive(syncJob)) return;
    setSyncError(null);
    setSyncJob(null);
    try {
      const started = await startDataSync({
        ...(syncTarget === DRAFT_TARGET
          ? { symbols: targetSymbols }
          : { universe: syncTarget }),
        start: syncStart,
        end: syncEnd || undefined,
        mode: syncMode,
      });
      const job = await getDataSync(started.job_id);
      setSyncJob(job);
      setJobs((current) => upsertJob(current, job));
      onPullProgress?.(job.progress ?? null);
      if (!isActive(job)) {
        onPullProgress?.(null);
        if (job.status === "done") await checkCoverage();
        if (job.status === "failed") setSyncError(job.error ?? "Data sync failed.");
      }
    } catch (error) {
      setSyncError(String(error));
      onPullProgress?.(null);
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
    <section className="panel" data-testid="sync-data-page">
      <header className="panel-head">
        <div>
          <h3>Sync data</h3>
          <p className="panel-note">
            Select a saved universe to preserve its definition and symbol aliases, or sync the
            unsaved draft symbols directly.
          </p>
        </div>
        <span className="mode-chip">{syncSummary}</span>
      </header>

      {!canSubmit && (
        <p className="panel-note">Data sync unlocks when the local backend is running.</p>
      )}

      <div className="inline-tools">
        <label className="field">
          <span className="field-label">Sync target</span>
          <select
            aria-label="Sync target"
            value={syncTarget}
            onChange={(event) => {
              setSyncTarget(event.target.value);
              setCoverage([]);
              setSyncError(null);
            }}
            disabled={!canSubmit}
          >
            <option value={DRAFT_TARGET}>Current draft ({draftSymbols.length} symbols)</option>
            {universes.map((universe) => (
              <option key={universe.name} value={universe.name}>
                {universe.display_name ?? universe.name} ({universe.mode?.replace(/_/g, " ") ?? universe.source})
              </option>
            ))}
          </select>
        </label>
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

      {syncTarget !== DRAFT_TARGET && selectedUniverse && (
        <aside className="universe-integrity" data-testid="sync-universe-definition">
          <strong>{selectedUniverse.display_name ?? selectedUniverse.name}</strong>
          <span>
            Mode: {selectedUniverse.mode?.replace(/_/g, " ") ?? "point in time"} · Source: {selectedUniverse.source}
          </span>
          <span>
            Definition: {selectedUniverse.definition?.display_name ?? selectedUniverse.name} ·
            Fingerprint: {selectedUniverse.fingerprint ?? "unavailable"}
          </span>
          <span>
            Readiness: {selectedUniverse.readiness?.price_ready ?? selectedUniverse.cache_coverage?.complete
              ? "prices ready"
              : "price sync needed"}
          </span>
          {Object.keys(selectedUniverse.aliases ?? {}).length > 0 && (
            <small>
              {Object.keys(selectedUniverse.aliases ?? {}).length} provider alias(es): {Object.entries(selectedUniverse.aliases ?? {})
                .slice(0, 4).map(([symbol, alias]) => `${symbol} → ${alias}`).join(", ")}
            </small>
          )}
        </aside>
      )}

      <div className="actions">
        <button type="button" onClick={checkCoverage} disabled={!canSubmit || !canSyncTarget}>
          Check coverage
        </button>
        <button
          type="button"
          data-testid="sync-universe"
          onClick={syncData}
          disabled={!canSubmit || !canSyncTarget || isActive(syncJob)}
        >
          {isActive(syncJob) ? "Syncing..." : "Sync selected universe"}
        </button>
      </div>

      {activeJobs.length > 0 && (
        <section data-testid="active-sync-jobs">
          <div className="panel-head">
            <div>
              <h4>Active sync jobs</h4>
              <p className="hint">Reattach after navigation, or stop a pull without losing cached rows.</p>
            </div>
            <button type="button" className="ghost" onClick={refreshJobs}>Refresh</button>
          </div>
          <ul className="coverage-list">
            {activeJobs.map((job) => (
              <li key={job.job_id}>
                <strong>{jobTarget(job)}</strong>
                <span>{job.stopping ? "stopping" : job.status}</span>
                <span>{job.progress ? `${job.progress.done} / ${job.progress.total}` : "Waiting for progress"}</span>
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
        <ul className="coverage-list" data-testid="coverage-list">
          {coverage.map((item) => (
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
      )}

      {syncJob?.result && (
        <ul className="coverage-list" data-testid="sync-results">
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
      )}
      {syncError && <p className="error">{syncError}</p>}
    </section>
  );
}
