// Extend > Sync Data: check Parquet-cache coverage for the current universe's symbols and
// pull fresh prices into the cache. Pure relocation of the original "Sync data" section.

import { useEffect, useState } from "react";
import { getDataCoverage, getDataSync, startDataSync } from "../api/client";
import type { DataCoverage, DataSyncJob, SyncProgressSnapshot } from "../api/types";
import { uniqueSymbols, type UniverseRow } from "./toUniversePayload";

const DEFAULT_SYNC_START = "2020-01-01";

function delay(ms: number): Promise<void> {
  return new Promise((resolve) => window.setTimeout(resolve, ms));
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
  const [syncStart, setSyncStart] = useState(DEFAULT_SYNC_START);
  const [syncEnd, setSyncEnd] = useState("");
  const [syncMode, setSyncMode] = useState<"incremental" | "refresh">("incremental");
  const [coverage, setCoverage] = useState<DataCoverage[]>([]);
  const [syncJob, setSyncJob] = useState<DataSyncJob | null>(null);
  const [syncError, setSyncError] = useState<string | null>(null);
  const [syncBusy, setSyncBusy] = useState(false);

  const symbols = uniqueSymbols(rows);
  const staleCount = coverage.filter((item) => item.needs_sync).length;
  const currentCount = coverage.length - staleCount;
  const syncSummary =
    syncJob?.status === "running" || syncJob?.status === "queued"
      ? "Sync running"
      : coverage.length > 0
        ? `${staleCount} stale, ${currentCount} current`
        : `${symbols.length} symbols`;

  // Leaving this page mid-sync stops the poll loop below - clear the bottom bar rather than
  // leaving it frozen on a stale snapshot.
  useEffect(() => () => onPullProgress?.(null), [onPullProgress]);

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
      onPullProgress?.(job.progress ?? null);
      for (let i = 0; i < 120 && job.status !== "done" && job.status !== "failed"; i += 1) {
        await delay(500);
        job = await getDataSync(started.job_id);
        setSyncJob(job);
        onPullProgress?.(job.progress ?? null);
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
      onPullProgress?.(null);
    }
  }

  return (
    <section className="panel" data-testid="sync-data-page">
      <header className="panel-head">
        <div>
          <h3>Sync data</h3>
          <p className="panel-note">
            Check Parquet-cache coverage for this universe's symbols, and pull fresh prices.
          </p>
        </div>
        <span className="mode-chip">{syncSummary}</span>
      </header>

      {!canSubmit && (
        <p className="panel-note">Data sync unlocks when the local backend is running.</p>
      )}

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
        <button type="button" onClick={checkCoverage} disabled={!canSubmit || symbols.length === 0}>
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
    </section>
  );
}
