// A thin progress bar shown at the very bottom of the frontend whenever a data pull
// (price sync, membership-date sync, or a single "pull now" ticker fetch) is in flight.

import type { SyncProgressSnapshot } from "../api/types";

export function DataPullProgress({ progress }: { progress: SyncProgressSnapshot | null }) {
  if (!progress || progress.total === 0 || progress.done >= progress.total) return null;
  const pct = Math.round((progress.done / progress.total) * 100);
  return (
    <div className="data-pull-progress" data-testid="data-pull-progress">
      <span className="data-pull-progress__label">
        Syncing {progress.current_symbol ?? "..."} ({progress.done}/{progress.total})
      </span>
      <div className="progress-bar" role="progressbar" aria-valuenow={pct}>
        <span className="progress-bar__fill" style={{ width: `${pct}%` }} />
      </div>
    </div>
  );
}
