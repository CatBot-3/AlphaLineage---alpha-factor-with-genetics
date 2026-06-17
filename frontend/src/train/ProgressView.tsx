// Live progress for a running segment: a generation bar, a best/mean-fitness sparkline,
// and a Stop control. Reads the snapshot the backend updates as the GP evolves.

import type { ProgressSnapshot } from "../api/types";
import { Sparkline } from "../dashboard/Sparkline";

export function ProgressView({
  progress,
  phase,
  onStop,
}: {
  progress: ProgressSnapshot | null;
  phase: string;
  onStop: () => void;
}) {
  const generation = progress?.generation ?? 0;
  const target = progress?.target_generations ?? 0;
  const pct = target > 0 ? Math.round((generation / target) * 100) : 0;
  const best = progress?.best?.fitness;

  return (
    <div className="progress-view" data-testid="progress-view">
      <div className="progress-head">
        <span data-testid="progress-generation">
          generation {generation} / {target}
        </span>
        <span className="progress-phase">{phase}</span>
        {phase === "running" && (
          <button type="button" className="ghost" onClick={onStop} data-testid="stop-run">
            Stop
          </button>
        )}
      </div>
      <div className="progress-bar" role="progressbar" aria-valuenow={pct}>
        <span className="progress-bar__fill" style={{ width: `${pct}%` }} />
      </div>
      {best !== undefined && (
        <p className="progress-best">best fitness so far: {best.toFixed(4)}</p>
      )}
      {progress && progress.history.length > 0 && (
        <div className="progress-spark">
          <span className="history-label">Best fitness per generation</span>
          <Sparkline values={progress.history.map((h) => h.best_fitness)} />
        </div>
      )}
    </div>
  );
}
