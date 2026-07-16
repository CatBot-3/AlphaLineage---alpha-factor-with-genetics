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
  const best = progress?.best?.fitness;
  const activePhase = progress?.phase ?? phase;
  const resources = progress?.resources;
  const reportTotal = progress?.report_total ?? 0;
  const reportDone = progress?.report_done ?? 0;
  const showingReport = ["validating", "reporting", "finalizing"].includes(activePhase);
  const pct = showingReport && reportTotal > 0
    ? Math.round((reportDone / reportTotal) * 100)
    : target > 0
      ? Math.round((generation / target) * 100)
      : 0;
  const canStop =
    phase === "running" &&
    !["finalizing", "stopping", "done", "stopped", "failed"].includes(activePhase);

  return (
    <div className="progress-view" data-testid="progress-view">
      <div className="progress-head">
        <span data-testid="progress-generation">
          {showingReport && reportTotal > 0
            ? `validation ${reportDone} / ${reportTotal}`
            : `generation ${generation} / ${target}`}
        </span>
        <span className="progress-phase">{activePhase}</span>
        {canStop && (
          <button type="button" className="ghost" onClick={onStop} data-testid="stop-run">
            Stop
          </button>
        )}
      </div>
      <div className="progress-bar" role="progressbar" aria-valuenow={pct}>
        <span className="progress-bar__fill" style={{ width: `${pct}%` }} />
      </div>
      {resources && (
        <p className="progress-resources" data-testid="progress-resources">
          {resources.profile === "custom" ? `${resources.percent}% custom` : resources.profile}
          {" · "}{resources.workers} worker{resources.workers === 1 ? "" : "s"}
          {resources.accelerated ? " · accelerated" : " · Python fallback"}
        </p>
      )}
      {(progress?.candidate_total ?? 0) > 0 && (
        <p className="progress-detail" data-testid="candidate-progress">
          scoring candidates {progress?.candidate_done ?? 0} / {progress?.candidate_total}
          {progress?.factors_per_second
            ? ` · ${progress.factors_per_second.toFixed(1)} factors/s`
            : ""}
        </p>
      )}
      {(progress?.report_total ?? 0) > 0 && (
        <p className="progress-detail" data-testid="report-progress">
          {activePhase === "finalizing" ? "finalizing" : "building"} report{" "}
          {progress?.report_done ?? 0} / {progress?.report_total}
        </p>
      )}
      {activePhase === "finalizing" && (
        <p className="hint">Finalizing the locked report; stopping is disabled during this phase.</p>
      )}
      {progress?.termination_reason && progress.termination_reason !== "completed" && (
        <p className="progress-detail">ended early: {progress.termination_reason.replace("_", " ")}</p>
      )}
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
