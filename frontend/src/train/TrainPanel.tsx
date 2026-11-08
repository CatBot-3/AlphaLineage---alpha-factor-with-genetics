// The Train tab: launch a session, watch it run, and once done continue it with changed
// settings or open the results. The completed segment's result is lifted to the App so the
// existing Dashboard / Factor / Genealogy views render it unchanged.

import { useEffect, useRef, useState } from "react";
import { getSession, listSessions } from "../api/client";
import type {
  RunResult,
  SessionState,
  SessionSummary,
  TrainingResourcesRequest,
} from "../api/types";
import { ProgressView } from "./ProgressView";
import { RunConfigForm, type RunRequestForm } from "./RunConfigForm";
import { useSession, type SessionController } from "./useSession";
import { executionTimingLabel } from "./defaults";
import { TrainingResourcePicker } from "./TrainingResourcePicker";

export function TrainPanel({
  seedIds = [],
  controller,
  restoreSessionId = null,
  onComplete,
  onRunningChange,
  onOpenDashboard,
  onOpenUniverseEditor,
  onOpenDataSync,
  onOpenFormulaEditor,
}: {
  seedIds?: string[];
  controller?: SessionController;
  restoreSessionId?: string | null;
  onComplete?: (result: RunResult) => void;
  onRunningChange?: (running: boolean, sessionId: string | null) => void;
  onOpenDashboard?: () => void;
  onOpenUniverseEditor?: (universeName: string) => void;
  onOpenDataSync?: (universeName: string) => void;
  onOpenFormulaEditor?: () => void;
}) {
  // App mode supplies an application-owned controller so polling survives tab changes.
  // The local controller keeps TrainPanel independently testable and compatible for embeds.
  const localController = useSession(controller ? undefined : onComplete);
  const {
    sessionId, state, error, notice, phase, connection, submitting, start, cont, stop, attach, reset,
  } = controller ?? localController;
  const [moreGenerations, setMoreGenerations] = useState(5);
  const [overrideResources, setOverrideResources] = useState(false);
  const [continueResources, setContinueResources] = useState<TrainingResourcesRequest>({
    profile: "auto",
    cpu_budget_percent: null,
  });
  const [recentSessions, setRecentSessions] = useState<SessionSummary[]>([]);
  const [defaultSession, setDefaultSession] = useState<SessionState | null>(null);

  // On reload, re-attach to a persisted session so an in-flight run keeps streaming progress.
  const restored = useRef(false);
  useEffect(() => {
    if (!controller && restoreSessionId && !restored.current) {
      restored.current = true;
      attach(restoreSessionId);
    }
  }, [controller, restoreSessionId, attach]);

  // Surface whether a search is in progress (and which session) so Quit can warn / stop it.
  useEffect(() => {
    onRunningChange?.(phase === "running" || Boolean(submitting), sessionId);
  }, [phase, submitting, sessionId, onRunningChange]);

  useEffect(() => {
    let active = true;
    listSessions()
      .then(async (items) => {
        if (!active) return;
        setRecentSessions(items);
        const latest = items.find(
          (item) => item.has_checkpoint || item.latest_completed_round != null,
        );
        if (!latest) return;
        try {
          const previous = await getSession(latest.id);
          if (active) setDefaultSession(previous);
        } catch {
          // The list and detail can race a manual cache clear. The ordinary defaults remain.
        }
      })
      .catch(() => {
        if (active) setRecentSessions([]);
      });
    return () => {
      active = false;
    };
  }, [state?.segments.length]);

  const running = phase === "running";
  // A run can stop before its first report exists; its persisted checkpoint/session is still
  // valid and must remain resumable.
  const interrupted = phase === "interrupted";
  const done = phase === "done" || phase === "stopped" ||
    ((interrupted || phase === "failed") && state?.recovery?.can_resume === true);
  const connected = connection === "connected";
  const hasCompletedRound = Boolean(
    state?.rounds?.length || state?.result?.selection || state?.result?.report,
  );
  const lastSegment = state?.segments[state.segments.length - 1];
  const latestRound = state?.rounds?.[state.rounds.length - 1];
  const restartRequired = latestRound?.restart_required === true;
  const reportCancelled = phase === "stopped" && Boolean(lastSegment?.report_cancelled);

  useEffect(() => {
    if (interrupted && state?.recovery) {
      setMoreGenerations(Math.max(1,
        state.recovery.target_generations - state.recovery.generation));
      return;
    }
    if (!lastSegment) return;
    const generationDelta =
      Number.isFinite(lastSegment.gen_end) && Number.isFinite(lastSegment.gen_start)
        ? lastSegment.gen_end - lastSegment.gen_start
        : 5;
    const previousSize =
      lastSegment.requested_generations ??
      Math.max(1, generationDelta);
    setMoreGenerations(previousSize);
  }, [interrupted, state?.recovery?.generation, state?.recovery?.target_generations,
      lastSegment?.index, lastSegment?.requested_generations]);

  function handleStart(req: RunRequestForm) {
    void start({
      name: req.name,
      universe: req.universe,
      as_of: req.as_of,
      config: req.config,
      resources: req.resources,
      seed_factor_ids: req.seed_factor_ids,
    });
  }

  return (
    <div className="train-panel" data-testid="train-panel">
      {notice && <p className="surface-message" data-testid="train-notice">{notice}</p>}
      {connection === "reconnecting" && (
        <p className="surface-message" role="status" data-testid="connection-notice">
          Connection lost—reconnecting. Your last reported progress is shown below.
        </p>
      )}
      {interrupted && (
        <p className="surface-message" role="status">
          {state?.recovery?.can_resume
            ? "The app stopped during training. Your saved checkpoint is available to resume."
            : state?.recovery?.restart_reason ??
              "The app stopped before a recovery checkpoint was available. Start a new session with the saved settings."}
        </p>
      )}

      {recentSessions.length > 0 && (
        <label className="field recent-session-picker">
          <span className="field-label">Recent session</span>
          <select
            aria-label="Recent session"
            value={sessionId ?? ""}
            onChange={(event) => {
              if (event.target.value) attach(event.target.value);
            }}
          >
            <option value="">Choose a saved session</option>
            {recentSessions.map((item) => (
              <option key={item.id} value={item.id}>
                {item.name} · {item.current_generation ?? 0} generations ·{" "}
                {item.latest_completed_round !== null &&
                item.latest_completed_round !== undefined
                  ? `${item.latest_completed_round + 1} rounds`
                  : item.last_status ?? "saved"}
              </option>
            ))}
          </select>
        </label>
      )}

      {!sessionId && (
        <RunConfigForm
          initialSeedIds={seedIds}
          initialSession={defaultSession}
          onStart={handleStart}
          disabled={running || !connected || Boolean(submitting)}
          onEditUniverse={onOpenUniverseEditor}
          onOpenDataSync={onOpenDataSync}
          onOpenFormulaEditor={onOpenFormulaEditor}
        />
      )}

      {error && <p className="error surface-message">{error}</p>}

      {sessionId && (
        <section className="train-status">
          <div className="train-status-head">
            <ProgressView progress={state?.job?.progress ?? null} phase={phase} onStop={stop} connected={connected} />
            <button
              type="button"
              className="ghost"
              data-testid="new-session"
              disabled={Boolean(submitting)}
              onClick={() => {
                if (state?.segments.length) setDefaultSession(state);
                reset();
              }}
            >
              New session
            </button>
          </div>

          {state && (
            <dl className="train-counts" data-testid="train-counts">
              <div>
                <dt>completed rounds</dt>
                <dd>
                  {state.rounds?.length ??
                    state.segments.filter((segment) => segment.status === "done").length}
                </dd>
              </div>
              <div>
                <dt>cumulative trials</dt>
                <dd>{state.cumulative_trials}</dd>
              </div>
              <div>
                <dt>explicit holdout reads</dt>
                <dd>{state.session_holdout_reads ?? state.test_reads}</dd>
              </div>
              <div>
                <dt>execution timing</dt>
                <dd data-testid="session-execution">
                  {executionTimingLabel(state.config?.execution)}
                </dd>
              </div>
            </dl>
          )}

          {reportCancelled && (
            <p className="hint" role="status" data-testid="report-cancelled-note">
              Validation was cancelled before a new round could be committed. The checkpoint is
              saved, and every earlier completed round was retained.
            </p>
          )}

          {done && (
            <div className="train-continue" data-testid="train-continue">
              <h4>
                {restartRequired
                  ? "Restart under the corrected training semantics"
                  : interrupted ? "Resume from the saved generation" : "Continue training from this generation"}
              </h4>
              {restartRequired && (
                <p className="oos-warning" role="alert">
                  {latestRound?.invalid_reason ??
                    "This checkpoint predates the corrected scorer and cannot be continued safely."}
                </p>
              )}
              {!restartRequired && (
                <>
              <label className="field">
                <span className="field-label">Additional generations</span>
                <input
                  type="number"
                  min={1}
                  aria-label="Additional generations"
                  value={moreGenerations}
                  disabled={!connected || Boolean(submitting)}
                  onChange={(e) => setMoreGenerations(Number(e.target.value))}
                />
              </label>
              <p className="hint" data-testid="continue-target">
                Will finish at generation {(state?.recovery?.generation ??
                  state?.job?.progress?.generation ?? 0) + moreGenerations}.
              </p>
              <label className="seed-option resource-override-toggle">
                <input
                  type="checkbox"
                  checked={overrideResources}
                  onChange={(event) => setOverrideResources(event.target.checked)}
                />
                <span>Change computer resources for this segment</span>
              </label>
              {overrideResources ? (
                <TrainingResourcePicker
                  value={continueResources}
                  onChange={setContinueResources}
                  label="Continue resources"
                />
              ) : (
                <p className="hint">Uses this session's current resource profile.</p>
              )}
              <div className="train-actions">
                <button
                  type="button"
                  className="ghost"
                  data-testid="continue-run"
                  disabled={!connected || Boolean(submitting)}
                  onClick={() =>
                    void cont({
                      generations: moreGenerations,
                      ...(overrideResources ? { resources: continueResources } : {}),
                    })
                  }
                >
                  {submitting === "continue" ? "Resuming…" : interrupted ? "Resume" : "Continue"}
                </button>
                {hasCompletedRound && (
                  <button type="button" className="primary-action" onClick={onOpenDashboard}>
                    Review result
                  </button>
                )}
              </div>
              <p className="hint">
                Continuing creates another validation-only round from the latest checkpoint. It
                does not read the frozen holdout; finalize a selected round explicitly when you
                are ready to inspect out-of-sample performance.
              </p>
                </>
              )}
              {restartRequired && (
                <div className="train-actions">
                  <button
                    type="button"
                    className="primary-action"
                    data-testid="restart-session"
                    onClick={() => {
                      if (state) setDefaultSession(state);
                      reset();
                    }}
                  >
                    Restart with same setup
                  </button>
                  {hasCompletedRound && (
                    <button type="button" className="ghost" onClick={onOpenDashboard}>
                      View invalid legacy report
                    </button>
                  )}
                </div>
              )}
            </div>
          )}
        </section>
      )}
    </div>
  );
}
