// The Train tab: launch a session, watch it run, and once done continue it with changed
// settings or open the results. The completed segment's result is lifted to the App so the
// existing Dashboard / Factor / Genealogy views render it unchanged.

import { useEffect, useRef, useState } from "react";
import type { RunResult } from "../api/types";
import { ProgressView } from "./ProgressView";
import { RunConfigForm, type RunRequestForm } from "./RunConfigForm";
import { useSession } from "./useSession";

export function TrainPanel({
  seedIds = [],
  restoreSessionId = null,
  onComplete,
  onRunningChange,
  onOpenDashboard,
  onOpenUniverseEditor,
}: {
  seedIds?: string[];
  restoreSessionId?: string | null;
  onComplete?: (result: RunResult) => void;
  onRunningChange?: (running: boolean, sessionId: string | null) => void;
  onOpenDashboard?: () => void;
  onOpenUniverseEditor?: (universeName: string) => void;
}) {
  const { sessionId, state, error, notice, phase, start, cont, stop, attach, reset } =
    useSession(onComplete);
  const [moreGenerations, setMoreGenerations] = useState(5);

  // On reload, re-attach to a persisted session so an in-flight run keeps streaming progress.
  const restored = useRef(false);
  useEffect(() => {
    if (restoreSessionId && !restored.current) {
      restored.current = true;
      attach(restoreSessionId);
    }
  }, [restoreSessionId, attach]);

  // Surface whether a search is in progress (and which session) so Quit can warn / stop it.
  useEffect(() => {
    onRunningChange?.(phase === "running", sessionId);
  }, [phase, sessionId, onRunningChange]);

  const running = phase === "running";
  const done = phase === "done";

  function handleStart(req: RunRequestForm) {
    void start({
      name: req.name,
      universe: req.universe,
      config: req.config,
      seed_factor_ids: req.seed_factor_ids,
    });
  }

  return (
    <div className="train-panel" data-testid="train-panel">
      {notice && <p className="surface-message" data-testid="train-notice">{notice}</p>}

      {!sessionId && (
        <RunConfigForm
          initialSeedIds={seedIds}
          onStart={handleStart}
          disabled={running}
          onEditUniverse={onOpenUniverseEditor}
        />
      )}

      {error && <p className="error surface-message">{error}</p>}

      {sessionId && (
        <section className="train-status">
          <div className="train-status-head">
            <ProgressView progress={state?.job?.progress ?? null} phase={phase} onStop={stop} />
            <button
              type="button"
              className="ghost"
              data-testid="new-session"
              onClick={reset}
            >
              New session
            </button>
          </div>

          {state && (
            <dl className="train-counts" data-testid="train-counts">
              <div>
                <dt>segments</dt>
                <dd>{state.segments.length}</dd>
              </div>
              <div>
                <dt>cumulative trials</dt>
                <dd>{state.cumulative_trials}</dd>
              </div>
              <div>
                <dt>OOS reads</dt>
                <dd>{state.test_reads}</dd>
              </div>
            </dl>
          )}

          {done && (
            <div className="train-continue" data-testid="train-continue">
              <h4>Continue training from this generation</h4>
              <label className="field">
                <span className="field-label">Additional generations</span>
                <input
                  type="number"
                  min={1}
                  aria-label="Additional generations"
                  value={moreGenerations}
                  onChange={(e) => setMoreGenerations(Number(e.target.value))}
                />
              </label>
              <div className="train-actions">
                <button
                  type="button"
                  className="primary-action"
                  data-testid="continue-run"
                  onClick={() => void cont({ generations: moreGenerations })}
                >
                  Continue
                </button>
                <button type="button" className="ghost" onClick={onOpenDashboard}>
                  Open dashboard
                </button>
              </div>
              <p className="hint">
                The locked test boundary is frozen for this session; continuing reads the
                out-of-sample set again, which is counted and shown above.
              </p>
            </div>
          )}
        </section>
      )}
    </div>
  );
}
