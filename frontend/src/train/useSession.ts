// Drives an iterative training session: create or continue a segment, poll its job to
// completion, expose live progress, and surface the final result. One source of truth for
// the Train tab and (via onComplete) the Dashboard / Factor / Genealogy views.

import { useCallback, useEffect, useRef, useState } from "react";
import {
  ApiError,
  continueSession,
  createSession,
  getSession,
  stopSession,
} from "../api/client";
import type {
  RunResult,
  SessionContinueRequest,
  SessionCreateRequest,
  SessionState,
} from "../api/types";

export type SessionPhase = "idle" | "running" | "done" | "stopped" | "failed" | "interrupted";
export type SessionConnection = "connecting" | "connected" | "reconnecting";

const POLL_MS = 1000;
const RETRY_MS = [1000, 2000, 4000, 8000, 15000];

function retryableConnection(error: unknown): boolean {
  const name = error && typeof error === "object" && "name" in error ? error.name : null;
  return error instanceof TypeError ||
    name === "ConnectionError" || name === "AbortError" || name === "TypeError" ||
    (error instanceof ApiError && error.status >= 500);
}

export function useSession(
  onComplete?: (result: RunResult) => void,
  initialSessionId: string | null = null,
) {
  const [sessionId, setSessionId] = useState<string | null>(initialSessionId);
  const [pollToken, setPollToken] = useState(0);
  const [state, setState] = useState<SessionState | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState<"start" | "continue" | null>(null);
  const submissionRef = useRef(false);
  const [connection, setConnection] = useState<SessionConnection>(
    initialSessionId ? "connecting" : "connected",
  );
  const onCompleteRef = useRef(onComplete);
  onCompleteRef.current = onComplete;
  const firedRef = useRef(false);

  const reset = useCallback(() => {
    firedRef.current = false;
    setSessionId(null);
    setState(null);
    setError(null);
    setConnection("connected");
    setPollToken((t) => t + 1);
  }, []);

  useEffect(() => {
    if (!sessionId) return;
    let active = true;
    let timer: ReturnType<typeof setTimeout> | undefined;
    let failures = 0;

    const tick = async () => {
      try {
        const next = await getSession(sessionId);
        if (!active) return;
        failures = 0;
        setConnection("connected");
        setState(next);
        const status = next.job?.status;
        const finalizationStatus = next.finalization_job?.status;
        if (status === "done") {
          // Completed searches now publish a validation round. A locked-holdout report is
          // intentionally absent until the user explicitly finalizes a selected round.
          if (next.result && !firedRef.current) {
            firedRef.current = true;
            onCompleteRef.current?.(next.result);
          }
        } else if (status === "stopped") {
          // A stop during research-only validation intentionally has no locked-test report.
          // Keep that session resumable without publishing an incomplete result to Dashboard.
          if (next.result && !firedRef.current) {
            firedRef.current = true;
            onCompleteRef.current?.(next.result);
          }
        } else if (status === "failed") {
          // The job carries why it failed and the finalization branch below already shows it.
          // Dropping it here sent users to a log they have no way to open from the app.
          setError(next.job?.error ?? "the run failed; see the backend logs");
          return;
        }
        setError(null);
        if (finalizationStatus === "failed") {
          setError(next.finalization_job?.error ?? "holdout finalization failed");
          return;
        }
        const searchActive = status === "queued" || status === "running";
        const finalizationActive =
          finalizationStatus === "queued" || finalizationStatus === "running";
        if (!searchActive && !finalizationActive) return;
      } catch (e) {
        if (!active) return;
        // A 404 means the session no longer exists (e.g. its data was cleared) - drop back to
        // the idle form rather than trapping the user on a dead error page.
        if (retryableConnection(e)) {
          setConnection("reconnecting");
          timer = setTimeout(tick, RETRY_MS[Math.min(failures++, RETRY_MS.length - 1)]);
          return;
        } else if (e instanceof ApiError && e.status === 404) {
          firedRef.current = false;
          setSessionId(null);
          setState(null);
          setError(null);
          setNotice("That session no longer exists - start a new one.");
          setConnection("connected");
        } else {
          setError(String(e));
        }
        return;
      }
      timer = setTimeout(tick, POLL_MS);
    };

    void tick();
    return () => {
      active = false;
      if (timer) clearTimeout(timer);
    };
  }, [sessionId, pollToken]);

  const start = useCallback(async (req: SessionCreateRequest) => {
    if (submissionRef.current) return;
    submissionRef.current = true;
    setSubmitting("start");
    setError(null);
    setNotice(null);
    setState(null);
    firedRef.current = false;
    try {
      const handle = await createSession(req);
      setConnection("connecting");
      setSessionId(handle.session_id);
      setPollToken((t) => t + 1);
    } catch (e) {
      setError(String(e));
    } finally {
      submissionRef.current = false;
      setSubmitting(null);
    }
  }, []);

  const cont = useCallback(
    async (req: SessionContinueRequest) => {
      if (!sessionId || submissionRef.current) return;
      submissionRef.current = true;
      setSubmitting("continue");
      setError(null);
      firedRef.current = false;
      try {
        await continueSession(sessionId, req);
        setConnection("connecting");
        setPollToken((t) => t + 1); // restart polling for the new segment
      } catch (e) {
        setError(String(e));
        // Reconcile a lost continuation response using GET only. The POST may already
        // have claimed a job; replaying it could submit work the user never intended.
        if ((retryableConnection(e) && !(e instanceof ApiError)) ||
            (e instanceof ApiError && e.status === 409)) {
          setConnection("reconnecting");
          setPollToken((t) => t + 1);
        }
      } finally {
        submissionRef.current = false;
        setSubmitting(null);
      }
    },
    [sessionId],
  );

  const stop = useCallback(async () => {
    if (!sessionId) return;
    try {
      await stopSession(sessionId);
    } catch (e) {
      setError(String(e));
    }
  }, [sessionId]);

  const attach = useCallback((id: string) => {
    firedRef.current = false;
    setError(null);
    setNotice(null);
    setState(null);
    setConnection("connecting");
    setSessionId(id);
    setPollToken((t) => t + 1);
  }, []);

  const refresh = useCallback(() => {
    setPollToken((token) => token + 1);
  }, []);

  const status = state?.job?.status;
  const phase: SessionPhase = status === "queued" ? "running" :
    (status as SessionPhase | undefined) ?? (sessionId ? "running" : "idle");
  const finalizationStatus = state?.finalization_job?.status ?? null;
  const finalizing = finalizationStatus === "queued" || finalizationStatus === "running";

  return {
    sessionId,
    state,
    error,
    notice,
    connection,
    submitting,
    phase,
    finalizationStatus,
    finalizing,
    start,
    cont,
    stop,
    attach,
    refresh,
    reset,
  };
}

export type SessionController = ReturnType<typeof useSession>;
