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

export type SessionPhase = "idle" | "running" | "done" | "failed";

const POLL_MS = 1000;

export function useSession(onComplete?: (result: RunResult) => void) {
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [pollToken, setPollToken] = useState(0);
  const [state, setState] = useState<SessionState | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const onCompleteRef = useRef(onComplete);
  onCompleteRef.current = onComplete;
  const firedRef = useRef(false);

  const reset = useCallback(() => {
    firedRef.current = false;
    setSessionId(null);
    setState(null);
    setError(null);
    setPollToken((t) => t + 1);
  }, []);

  useEffect(() => {
    if (!sessionId) return;
    let active = true;
    let timer: ReturnType<typeof setTimeout> | undefined;

    const tick = async () => {
      try {
        const next = await getSession(sessionId);
        if (!active) return;
        setState(next);
        const status = next.job?.status;
        if (status === "done") {
          if (next.result && !firedRef.current) {
            firedRef.current = true;
            onCompleteRef.current?.(next.result);
          }
          return; // stop polling
        }
        if (status === "failed") {
          setError("the run failed; see the backend logs");
          return;
        }
      } catch (e) {
        if (!active) return;
        // A 404 means the session no longer exists (e.g. its data was cleared) - drop back to
        // the idle form rather than trapping the user on a dead error page.
        if (e instanceof ApiError && e.status === 404) {
          firedRef.current = false;
          setSessionId(null);
          setState(null);
          setError(null);
          setNotice("That session no longer exists - start a new one.");
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
    setError(null);
    setNotice(null);
    setState(null);
    firedRef.current = false;
    try {
      const handle = await createSession(req);
      setSessionId(handle.session_id);
      setPollToken((t) => t + 1);
    } catch (e) {
      setError(String(e));
    }
  }, []);

  const cont = useCallback(
    async (req: SessionContinueRequest) => {
      if (!sessionId) return;
      setError(null);
      firedRef.current = false;
      try {
        await continueSession(sessionId, req);
        setPollToken((t) => t + 1); // restart polling for the new segment
      } catch (e) {
        setError(String(e));
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
    setSessionId(id);
    setPollToken((t) => t + 1);
  }, []);

  const phase: SessionPhase = error
    ? "failed"
    : (state?.job?.status as SessionPhase | undefined) ?? (sessionId ? "running" : "idle");

  return { sessionId, state, error, notice, phase, start, cont, stop, attach, reset };
}
