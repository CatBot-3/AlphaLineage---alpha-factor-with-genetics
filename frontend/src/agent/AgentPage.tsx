// P11-T7 - the Agent page: one chat thread bound to one training session.
//
// This replaced two separate panels ("ask a model" and "let a model work on this") and a
// deterministic aspect-chip readout. The user no longer classifies their own question; they type,
// and the model picks its tools. The chips are gone because the labels behind them came from a
// hand-written dictionary that never grew — labels now arrive in the model's own answer.
//
// The thread renders like a chat client: your message, the model's reply, and a collapsible
// record of the work it did in between. That record is not decoration — a proposal is only worth
// trusting if the number that justified it can be traced back to the call that produced it.

import { useCallback, useEffect, useRef, useState } from "react";
import {
  applyAgentConfig,
  getAgentJob,
  getAgentTools,
  getConversation,
  clearConversation,
  promoteAgentProposal,
  sendAgentMessage,
} from "../api/client";
import type { AgentToolCatalog, Conversation } from "../api/types";
import { Composer } from "./Composer";
import { ConversationView } from "./ConversationView";

const POLL_INTERVAL_MS = 1200;

export function AgentPage({
  sessionId,
  sessionName,
  roundIndex,
  onOpenSettings,
  onRoundsChanged,
}: {
  sessionId?: string;
  sessionName?: string;
  roundIndex?: number;
  onOpenSettings?: () => void;
  onRoundsChanged?: () => void;
}) {
  const [catalog, setCatalog] = useState<AgentToolCatalog | null>(null);
  const [conversation, setConversation] = useState<Conversation | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [pending, setPending] = useState<string | null>(null);
  const pollRef = useRef<number | null>(null);

  const refresh = useCallback(async () => {
    if (!sessionId) return;
    try {
      setConversation(await getConversation(sessionId));
    } catch {
      setError("The conversation could not be loaded.");
    }
  }, [sessionId]);

  useEffect(() => {
    let cancelled = false;
    void (async () => {
      try {
        const loaded = await getAgentTools();
        if (!cancelled) setCatalog(loaded);
      } catch {
        if (!cancelled) setError("The agent tool catalog could not be loaded.");
      }
      await refresh();
    })();
    return () => {
      cancelled = true;
      if (pollRef.current !== null) window.clearTimeout(pollRef.current);
    };
  }, [refresh]);

  const poll = useCallback(
    (jobId: string) => {
      pollRef.current = window.setTimeout(async () => {
        try {
          const job = await getAgentJob(jobId);
          if (job.status === "done") {
            setBusy(false);
            setPending(null);
            await refresh();
            return;
          }
          if (job.status === "failed" || job.status === "stopped") {
            setBusy(false);
            setPending(null);
            setError(job.error ?? "The agent did not finish.");
            await refresh();
            return;
          }
          poll(jobId);
        } catch (err) {
          setBusy(false);
          setPending(null);
          setError(err instanceof Error ? err.message : "Lost contact with the backend.");
        }
      }, POLL_INTERVAL_MS);
    },
    [refresh],
  );

  const send = useCallback(
    async (text: string, budget: { max_tool_calls: number; max_evaluations: number }) => {
      if (!sessionId) return;
      setError(null);
      setBusy(true);
      setPending(text);
      try {
        const { job_id } = await sendAgentMessage(sessionId, {
          message: text,
          round_index: roundIndex,
          budget: { ...budget, max_seconds: catalog?.defaults.max_seconds ?? 300 },
        });
        poll(job_id);
      } catch (err) {
        setBusy(false);
        setPending(null);
        setError(err instanceof Error ? err.message : "The message could not be sent.");
      }
    },
    [sessionId, roundIndex, catalog, poll],
  );

  const promote = useCallback(
    async (proposalId: string) => {
      if (!sessionId) return;
      setError(null);
      setBusy(true);
      try {
        const { job_id } = await promoteAgentProposal(sessionId, proposalId);
        poll(job_id);
        onRoundsChanged?.();
      } catch (err) {
        setBusy(false);
        setError(err instanceof Error ? err.message : "That proposal could not be promoted.");
      }
    },
    [sessionId, poll, onRoundsChanged],
  );

  const applyConfig = useCallback(
    async (proposalId: string) => {
      if (!sessionId) return;
      setError(null);
      try {
        await applyAgentConfig(sessionId, proposalId);
        await refresh();
      } catch (err) {
        setError(err instanceof Error ? err.message : "That patch could not be applied.");
      }
    },
    [sessionId, refresh],
  );

  const clear = useCallback(async () => {
    if (!sessionId) return;
    if (!window.confirm("Clear this conversation? The training session is unaffected.")) return;
    await clearConversation(sessionId);
    await refresh();
  }, [sessionId, refresh]);

  if (!sessionId) {
    return (
      <div className="agent-page" data-testid="agent-page">
        <p className="hint">
          The agent works on one training session at a time. Start or open a session, then come
          back — it needs the session's frozen split boundaries and its search history to be
          useful, and it cannot read either without one.
        </p>
      </div>
    );
  }

  const unavailable = catalog?.unavailable_reason;

  return (
    <div className="agent-page" data-testid="agent-page">
      {error && (
        <p className="surface-message" role="alert">
          {error}
        </p>
      )}
      {unavailable && (
        <p className="surface-message" role="status">
          {unavailable}
        </p>
      )}

      <ConversationView
        conversation={conversation}
        sessionName={sessionName ?? conversation?.session_name}
        pending={pending}
        busy={busy}
        onPromote={promote}
        onApplyConfig={applyConfig}
      />

      <Composer
        catalog={catalog}
        busy={busy}
        disabled={Boolean(unavailable)}
        hasHistory={Boolean(conversation?.turns.length)}
        onSend={send}
        onClear={clear}
        onOpenSettings={onOpenSettings}
      />
    </div>
  );
}
