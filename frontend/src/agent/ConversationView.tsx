// P11-T7 - the thread. User turns, assistant turns, and the work between them.
//
// The working block collapses by default once a turn finishes and expands on click, which is the
// behaviour people already expect from a coding assistant: the answer is what you read, the tool
// calls are what you check when the answer surprises you.

import { useState } from "react";
import type { AgentToolCall, AgentTurn, Conversation } from "../api/types";
import { ProposalCard } from "./ProposalCard";

function fmt(value: unknown): string {
  const number = Number(value);
  return Number.isFinite(number) ? number.toFixed(4) : "n/a";
}

/** One line per call, so "it looked something up" and "it spent a trial" read differently. */
function headline(call: AgentToolCall): string {
  const result = (call.result ?? {}) as Record<string, unknown>;
  if (!call.ok) return String(result.error ?? "failed");
  if (call.tool === "evaluate_expression") {
    const holdout = (result.inner_holdout ?? {}) as Record<string, unknown>;
    const parts = [`inner-holdout IC ${fmt(holdout.oriented_ic)}`];
    if (result.generalization_gap != null) parts.push(`gap ${fmt(result.generalization_gap)}`);
    if (result.correlation_with_current_factor != null) {
      parts.push(`corr vs current ${fmt(result.correlation_with_current_factor)}`);
    }
    return parts.join(" · ");
  }
  if (call.tool === "validate_expression") {
    return `valid · ${result.nodes} nodes · lookback ~${result.effective_lookback_bars} bars`;
  }
  if (call.tool === "annotate_factor") return `${result.attached} label(s) attached`;
  if (call.tool === "search_workspace") return `${result.count} document(s) found`;
  if (result.staged) return "staged for your approval";
  return "";
}

function argSummary(call: AgentToolCall): string {
  const args = (call.arguments ?? {}) as Record<string, unknown>;
  if (typeof args.expression === "string") return args.expression;
  if (typeof args.query === "string") return args.query;
  if (args.patch && typeof args.patch === "object") {
    return Object.entries(args.patch as Record<string, unknown>)
      .map(([key, value]) => `${key}=${String(value)}`)
      .join(", ");
  }
  if (Array.isArray(args.labels)) return `${args.labels.length} label(s)`;
  return "";
}

function ToolCallRow({ call }: { call: AgentToolCall }) {
  const [open, setOpen] = useState(false);
  return (
    <li data-ok={call.ok} data-testid="tool-call">
      <button type="button" className="agent-call__head" onClick={() => setOpen(!open)}>
        <span className="agent-call__chevron" aria-hidden="true">
          {open ? "▾" : "▸"}
        </span>
        <code>{call.tool}</code>
        <span className={`agent-safety agent-safety--${call.safety}`}>{call.safety}</span>
        {argSummary(call) && <span className="agent-call__args">{argSummary(call)}</span>}
      </button>
      {headline(call) && <p className="hint agent-call__headline">{headline(call)}</p>}
      {open && (
        <pre className="agent-call__result">
          <code>{JSON.stringify(call.result, null, 2)}</code>
        </pre>
      )}
    </li>
  );
}

function AssistantTurn({
  turn,
  onPromote,
  onApplyConfig,
}: {
  turn: AgentTurn;
  onPromote: (id: string) => void;
  onApplyConfig: (id: string) => void;
}) {
  const [showWork, setShowWork] = useState(false);
  const evaluations = turn.calls.filter((c) => c.safety === "evaluate").length;

  return (
    <article className="agent-turn agent-turn--assistant" data-testid="assistant-turn">
      {turn.calls.length > 0 && (
        <div className="agent-work">
          <button
            type="button"
            className="agent-work__toggle"
            onClick={() => setShowWork(!showWork)}
            data-testid="toggle-work"
          >
            {showWork ? "▾" : "▸"} Worked for {turn.calls.length} step
            {turn.calls.length === 1 ? "" : "s"}
            {evaluations > 0 && ` · ${evaluations} evaluation${evaluations === 1 ? "" : "s"}`}
          </button>
          {showWork && (
            <ol className="agent-calls" data-testid="tool-calls">
              {turn.calls.map((call, index) => (
                <ToolCallRow key={index} call={call} />
              ))}
            </ol>
          )}
        </div>
      )}

      {turn.text && <div className="agent-turn__text">{turn.text}</div>}

      {turn.labels.length > 0 && (
        <ul className="agent-labels" data-testid="agent-labels">
          {turn.labels.map((label, index) => (
            <li key={index}>
              <span className="agent-label__name">{label.name}</span>
              {label.evidence && <code className="agent-label__evidence">{label.evidence}</code>}
              {label.reading && <p>{label.reading}</p>}
              {label.risk && <p className="hint">Risk: {label.risk}</p>}
            </li>
          ))}
        </ul>
      )}

      {turn.proposals.length > 0 && (
        <ul className="agent-proposals" data-testid="agent-proposals">
          {turn.proposals.map((proposal) => (
            <li key={proposal.id}>
              <ProposalCard
                proposal={proposal}
                onPromote={() => onPromote(proposal.id)}
                onApplyConfig={() => onApplyConfig(proposal.id)}
              />
            </li>
          ))}
        </ul>
      )}

      {turn.stop_reason === "budget_exhausted" && (
        <p className="hint">
          It ran out of budget for this turn and wrapped up. Raise the limits below to let it go
          further.
        </p>
      )}
      {turn.error && <p className="explanation-warning">{turn.error}</p>}
    </article>
  );
}

export function ConversationView({
  conversation,
  sessionName,
  pending,
  busy,
  onPromote,
  onApplyConfig,
}: {
  conversation: Conversation | null;
  sessionName?: string;
  pending?: string | null;
  busy?: boolean;
  onPromote: (id: string) => void;
  onApplyConfig: (id: string) => void;
}) {
  const turns = conversation?.turns ?? [];

  return (
    <section className="agent-thread" aria-label="Conversation" data-testid="agent-thread">
      {turns.length === 0 && !pending && (
        <div className="agent-empty">
          <p>
            Ask about {sessionName ? <strong>{sessionName}</strong> : "this session"} — what its
            factor measures, why the search behaved the way it did, or how to improve it. The
            agent decides which of its tools a question needs.
          </p>
          <ul className="hint">
            <li>“What does this factor actually measure?”</li>
            <li>“Why did the search stop improving?”</li>
            <li>“Try adding a volume term and tell me if it helps.”</li>
          </ul>
        </div>
      )}

      {turns.map((turn, index) =>
        turn.role === "user" ? (
          <article key={index} className="agent-turn agent-turn--user" data-testid="user-turn">
            {turn.text}
          </article>
        ) : (
          <AssistantTurn
            key={index}
            turn={turn}
            onPromote={onPromote}
            onApplyConfig={onApplyConfig}
          />
        ),
      )}

      {pending && (
        <>
          <article className="agent-turn agent-turn--user">{pending}</article>
          {busy && (
            <p className="agent-thinking" role="status">
              Working…
            </p>
          )}
        </>
      )}
    </section>
  );
}
