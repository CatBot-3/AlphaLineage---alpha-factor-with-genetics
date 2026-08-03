// P11-T7 - the message box, the budget for one turn, and the capability disclosure.
//
// The budget sits next to the box rather than in a settings panel because it is a per-message
// decision: "answer a question" needs no evaluations at all, "try three variants" needs three.

import { useState } from "react";
import type { AgentToolCatalog } from "../api/types";

export function Composer({
  catalog,
  busy,
  disabled,
  hasHistory,
  onSend,
  onClear,
  onOpenSettings,
}: {
  catalog: AgentToolCatalog | null;
  busy?: boolean;
  disabled?: boolean;
  hasHistory?: boolean;
  onSend: (text: string, budget: { max_tool_calls: number; max_evaluations: number }) => void;
  onClear: () => void;
  onOpenSettings?: () => void;
}) {
  const [text, setText] = useState("");
  const [maxToolCalls, setMaxToolCalls] = useState(catalog?.defaults.max_tool_calls ?? 12);
  const [maxEvaluations, setMaxEvaluations] = useState(catalog?.defaults.max_evaluations ?? 8);

  function submit() {
    const trimmed = text.trim();
    if (!trimmed || busy || disabled) return;
    onSend(trimmed, { max_tool_calls: maxToolCalls, max_evaluations: maxEvaluations });
    setText("");
  }

  return (
    <section className="agent-composer" aria-label="Message the agent">
      <textarea
        rows={3}
        value={text}
        disabled={disabled}
        placeholder="Ask about this factor, or tell the agent what to try…"
        onChange={(event) => setText(event.target.value)}
        onKeyDown={(event) => {
          // Enter sends, Shift+Enter is a newline — the convention people already have.
          if (event.key === "Enter" && !event.shiftKey) {
            event.preventDefault();
            submit();
          }
        }}
        data-testid="agent-input"
      />

      <div className="agent-composer__row">
        <label>
          Tool calls
          <input
            type="number"
            min={1}
            max={60}
            value={maxToolCalls}
            onChange={(event) => setMaxToolCalls(Number(event.target.value))}
            data-testid="agent-max-tool-calls"
          />
        </label>
        <label>
          Evaluations
          <input
            type="number"
            min={1}
            max={40}
            value={maxEvaluations}
            onChange={(event) => setMaxEvaluations(Number(event.target.value))}
            data-testid="agent-max-evaluations"
          />
        </label>
        <button
          type="button"
          className="primary-action"
          onClick={submit}
          disabled={busy || disabled || !text.trim()}
          data-testid="agent-send"
        >
          {busy ? "Working…" : "Send"}
        </button>
        {hasHistory && (
          <button type="button" className="link-action" onClick={onClear}>
            Clear thread
          </button>
        )}
        {onOpenSettings && (
          <button type="button" className="link-action" onClick={onOpenSettings}>
            Model settings
          </button>
        )}
      </div>

      {catalog && (
        <details className="agent-capability">
          <summary>What it can do ({catalog.tools.length} tools)</summary>
          <ul>
            {catalog.tools.map((tool) => (
              <li key={tool.name}>
                <code>{tool.name}</code>
                <span className={`agent-safety agent-safety--${tool.safety}`}>{tool.safety}</span>
                <p className="hint">{tool.description}</p>
              </li>
            ))}
          </ul>
          <p className="hint">
            Nothing outside this list is reachable. It reads training data only — the validation
            window and the locked holdout are absent from the data it is given, not merely
            forbidden. Settings it cannot change:{" "}
            {Object.keys(catalog.protected_config_keys).join(", ")}.
          </p>
        </details>
      )}
    </section>
  );
}
