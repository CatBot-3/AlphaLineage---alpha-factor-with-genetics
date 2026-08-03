// P11-T7 - one staged proposal, and the buttons that are the entire approval gate.
//
// Promoting says what promotion actually does, including the two parts a user would not guess:
// the session's real validation pass runs (the agent only ever saw an inner holdout inside the
// training window), and the agent's evaluations are added to the session's trial count, which
// makes every deflated statistic afterwards slightly more conservative. That is the honest trade
// and it belongs on the button, not in a footnote.

import type { AgentProposal } from "../api/types";

function ConfigDiff({ diff }: { diff: { key: string; from: unknown; to: unknown }[] }) {
  return (
    <table className="agent-diff">
      <thead>
        <tr>
          <th scope="col">Setting</th>
          <th scope="col">Now</th>
          <th scope="col">Proposed</th>
        </tr>
      </thead>
      <tbody>
        {diff.map((row) => (
          <tr key={row.key}>
            <td>
              <code>{row.key}</code>
            </td>
            <td>{String(row.from)}</td>
            <td>
              <strong>{String(row.to)}</strong>
            </td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

export function ProposalCard({
  proposal,
  onPromote,
  onApplyConfig,
}: {
  proposal: AgentProposal;
  onPromote: () => void;
  onApplyConfig: () => void;
}) {
  const isFactor = proposal.kind === "factor";

  return (
    <article className="agent-proposal" data-testid={`proposal-${proposal.id}`}>
      <header>
        <h5>{isFactor ? proposal.payload.name || "Candidate factor" : "Configuration change"}</h5>
        <span className="agent-proposal__kind">{proposal.kind}</span>
      </header>

      {isFactor && proposal.payload.expression && (
        <pre>
          <code>{proposal.payload.expression}</code>
        </pre>
      )}
      {!isFactor && proposal.payload.diff && <ConfigDiff diff={proposal.payload.diff} />}

      <p className="agent-proposal__rationale">{proposal.rationale}</p>

      {proposal.applied ? (
        <div className="agent-proposal__applied" role="status">
          <strong>Applied.</strong>{" "}
          {isFactor
            ? `Validated and added as round ${proposal.round_index ?? "?"}.`
            : "The next segment will use the new configuration."}
        </div>
      ) : (
        <div className="agent-proposal__actions">
          <button
            type="button"
            className="primary-action"
            onClick={isFactor ? onPromote : onApplyConfig}
            data-testid={isFactor ? "promote" : "apply-config"}
          >
            {isFactor ? "Validate & add as a round" : "Apply to the session"}
          </button>
          <p className="hint">
            {isFactor
              ? "Runs the session's real validation pass — the agent only saw an inner holdout "
                + "inside the training window — then adds it as a new round you can finalize "
                + "like any other. This also adds the agent's evaluations to the session's trial "
                + "count, so later deflated statistics stay honest about the search behind it."
              : "Updates the stored configuration for the next segment. Nothing runs yet."}
          </p>
        </div>
      )}
    </article>
  );
}
