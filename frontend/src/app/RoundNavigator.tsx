import type { SessionRoundSummary } from "../api/types";

function roundEvidenceLabel(round: SessionRoundSummary): string {
  if (round.validity === "invalid_legacy_semantics") return "Invalid legacy evidence";
  if (round.finalization_available) return "Holdout finalized";
  if (round.evidence_status === "post_holdout_adaptive") {
    return "Post-holdout adaptive · exploratory";
  }
  if (
    round.evidence_status === "repeated_same_holdout" ||
    round.evidence_status === "exploratory_repeat"
  ) {
    return "Exploratory repeated holdout";
  }
  if (
    round.evidence_status === "locked_first_read" ||
    round.evidence_status === "locked"
  ) {
    return "Initial locked holdout";
  }
  return "Validation only · holdout unopened";
}

export function RoundNavigator({
  rounds,
  selectedRound,
  pending = false,
  onSelect,
  onContinue,
}: {
  rounds: SessionRoundSummary[];
  selectedRound: number | null;
  pending?: boolean;
  onSelect: (round: number) => void;
  onContinue?: () => void;
}) {
  const available = rounds.filter(
    (round) =>
      round.status === "done" ||
      round.status === "completed" ||
      round.report_available ||
      round.finalization_available,
  );
  if (available.length === 0) return null;
  const selectedPosition = Math.max(
    0,
    available.findIndex((round) => round.index === selectedRound),
  );
  const selected = available[selectedPosition];
  const previous = available[selectedPosition - 1];
  const next = available[selectedPosition + 1];

  return (
    <nav className="round-navigator" aria-label="Training round">
      <button
        type="button"
        className="round-navigator__arrow"
        aria-label="Previous training round"
        disabled={!previous}
        onClick={() => previous && onSelect(previous.index)}
      >
        ←
      </button>
      <div className="round-navigator__center">
        <div className="round-navigator__label">
          <strong>
            Round {selectedPosition + 1} of {available.length}
          </strong>
          <span>{roundEvidenceLabel(selected)}</span>
        </div>
        <div className="round-navigator__dots" aria-label="Available training rounds">
          {available.map((round, index) => (
            <button
              type="button"
              key={round.index}
              className={round.index === selected.index ? "is-current" : ""}
              aria-label={`Open training round ${index + 1}`}
              aria-current={round.index === selected.index ? "step" : undefined}
              onClick={() => onSelect(round.index)}
            />
          ))}
          {pending && (
            <span
              className="round-navigator__pending"
              aria-label={`Training round ${available.length + 1} is in progress`}
            />
          )}
        </div>
      </div>
      <button
        type="button"
        className="round-navigator__arrow"
        aria-label="Next training round"
        disabled={!next}
        onClick={() => next && onSelect(next.index)}
      >
        →
      </button>
      {onContinue && (
        <button type="button" className="ghost round-navigator__continue" onClick={onContinue}>
          Continue training
        </button>
      )}
    </nav>
  );
}
