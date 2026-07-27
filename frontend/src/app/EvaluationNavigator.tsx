import type { SessionFinalizationSummary } from "../api/types";

function label(
  item: SessionFinalizationSummary,
  index: number,
  total: number,
): string {
  const evidence =
    item.evidence_status === "locked_first_read"
      ? "first locked read"
      : "exploratory repeat";
  const strategy = item.primary_strategy_id
    ? ` · ${item.primary_strategy_id.replace(/_/g, " ")}`
    : "";
  return `Evaluation ${index + 1} of ${total} · ${evidence}${strategy}`;
}

export function EvaluationNavigator({
  evaluations,
  selectedId,
  onSelect,
}: {
  evaluations: SessionFinalizationSummary[];
  selectedId: string | null;
  onSelect: (evaluationId: string) => void;
}) {
  if (evaluations.length === 0) return null;
  const selectedIndex = Math.max(
    0,
    evaluations.findIndex((item) => item.evaluation_id === selectedId),
  );
  const selected = evaluations[selectedIndex];

  return (
    <nav className="evaluation-navigator" aria-label="Holdout evaluations">
      <div>
        <strong>Holdout evidence</strong>
        <small>
          The first locked read remains selectable; later evaluations are exploratory.
        </small>
      </div>
      <button
        type="button"
        aria-label="Previous holdout evaluation"
        disabled={selectedIndex === 0}
        onClick={() => onSelect(evaluations[selectedIndex - 1].evaluation_id)}
      >
        ←
      </button>
      <label>
        <span className="sr-only">Holdout evaluation</span>
        <select
          value={selected.evaluation_id}
          onChange={(event) => onSelect(event.target.value)}
        >
          {evaluations.map((item, index) => (
            <option key={item.evaluation_id} value={item.evaluation_id}>
              {label(item, index, evaluations.length)}
            </option>
          ))}
        </select>
      </label>
      <button
        type="button"
        aria-label="Next holdout evaluation"
        disabled={selectedIndex === evaluations.length - 1}
        onClick={() => onSelect(evaluations[selectedIndex + 1].evaluation_id)}
      >
        →
      </button>
    </nav>
  );
}
