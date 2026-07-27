import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import type { SessionFinalizationSummary } from "../api/types";
import { EvaluationNavigator } from "./EvaluationNavigator";

const evaluations: SessionFinalizationSummary[] = [
  {
    evaluation_id: "first",
    round_index: 0,
    status: "done",
    evidence_status: "locked_first_read",
    same_holdout_read_index: 1,
    session_holdout_reads: 1,
    holdout_fingerprint: "holdout",
    report_available: true,
    primary_strategy_id: "quantile_ls_20",
  },
  {
    evaluation_id: "repeat",
    round_index: 1,
    status: "done",
    evidence_status: "repeated_same_holdout",
    same_holdout_read_index: 2,
    session_holdout_reads: 2,
    holdout_fingerprint: "holdout",
    report_available: true,
    primary_strategy_id: "rank_proportional",
  },
];

describe("EvaluationNavigator", () => {
  it("keeps the first locked artifact visible and navigable", () => {
    const select = vi.fn();
    render(
      <EvaluationNavigator
        evaluations={evaluations}
        selectedId="repeat"
        onSelect={select}
      />,
    );

    expect(screen.getByRole("option", { name: /first locked read/i })).toBeInTheDocument();
    expect(screen.getByRole("option", { name: /exploratory repeat/i })).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Previous holdout evaluation" }));
    expect(select).toHaveBeenCalledWith("first");
  });
});
