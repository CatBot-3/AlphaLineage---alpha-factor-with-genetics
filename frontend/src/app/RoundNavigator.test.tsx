import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { RoundNavigator } from "./RoundNavigator";

const rounds = [
  {
    index: 0,
    segment_index: 0,
    report_available: false,
    test_read_index: null,
    evidence_status: "validation_only" as const,
    status: "done",
    requested_generations: 12,
    gen_start: 0,
    gen_end: 12,
  },
  {
    index: 1,
    segment_index: 1,
    report_available: false,
    test_read_index: null,
    evidence_status: "post_holdout_adaptive" as const,
    status: "done",
    requested_generations: 5,
    gen_start: 12,
    gen_end: 17,
  },
];

describe("RoundNavigator", () => {
  it("switches completed rounds and exposes a pending continuation", () => {
    const select = vi.fn();
    render(
      <RoundNavigator
        rounds={rounds}
        selectedRound={1}
        pending
        onSelect={select}
      />,
    );
    expect(screen.getByText("Round 2 of 2")).toBeInTheDocument();
    expect(screen.getByText("Post-holdout adaptive · exploratory")).toBeInTheDocument();
    expect(screen.getByLabelText("Training round 3 is in progress")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Previous training round" }));
    expect(select).toHaveBeenCalledWith(0);
    expect(screen.getByRole("button", { name: "Next training round" })).toBeDisabled();
  });

  // An agent round is a real round: same numbering, same finalization path. It is marked so the
  // reader knows the formula came from a model rather than from the search, not so it ranks lower.
  it("marks an agent-authored round without moving it out of the sequence", () => {
    const withAgent = [
      ...rounds,
      {
        index: 2,
        segment_index: 1,
        origin: "agent" as const,
        agent_expression: "rank(ts_mean(volume, 20))",
        parent_round_index: 1,
        report_available: true,
        test_read_index: null,
        evidence_status: "validation_only" as const,
        status: "done",
        requested_generations: 0,
        gen_start: 17,
        gen_end: 17,
      },
    ];
    render(
      <RoundNavigator rounds={withAgent} selectedRound={2} onSelect={vi.fn()} />,
    );

    expect(screen.getByTestId("agent-round-badge")).toHaveTextContent("Agent");
    expect(screen.getByText("Round 3 of 3")).toBeInTheDocument();
    expect(screen.getByLabelText("Open agent round 3")).toHaveClass("is-agent");
    expect(screen.getByLabelText("Open training round 1")).not.toHaveClass("is-agent");
  });

  it("does not badge a GP round", () => {
    render(<RoundNavigator rounds={rounds} selectedRound={1} onSelect={vi.fn()} />);
    expect(screen.queryByTestId("agent-round-badge")).not.toBeInTheDocument();
  });
});
