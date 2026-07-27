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
});
