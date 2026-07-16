import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import type { ProgressSnapshot } from "../api/types";
import { ProgressView } from "./ProgressView";

function progress(phase: string): ProgressSnapshot {
  return {
    phase,
    generation: 12,
    target_generations: 12,
    history: [],
    best: null,
    report_done: 2,
    report_total: 4,
  };
}

describe("ProgressView stop policy", () => {
  it.each(["validating", "reporting"])("allows stopping while %s", (activePhase) => {
    const onStop = vi.fn();
    render(<ProgressView progress={progress(activePhase)} phase="running" onStop={onStop} />);

    fireEvent.click(screen.getByTestId("stop-run"));
    expect(onStop).toHaveBeenCalledOnce();
  });

  it("shows report progress instead of a completed generation bar while validating", () => {
    render(
      <ProgressView
        phase="running"
        onStop={vi.fn()}
        progress={{
          phase: "validating",
          generation: 12,
          target_generations: 12,
          history: [],
          best: null,
          resources: null,
          candidate_done: 0,
          candidate_total: 0,
          report_done: 3,
          report_total: 10,
          factors_per_second: null,
          termination_reason: null,
        }}
      />,
    );

    expect(screen.getByTestId("progress-generation")).toHaveTextContent("validation 3 / 10");
    expect(screen.getByRole("progressbar")).toHaveAttribute("aria-valuenow", "30");
  });

  it("disables stopping once locked-test finalization starts", () => {
    render(<ProgressView progress={progress("finalizing")} phase="running" onStop={vi.fn()} />);

    expect(screen.queryByTestId("stop-run")).not.toBeInTheDocument();
    expect(screen.getByText(/stopping is disabled/i)).toBeInTheDocument();
  });

  it.each(["done", "stopped", "failed"])(
    "does not offer Stop in the terminal %s phase",
    (terminalPhase) => {
      render(
        <ProgressView progress={progress(terminalPhase)} phase="running" onStop={vi.fn()} />,
      );

      expect(screen.queryByTestId("stop-run")).not.toBeInTheDocument();
    },
  );
});
