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

function snapshot(extra: Partial<ProgressSnapshot> = {}): ProgressSnapshot {
  return { ...progress("training"), report_done: 0, report_total: 0, ...extra };
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

  it("explains the worker count it measured rather than the percentage it was given", () => {
    render(
      <ProgressView
        progress={snapshot({
          resources: {
            profile: "auto",
            percent: 50,
            workers: 15,
            accelerated: true,
          } as never,
          tuning: {
            state: "settled",
            workers: 8,
            max_workers: 15,
            ladder: [1, 2, 4, 8, 15],
            samples: 15,
            speedup: 3.4,
            cost_by_workers: { "1": 1.0, "8": 0.29 },
          },
        })}
        phase="running"
        onStop={() => {}}
      />,
    );
    expect(screen.getByTestId("progress-resources")).toHaveTextContent("8 of 15 workers");
    expect(screen.getByTestId("progress-tuning")).toHaveTextContent(
      "measured 3.4× faster than one worker; more than 8 stopped helping",
    );
  });

  it("says it is still timing the machine while calibrating", () => {
    render(
      <ProgressView
        progress={snapshot({
          tuning: {
            state: "calibrating",
            workers: 4,
            max_workers: 4,
            ladder: [1, 2, 4],
            samples: 2,
            speedup: null,
            cost_by_workers: {},
          },
        })}
        phase="running"
        onStop={() => {}}
      />,
    );
    expect(screen.getByTestId("progress-tuning")).toHaveTextContent(
      "timing worker counts 1, 2, 4 on this machine",
    );
  });
});
