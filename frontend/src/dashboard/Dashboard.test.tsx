import { render, screen, within } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import type { HistoryPoint, Report, RunResult } from "../api/types";
import { Dashboard } from "./Dashboard";

const report: Report = {
  oos_ic: 0.12,
  deflated_sharpe: 0.03,
  pbo: 0.64,
  train_ic: 0.82,
  n_trials: 6210,
  significant: false,
};

const history: HistoryPoint[] = [
  { generation: 0, best_fitness: 0.4, mean_fitness: 0.2, best_ic: 0.41 },
  { generation: 1, best_fitness: 0.5, mean_fitness: 0.3, best_ic: 0.51 },
];

describe("dashboard (P6-T3)", () => {
  it("shows the deflated / OOS metrics as the default (guardrail)", () => {
    render(<Dashboard report={report} history={history} />);
    const primary = screen.getByTestId("primary-metric");
    expect(primary).toHaveTextContent(/out-of-sample/i);
    expect(within(primary).getByText("Deflated Sharpe")).toBeInTheDocument();
    expect(within(primary).getByText("0.030")).toBeInTheDocument(); // deflated sharpe value
    expect(within(primary).getByText("0.120")).toBeInTheDocument(); // OOS IC value
  });

  it("keeps train (in-sample) metrics secondary, not the default", () => {
    render(<Dashboard report={report} history={history} />);
    expect(screen.getByTestId("primary-metric")).not.toHaveTextContent(/train/i);
    expect(screen.getByTestId("secondary-metric")).toHaveTextContent(/train/i);
  });

  it("warns when the out-of-sample set has been read more than once (P3)", () => {
    const thrice = { test_reads: 3 } as unknown as RunResult;
    const { rerender } = render(<Dashboard report={report} history={history} extra={thrice} />);
    expect(screen.getByTestId("oos-warning")).toHaveTextContent(/3 times/);

    // a single read is the honest baseline - no warning
    const once = { test_reads: 1 } as unknown as RunResult;
    rerender(<Dashboard report={report} history={history} extra={once} />);
    expect(screen.queryByTestId("oos-warning")).not.toBeInTheDocument();
  });
});
