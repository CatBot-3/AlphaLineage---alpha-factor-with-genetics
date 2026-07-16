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
  it("shows the correctly labelled holdout and overfitting metrics first", () => {
    render(<Dashboard report={report} history={history} />);
    const primary = screen.getByTestId("primary-metric");
    const controls = screen.getByTestId("overfitting-controls");
    expect(within(primary).getByText("OOS mean |rank IC|")).toBeInTheDocument();
    expect(within(primary).getByText("0.120")).toBeInTheDocument();
    expect(within(controls).getByText("Deflated Sharpe probability")).toBeInTheDocument();
    expect(within(controls).getByText("3.0%")).toBeInTheDocument();
    expect(within(controls).getByText("64.0%")).toBeInTheDocument();
    expect(within(controls).getByText("6210")).toBeInTheDocument();
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

  it("renders the locked equity and separate convergence charts when evidence is available", () => {
    const result = {
      oos_backtest: {
        start: "2025-01-01",
        end: "2025-01-03",
        observations: 3,
        metrics: {
          signed_ic: 0.08,
          mean_abs_ic: 0.12,
          ic_ir: 0.5,
          gross_sharpe: 1.1,
          net_sharpe: 0.9,
          max_drawdown: -0.1,
          turnover: 0.2,
          avg_gross: 1,
          avg_positions: 20,
          max_position: 0.05,
          usable: true,
        },
        returns: [],
        normalized_equity: [
          { date: "2025-01-01", value: 1 },
          { date: "2025-01-02", value: 1.02 },
          { date: "2025-01-03", value: 1.01 },
        ],
      },
    } as unknown as RunResult;
    render(<Dashboard report={report} history={history} extra={result} />);
    expect(screen.getByText("Normalized net equity")).toBeInTheDocument();
    expect(screen.getByText("Population fitness")).toBeInTheDocument();
    expect(screen.getAllByText("Best |rank IC|").length).toBeGreaterThan(0);
    expect(screen.getAllByRole("img")).toHaveLength(3);
  });
});
