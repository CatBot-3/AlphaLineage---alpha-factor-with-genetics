import { fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

const listStrategyComparisons = vi.fn();
const getStrategyComparison = vi.fn();
const listFinalizationPlans = vi.fn();
const createFinalizationPlan = vi.fn();
const createStrategyComparison = vi.fn();
const getRun = vi.fn();

vi.mock("../api/client", () => ({
  listStrategyComparisons: (...args: unknown[]) => listStrategyComparisons(...args),
  getStrategyComparison: (...args: unknown[]) => getStrategyComparison(...args),
  listFinalizationPlans: (...args: unknown[]) => listFinalizationPlans(...args),
  createFinalizationPlan: (...args: unknown[]) => createFinalizationPlan(...args),
  createStrategyComparison: (...args: unknown[]) => createStrategyComparison(...args),
  getRun: (...args: unknown[]) => getRun(...args),
}));

import { StrategyComparisonPanel } from "./StrategyComparisonPanel";

const health = {
  eligible_dates: 100,
  calendar_observations: 100,
  active_observations: 90,
  two_sided_observations: 90,
  exposure_coverage: 0.9,
  flat_factor_dates: 10,
  valid: true,
  reason: null,
};

function backtest(valid = true) {
  return {
    start: "2024-01-01",
    end: "2024-06-01",
    observations: 100,
    metrics: {
      signed_ic: 0.03,
      mean_abs_ic: 0.05,
      ic_ir: 0.5,
      gross_sharpe: valid ? 1.1 : null,
      net_sharpe: valid ? 0.9 : null,
      max_drawdown: valid ? -0.12 : null,
      turnover: valid ? 0.2 : null,
      avg_gross: valid ? 1 : 0,
      avg_positions: valid ? 18 : 0,
      max_position: valid ? 0.08 : 0,
      usable: valid,
    },
    returns: [],
    normalized_equity: valid
      ? [
          { date: "2024-01-01", value: 1 },
          { date: "2024-06-01", value: 1.1 },
        ]
      : [],
    portfolio_health: valid
      ? health
      : {
          ...health,
          active_observations: 0,
          two_sided_observations: 0,
          exposure_coverage: 0,
          valid: false,
          reason: "no_exposure",
        },
  };
}

afterEach(() => vi.clearAllMocks());

describe("StrategyComparisonPanel", () => {
  it("keeps fixed ordering, excludes invalid portfolios, and requires an immutable pin", async () => {
    listStrategyComparisons.mockResolvedValue([
      {
        comparison_id: "cmp-1",
        round_index: 0,
        status: "completed",
        result_available: true,
        strategies: [],
      },
    ]);
    getStrategyComparison.mockResolvedValue({
      comparison_id: "cmp-1",
      round_index: 0,
      status: "completed",
      strategies: [],
      strategy_results: [
        {
          strategy_id: "quantile_ls_20",
          spec: { id: "quantile_ls_20", scheme: "quantile_ls", quantile: 0.2 },
          validation_backtest: backtest(true),
          eligible: true,
        },
        {
          strategy_id: "rank_proportional",
          spec: { id: "rank_proportional", scheme: "rank_proportional", quantile: null },
          validation_backtest: backtest(false),
          eligible: false,
        },
      ],
    });
    listFinalizationPlans.mockResolvedValue([]);
    createFinalizationPlan.mockResolvedValue({
      strategy_plan_id: "plan-1",
      session_id: "session-1",
      round_index: 0,
      comparison_id: "cmp-1",
      primary_strategy_id: "quantile_ls_20",
      strategies: [],
    });
    const finalize = vi.fn();

    render(
      <StrategyComparisonPanel
        sessionId="session-1"
        roundIndex={0}
        onFinalize={finalize}
      />,
    );

    const cards = await screen.findAllByText(/Quantile Long\/Short|Rank Proportional/);
    expect(cards[0]).toHaveTextContent("Quantile Long/Short");
    expect(screen.getByText("No tradable cross-sectional variation")).toBeInTheDocument();
    expect(
      within(screen.getByText("Rank Proportional").closest("label")!).getByRole("radio"),
    ).toBeDisabled();
    expect(screen.getByTestId("finalize-round")).toBeDisabled();

    fireEvent.click(screen.getByRole("button", { name: "Pin primary strategy" }));
    await waitFor(() =>
      expect(createFinalizationPlan).toHaveBeenCalledWith(
        "session-1",
        0,
        "cmp-1",
        "quantile_ls_20",
      ),
    );
    fireEvent.click(screen.getByTestId("finalize-round"));
    expect(finalize).toHaveBeenCalledWith("plan-1");
  });

  it("starts with the two fixed default strategies and offers optional quantiles", async () => {
    listStrategyComparisons.mockResolvedValue([]);
    createStrategyComparison.mockResolvedValue({
      session_id: "session-1",
      round_index: 0,
      comparison_id: "cmp-2",
      job_id: "job-2",
      status: "queued",
    });

    render(
      <StrategyComparisonPanel
        sessionId="session-1"
        roundIndex={0}
        onFinalize={() => undefined}
      />,
    );

    await screen.findByText("Strategies to compare");
    const q10 = screen.getByLabelText(/10%/);
    fireEvent.click(q10);
    fireEvent.click(screen.getByRole("button", { name: "Compare strategies" }));

    await waitFor(() =>
      expect(createStrategyComparison).toHaveBeenCalledWith("session-1", 0, [
        { id: "quantile_ls_20", scheme: "quantile_ls", quantile: 0.2 },
        { id: "rank_proportional", scheme: "rank_proportional", quantile: null },
        { id: "quantile_ls_10", scheme: "quantile_ls", quantile: 0.1 },
      ], false),
    );
  });

  it("requires explicit confirmation before changing strategies after a holdout read", async () => {
    listStrategyComparisons.mockResolvedValue([]);
    createStrategyComparison.mockResolvedValue({
      session_id: "session-1",
      round_index: 0,
      comparison_id: "cmp-repeat",
      job_id: "job-repeat",
      status: "queued",
    });
    const confirm = vi.spyOn(window, "confirm").mockReturnValue(true);

    render(
      <StrategyComparisonPanel
        sessionId="session-1"
        roundIndex={0}
        onFinalize={() => undefined}
        postHoldout
      />,
    );

    fireEvent.click(
      await screen.findByRole("button", { name: "Compare another strategy bundle" }),
    );
    expect(confirm).toHaveBeenCalledOnce();
    await waitFor(() =>
      expect(createStrategyComparison).toHaveBeenCalledWith(
        "session-1",
        0,
        [
          { id: "quantile_ls_20", scheme: "quantile_ls", quantile: 0.2 },
          { id: "rank_proportional", scheme: "rank_proportional", quantile: null },
        ],
        true,
      ),
    );
  });
});
