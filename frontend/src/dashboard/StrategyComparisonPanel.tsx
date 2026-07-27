import { useCallback, useEffect, useMemo, useState } from "react";
import {
  createFinalizationPlan,
  createStrategyComparison,
  getRun,
  getStrategyComparison,
  listFinalizationPlans,
  listStrategyComparisons,
} from "../api/client";
import type {
  FinalizationPlan,
  PortfolioStrategySpec,
  StrategyBacktestResult,
  StrategyComparisonDetail,
} from "../api/types";
import { LineChart } from "./LineChart";

export const PORTFOLIO_STRATEGIES: Array<
  PortfolioStrategySpec & { label: string; optional: boolean }
> = [
  {
    id: "quantile_ls_20",
    scheme: "quantile_ls",
    quantile: 0.2,
    label: "Quantile Long/Short · 20%",
    optional: false,
  },
  {
    id: "rank_proportional",
    scheme: "rank_proportional",
    quantile: null,
    label: "Rank Proportional",
    optional: false,
  },
  {
    id: "quantile_ls_10",
    scheme: "quantile_ls",
    quantile: 0.1,
    label: "Quantile Long/Short · 10%",
    optional: true,
  },
  {
    id: "quantile_ls_30",
    scheme: "quantile_ls",
    quantile: 0.3,
    label: "Quantile Long/Short · 30%",
    optional: true,
  },
];

const COLORS = ["#2563eb", "#0f5b3d", "#9f1d20", "#7656a8"];

export function portfolioStrategyLabel(id: string): string {
  return (
    PORTFOLIO_STRATEGIES.find((strategy) => strategy.id === id)?.label ??
    id.replace(/_/g, " ")
  );
}

function decimal(value: number | null | undefined, digits = 3): string {
  return typeof value === "number" && Number.isFinite(value) ? value.toFixed(digits) : "—";
}

function percent(value: number | null | undefined, digits = 1): string {
  return typeof value === "number" && Number.isFinite(value)
    ? `${(value * 100).toFixed(digits)}%`
    : "—";
}

function isViable(result: StrategyBacktestResult): boolean {
  const backtest = result.validation_backtest ?? result.oos_backtest;
  if (!backtest || result.eligible === false) return false;
  const health = backtest.portfolio_health;
  if (health) return health.valid && health.active_observations > 0;
  return backtest.metrics.usable;
}

function completed(status: string): boolean {
  return status === "done" || status === "completed";
}

export function StrategyComparisonPanel({
  sessionId,
  roundIndex,
  onFinalize,
  finalizing = false,
  postHoldout = false,
}: {
  sessionId: string;
  roundIndex: number;
  onFinalize: (strategyPlanId: string) => void | Promise<void>;
  finalizing?: boolean;
  postHoldout?: boolean;
}) {
  const [selectedIds, setSelectedIds] = useState<string[]>([
    "quantile_ls_20",
    "rank_proportional",
  ]);
  const [comparison, setComparison] = useState<StrategyComparisonDetail | null>(null);
  const [plan, setPlan] = useState<FinalizationPlan | null>(null);
  const [primaryId, setPrimaryId] = useState<string>("");
  const [jobId, setJobId] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const loadComparison = useCallback(
    async (comparisonId: string) => {
      const detail = await getStrategyComparison(sessionId, roundIndex, comparisonId);
      setComparison(detail);
      const plans = await listFinalizationPlans(sessionId, roundIndex).catch(() => []);
      const matching = plans.filter((item) => item.comparison_id === comparisonId);
      const latest = matching[matching.length - 1] ?? null;
      setPlan(latest);
      if (latest) {
        setPrimaryId(latest.primary_strategy_id);
      } else {
        const firstViable = detail.strategy_results.find(isViable);
        setPrimaryId(firstViable?.strategy_id ?? "");
      }
    },
    [roundIndex, sessionId],
  );

  useEffect(() => {
    let active = true;
    setComparison(null);
    setPlan(null);
    setJobId(null);
    setError(null);
    void listStrategyComparisons(sessionId, roundIndex)
      .then(async (items) => {
        if (!active) return;
        const eligibleItems = postHoldout
          ? items.filter((item) => item.evidence_status === "post_holdout_adaptive")
          : items.filter((item) => item.evidence_status !== "post_holdout_adaptive");
        if (eligibleItems.length === 0) return;
        const latest = eligibleItems[eligibleItems.length - 1];
        if (completed(latest.status) || latest.result_available) {
          await loadComparison(latest.comparison_id);
        } else if (latest.job_id && (latest.status === "queued" || latest.status === "running")) {
          setJobId(latest.job_id);
          setBusy(true);
        }
      })
      .catch((reason) => {
        if (active) setError(reason instanceof Error ? reason.message : String(reason));
      });
    return () => {
      active = false;
    };
  }, [loadComparison, postHoldout, roundIndex, sessionId]);

  useEffect(() => {
    if (!jobId) return;
    let active = true;
    let timeout: number | undefined;
    const poll = async () => {
      try {
        const job = await getRun(jobId);
        if (!active) return;
        if (job.status === "done" || job.status === "completed") {
          const summaries = await listStrategyComparisons(sessionId, roundIndex);
          const matching = [...summaries]
            .reverse()
            .find((item) => item.job_id === jobId || completed(item.status));
          if (!matching) throw new Error("The completed strategy comparison could not be found.");
          await loadComparison(matching.comparison_id);
          if (active) {
            setBusy(false);
            setJobId(null);
          }
          return;
        }
        if (job.status === "failed" || job.status === "stopped") {
          throw new Error(job.error ?? `Strategy comparison ${job.status}.`);
        }
        timeout = window.setTimeout(() => void poll(), 1_000);
      } catch (reason) {
        if (active) {
          setBusy(false);
          setJobId(null);
          setError(reason instanceof Error ? reason.message : String(reason));
        }
      }
    };
    void poll();
    return () => {
      active = false;
      if (timeout !== undefined) window.clearTimeout(timeout);
    };
  }, [jobId, loadComparison, roundIndex, sessionId]);

  const selectedStrategies = useMemo(
    () =>
      PORTFOLIO_STRATEGIES.filter((strategy) => selectedIds.includes(strategy.id)).map(
        ({ label: _label, optional: _optional, ...strategy }) => strategy,
      ),
    [selectedIds],
  );

  const viableResults = comparison?.strategy_results.filter(isViable) ?? [];
  const curveSeries =
    comparison?.strategy_results
      .filter(isViable)
      .map((item, index) => ({
        label: portfolioStrategyLabel(item.strategy_id),
        color: COLORS[index % COLORS.length],
        points: (item.validation_backtest ?? item.oos_backtest)?.normalized_equity.map((point) => ({
          x: point.date,
          value:
            typeof point.value === "number" && Number.isFinite(point.value)
              ? point.value - 1
              : null,
        })) ?? [],
      })) ?? [];

  async function compare() {
    if (
      postHoldout &&
      !window.confirm(
        "You have already viewed this frozen holdout. Comparing a new strategy bundle now is " +
          "adaptive research, and any later holdout result will be an exploratory repeated read. Continue?",
      )
    ) {
      return;
    }
    setBusy(true);
    setError(null);
    setComparison(null);
    setPlan(null);
    try {
      const handle = await createStrategyComparison(
        sessionId,
        roundIndex,
        selectedStrategies,
        postHoldout,
      );
      setJobId(handle.job_id);
    } catch (reason) {
      setBusy(false);
      setError(reason instanceof Error ? reason.message : String(reason));
    }
  }

  async function pinPrimary() {
    if (!comparison || !primaryId) return;
    setBusy(true);
    setError(null);
    try {
      const created = await createFinalizationPlan(
        sessionId,
        roundIndex,
        comparison.comparison_id,
        primaryId,
      );
      setPlan(created);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : String(reason));
    } finally {
      setBusy(false);
    }
  }

  return (
    <section className="strategy-comparison" data-testid="strategy-comparison">
      <header className="dashboard-section__head">
        <div>
          <h3>Compare portfolio strategies on validation</h3>
          <p>
            {postHoldout
              ? "The formula stays frozen, but the holdout has already been seen. A new bundle is adaptive research and any later finalization is exploratory."
              : "The formula stays frozen. Compare market-neutral implementations, then explicitly pin one primary strategy before opening the locked holdout."}
          </p>
        </div>
        <span>Maximum 4</span>
      </header>

      {!comparison && (
        <>
          <fieldset className="strategy-picker" disabled={busy}>
            <legend>Strategies to compare</legend>
            {PORTFOLIO_STRATEGIES.map((strategy) => (
              <label key={strategy.id}>
                <input
                  type="checkbox"
                  checked={selectedIds.includes(strategy.id)}
                  disabled={!strategy.optional}
                  onChange={(event) => {
                    setSelectedIds((current) =>
                      event.target.checked
                        ? [...current, strategy.id]
                        : current.filter((id) => id !== strategy.id),
                    );
                  }}
                />
                <span>{strategy.label}</span>
                {!strategy.optional && <small>default</small>}
              </label>
            ))}
          </fieldset>
          <button
            type="button"
            className="primary-action"
            onClick={() => void compare()}
            disabled={busy || selectedStrategies.length === 0 || selectedStrategies.length > 4}
          >
            {busy
              ? "Comparing validation strategies…"
              : postHoldout
                ? "Compare another strategy bundle"
                : "Compare strategies"}
          </button>
        </>
      )}

      {error && (
        <p className="error" role="alert">
          {error}
        </p>
      )}

      {comparison && (
        <>
          {curveSeries.length > 0 ? (
            <LineChart
              title="Validation cumulative return"
              description="All curves use the same frozen formula and validation dates. Ordering is fixed and does not imply a winner."
              baseline={0}
              domainFloor={-1}
              baselineAwarePadding
              formatValue={(value) => `${(value * 100).toFixed(1)}%`}
              series={curveSeries}
            />
          ) : (
            <p className="strategy-no-exposure" role="status">
              No compared strategy produced tradable two-sided exposure. This formula cannot be
              finalized as usable portfolio evidence.
            </p>
          )}

          <div className="strategy-result-grid">
            {comparison.strategy_results.map((item) => {
              const backtest = item.validation_backtest ?? item.oos_backtest;
              const health = backtest?.portfolio_health;
              const metrics = backtest?.metrics;
              const viable = isViable(item);
              return (
                <label
                  key={item.strategy_id}
                  className={`strategy-result${viable ? "" : " is-invalid"}`}
                >
                  <span className="strategy-result__title">
                    <input
                      type="radio"
                      name={`primary-strategy-${comparison.comparison_id}`}
                      value={item.strategy_id}
                      checked={primaryId === item.strategy_id}
                      disabled={!viable || Boolean(plan)}
                      onChange={() => setPrimaryId(item.strategy_id)}
                    />
                    <strong>{portfolioStrategyLabel(item.strategy_id)}</strong>
                  </span>
                  <span>Net Sharpe {decimal(metrics?.net_sharpe)}</span>
                  <span>Drawdown {percent(metrics?.max_drawdown)}</span>
                  <span>Turnover {decimal(metrics?.turnover)}</span>
                  <span>Average positions {decimal(metrics?.avg_positions, 1)}</span>
                  <span>Exposure coverage {percent(health?.exposure_coverage)}</span>
                  {!viable && (
                    <small>
                      {health?.reason === "no_exposure"
                        ? "No tradable cross-sectional variation"
                        : health?.reason ?? "Portfolio evidence is not usable"}
                    </small>
                  )}
                </label>
              );
            })}
          </div>

          <div className="strategy-actions">
            {postHoldout && (
              <button
                type="button"
                className="ghost"
                disabled={busy || finalizing}
                onClick={() => {
                  setComparison(null);
                  setPlan(null);
                  setPrimaryId("");
                  setError(null);
                }}
              >
                Configure another bundle
              </button>
            )}
            {plan ? (
              <p>
                Primary strategy pinned:{" "}
                <strong>{portfolioStrategyLabel(plan.primary_strategy_id)}</strong>
              </p>
            ) : (
              <button
                type="button"
                className="ghost"
                disabled={busy || !primaryId || viableResults.length === 0}
                onClick={() => void pinPrimary()}
              >
                {busy ? "Pinning…" : "Pin primary strategy"}
              </button>
            )}
            <button
              type="button"
              className="primary-action"
              disabled={!plan || finalizing}
              onClick={() => plan && void onFinalize(plan.strategy_plan_id)}
              data-testid="finalize-round"
            >
              {finalizing ? "Finalizing…" : "Finalize pinned bundle"}
            </button>
          </div>
        </>
      )}
    </section>
  );
}
