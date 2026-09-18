import type {
  HistoryPoint,
  Report,
  RunBacktestReport,
  RunResult,
  StrategyBacktestResult,
  ValidationFold,
} from "../api/types";
import { BenchmarkComparison } from "./BenchmarkComparison";
import { LineChart } from "./LineChart";
import { FactorOverlapPanel } from "./FactorOverlapPanel";
import { StrategyComparisonPanel } from "./StrategyComparisonPanel";
import { executionTimingLabel } from "../train/defaults";

function decimal(value: number | null | undefined, digits = 3): string {
  return typeof value === "number" && Number.isFinite(value) ? value.toFixed(digits) : "—";
}

function percent(value: number | null | undefined, digits = 1): string {
  return typeof value === "number" && Number.isFinite(value)
    ? `${(value * 100).toFixed(digits)}%`
    : "—";
}

function seconds(value: number | undefined): string {
  if (value === undefined || !Number.isFinite(value)) return "—";
  if (value < 60) return `${value.toFixed(1)}s`;
  return `${Math.floor(value / 60)}m ${Math.round(value % 60)}s`;
}

function Metric({
  label,
  value,
  note,
  emphasis = false,
}: {
  label: string;
  value: string;
  note?: string;
  emphasis?: boolean;
}) {
  return (
    <div className={`metric${emphasis ? " metric--emphasis" : ""}`}>
      <span className="metric-label">{label}</span>
      <span className="metric-value">{value}</span>
      {note && <small className="metric-note">{note}</small>}
    </div>
  );
}

function RunContextLine({ result }: { result?: RunResult }) {
  const context = result?.context;
  if (!context) return null;
  const boundaries = context.boundaries;
  const items = [
    context.universe && `Universe ${context.universe}`,
    context.universe_revision && `Revision ${context.universe_revision.slice(0, 12)}`,
    context.as_of && `As of ${context.as_of}`,
    boundaries?.test_start &&
      boundaries?.test_end &&
      `Frozen holdout ${boundaries.test_start}–${boundaries.test_end}`,
    context.horizon != null && `Horizon ${context.horizon}d`,
    // Contexts written before execution timing existed used the same-close assumption.
    context.boundaries && `Trade at: ${executionTimingLabel(context.execution)}`,
    context.weighting_scheme && context.weighting_scheme.replace(/_/g, " "),
    context.commission_bps != null &&
      context.slippage_bps != null &&
      `${context.commission_bps + context.slippage_bps} bps costs`,
  ].filter(Boolean) as string[];
  return (
    <div className="run-context" aria-label="Run context">
      {items.map((item) => (
        <span key={item}>{item}</span>
      ))}
    </div>
  );
}

function FoldMetric({ fold }: { fold: ValidationFold }) {
  const dates =
    fold.start && fold.end ? `${fold.start}–${fold.end}` : `${fold.valid_dates} valid dates`;
  return (
    <div className={`validation-fold${fold.positive ? " is-positive" : " is-negative"}`}>
      <div className="validation-fold__head">
        <strong>Fold {fold.index + 1}</strong>
        <span>{fold.positive ? "positive" : "not positive"}</span>
      </div>
      <span className="metric-value">{decimal(fold.oriented_ic)}</span>
      <small>{dates}</small>
      <small>
        IC-IR {decimal(fold.ic_ir)} · avg names {decimal(fold.avg_active_names, 1)}
      </small>
      <small>
        IC coverage {percent(fold.ic_coverage)} · two-sided {percent(fold.two_sided_coverage)}
      </small>
      {fold.coverage_failures && fold.coverage_failures.length > 0 && (
        <small className="error">
          {fold.coverage_failures.map((failure) => failure.replace(/_/g, " ")).join("; ")}
        </small>
      )}
    </div>
  );
}

function strategyName(result: StrategyBacktestResult): string {
  if (result.spec.scheme === "rank_proportional") return "Rank Proportional";
  return `Quantile Long/Short · ${Math.round((result.spec.quantile ?? 0.2) * 100)}%`;
}

function usableBacktest(backtest: RunBacktestReport | null | undefined): boolean {
  if (!backtest) return false;
  const health = backtest.portfolio_health;
  if (health) return health.valid && health.active_observations > 0;
  return backtest.metrics.usable !== false;
}

function FinalizedStrategyComparison({
  results,
  primaryStrategyId,
}: {
  results: StrategyBacktestResult[];
  primaryStrategyId?: string | null;
}) {
  const colors = ["#2563eb", "#0f5b3d", "#9f1d20", "#7656a8"];
  const usable = results.filter((item) => usableBacktest(item.oos_backtest));
  const series = usable.map((item, index) => ({
    label: `${strategyName(item)}${item.strategy_id === primaryStrategyId ? " · primary" : ""}`,
    color: colors[index % colors.length],
    points: (item.oos_backtest?.normalized_equity ?? []).map((point) => ({
      x: point.date,
      value:
        typeof point.value === "number" && Number.isFinite(point.value) ? point.value - 1 : null,
    })),
  }));

  return (
    <div className="finalized-strategies" data-testid="finalized-strategies">
      <header>
        <h4>Predeclared strategy sensitivity</h4>
        <p>
          All strategies below were frozen before this single holdout read. Only the pinned primary
          strategy controls the research verdict.
        </p>
      </header>
      {series.length > 0 ? (
        <LineChart
          title="Holdout cumulative return by strategy"
          description="Sensitivity curves for the same formula. The primary strategy is labelled; comparison curves are diagnostics."
          series={series}
          baseline={0}
          domainFloor={-1}
          baselineAwarePadding
          formatValue={(value) => `${(value * 100).toFixed(1)}%`}
        />
      ) : (
        <p className="strategy-no-exposure">
          No predeclared strategy produced tradable two-sided exposure on the holdout.
        </p>
      )}
      <div className="strategy-result-grid">
        {results.map((item) => {
          const backtest = item.oos_backtest;
          const health = backtest?.portfolio_health;
          const metrics = backtest?.metrics;
          const valid = usableBacktest(backtest);
          const performanceMetrics = valid ? metrics : undefined;
          return (
            <article
              key={item.strategy_id}
              className={`strategy-result${valid ? "" : " is-invalid"}`}
            >
              <span className="strategy-result__title">
                <strong>{strategyName(item)}</strong>
                {item.strategy_id === primaryStrategyId && <small>primary</small>}
              </span>
              <span>Net Sharpe {decimal(performanceMetrics?.net_sharpe)}</span>
              <span>Drawdown {percent(performanceMetrics?.max_drawdown)}</span>
              <span>Turnover {decimal(metrics?.turnover)}</span>
              <span>Active dates {health?.active_observations ?? "—"}</span>
              <span>Exposure coverage {percent(health?.exposure_coverage)}</span>
              {!valid && (
                <small>
                  {health?.reason === "no_exposure"
                    ? "No tradable cross-sectional variation"
                    : health?.reason ?? "Portfolio evidence is unavailable"}
                </small>
              )}
            </article>
          );
        })}
      </div>
    </div>
  );
}

export function Dashboard({
  report,
  history,
  extra,
  enableBenchmarks = false,
  onFinalize,
  finalizing = false,
}: {
  report: Report | null;
  history: HistoryPoint[];
  extra?: RunResult;
  enableBenchmarks?: boolean;
  onFinalize?: (strategyPlanId?: string) => void | Promise<void>;
  finalizing?: boolean;
}) {
  const selection = extra?.selection;
  const evidenceStatus = extra?.finalization?.evidence_status ?? extra?.evidence_status;
  const testReads =
    extra?.session_holdout_reads ??
    extra?.finalization?.session_holdout_reads ??
    extra?.test_reads;
  const repeatedEvidence =
    evidenceStatus === "repeated_same_holdout" ||
    evidenceStatus === "exploratory_repeat" ||
    (testReads ?? 0) > 1;
  const postHoldoutAdaptive =
    extra?.round_metadata?.evidence_status === "post_holdout_adaptive" ||
    evidenceStatus === "post_holdout_adaptive";
  const holdout = extra?.oos_backtest ?? report?.oos_backtest;
  const holdoutMetrics = holdout?.metrics;
  const holdoutHealth = holdout?.portfolio_health;
  const holdoutPerformanceMetrics = usableBacktest(holdout) ? holdoutMetrics : undefined;
  const strategyResults =
    extra?.strategy_results ?? extra?.finalization?.strategy_results ?? [];
  const primaryStrategyId =
    extra?.primary_strategy_id ?? extra?.finalization?.primary_strategy_id;
  const resources = extra?.resources;
  const invalidLegacy =
    extra?.validity === "invalid_legacy_semantics" ||
    extra?.round_metadata?.validity === "invalid_legacy_semantics";
  const validationFailed =
    selection?.validated === false || report?.validation_passed === false;
  const finalized = report !== null;
  const holdoutIntegrity = holdout?.integrity;
  const lastHistory = history[history.length - 1];
  const lowDiversity =
    lastHistory?.diversity_warning === true ||
    lastHistory?.diversity_warning === 1 ||
    (lastHistory?.unique_tree_ratio !== undefined && lastHistory.unique_tree_ratio < 0.6);
  const illustrativeUniverse =
    extra?.context?.universe?.toLowerCase().includes("sp500-lite") ||
    extra?.context?.universe?.toLowerCase().includes("illustrative");
  const freshLockedEvidence =
    !evidenceStatus ||
    evidenceStatus === "locked_first_read" ||
    evidenceStatus === "locked";
  const portfolioInvalid = finalized && holdoutHealth?.valid === false;

  const verdict = invalidLegacy
    ? "Invalid legacy report"
    : validationFailed
      ? "No validated signal"
      : portfolioInvalid
        ? "No tradable portfolio"
      : !finalized
        ? "Validated candidate"
        : repeatedEvidence || postHoldoutAdaptive
          ? "Exploratory evidence"
          : report?.significant && freshLockedEvidence
            ? "Passes checks"
            : "Needs more evidence";

  const validationOrientedIc =
    selection?.median_oriented_ic ??
    (selection
      ? selection.polarity *
        (selection.validation_metrics.signed_ic ?? selection.validation_metrics.ic ?? 0)
      : undefined);
  const parameterVariants =
    selection?.parameter_variants ??
    Object.values(extra?.parameter_coverage ?? {}).reduce(
      (total, item) => total + (item.distinct_parameter_tuples ?? 0),
      0,
    );

  return (
    <div className="dashboard">
      <RunContextLine result={extra} />

      {illustrativeUniverse && (
        <p className="oos-warning" data-testid="small-universe-warning" role="note">
          This 15-stock illustrative universe has high sampling variance. Use a broader prepared
          universe before drawing research conclusions.
        </p>
      )}
      {(repeatedEvidence || postHoldoutAdaptive) && (
        <p className="oos-warning" data-testid="oos-warning" role="alert">
          The trainer selected this formula using training and validation only; holdout returns
          did not influence its objective. This frozen holdout has now been viewed {testReads ?? 1}{" "}
          time{testReads === 1 ? "" : "s"}. Continuing or choosing results after seeing it makes
          later conclusions exploratory. A higher validation score can still produce a lower
          holdout return because the periods contain different market conditions.
        </p>
      )}
      {invalidLegacy && (
        <p className="oos-warning" data-testid="legacy-invalid-warning" role="alert">
          This report used superseded price-adjustment, scoring, or evolution semantics. Its
          metrics remain visible for provenance, but they are not valid evidence. Restart with the
          same setup to produce a corrected result.
        </p>
      )}

      <section className="primary verdict-panel" data-testid="primary-metric">
        <div className="primary-label">Research verdict</div>
        <div className="verdict-layout">
          <Metric
            emphasis
            label="Verdict"
            value={verdict}
            note={
              invalidLegacy
                ? "This result cannot be continued or interpreted under the corrected scorer."
                : validationFailed
                  ? selection?.reason ??
                    report?.validation_reason ??
                    "No candidate was positive in enough chronological validation folds."
                  : portfolioInvalid
                    ? holdoutHealth?.reason === "no_exposure"
                      ? "The formula had no tradable cross-sectional variation on the holdout."
                      : `The primary portfolio failed health checks: ${holdoutHealth?.reason ?? "invalid path"}.`
                  : !finalized
                    ? "Selected without reading the locked holdout. Finalize only when you are ready to inspect it."
                    : repeatedEvidence || postHoldoutAdaptive
                      ? "Useful for exploration, but no longer fresh confirmatory evidence."
                      : "Pass requires validation evidence, DSR probability above 95%, and PBO below 50%."
            }
          />
          <Metric
            label={finalized ? "Oriented OOS rank IC" : "Validation median oriented IC"}
            value={
              report
                ? decimal(holdoutMetrics?.signed_ic ?? report.oos_ic)
                : decimal(validationOrientedIc)
            }
            note={
              finalized
                ? "Direction was fixed before validation or holdout was read."
                : "Median across chronological folds, with polarity fixed from training."
            }
          />
        </div>
      </section>

      {selection && (
        <section className="dashboard-section selection-explanation" data-testid="selection-explanation">
          <header className="dashboard-section__head">
            <div>
              <h3>Why this formula won</h3>
              <p>
                Stable validation direction is ranked before complexity and deterministic
                tie-breaks.
              </p>
            </div>
            <span>
              {selection.positive_folds ?? 0} of {selection.folds?.length ?? 0} folds positive
            </span>
          </header>
          {selection.folds && selection.folds.length > 0 && (
            <div className="validation-fold-grid">
              {selection.folds.map((fold) => (
                <FoldMetric key={fold.index} fold={fold} />
              ))}
            </div>
          )}
          <div className="metric-grid metric-grid--secondary selection-evidence">
            <Metric
              label="Training direction"
              value={selection.polarity < 0 ? "−1 (inverted)" : "+1"}
              note={`Training oriented IC ${decimal(
                selection.training_metrics.oriented_ic ??
                  selection.polarity *
                    (selection.training_metrics.signed_ic ??
                      selection.training_metrics.ic ??
                      0),
              )}`}
            />
            <Metric
              label="Median fold IC"
              value={decimal(selection.median_oriented_ic ?? validationOrientedIc)}
            />
            <Metric label="Worst fold IC" value={decimal(selection.worst_fold_ic)} />
            <Metric
              label="Complexity deduction"
              value={decimal(selection.complexity?.deduction, 4)}
              note={
                selection.complexity
                  ? `${selection.complexity.expanded_nodes} expanded nodes · ${selection.complexity.mode.replace(/_/g, " ")}`
                  : undefined
              }
            />
            <Metric label="Final validation objective" value={decimal(selection.final_objective)} />
            <Metric
              label="Parameter variants tested"
              value={String(parameterVariants)}
            />
            <Metric
              label="Validation coverage"
              value={percent(selection.validation_metrics.valid_date_coverage)}
              note={`${selection.validation_metrics.valid_dates ?? 0} valid IC dates`}
            />
            <Metric
              label="Varying-factor coverage"
              value={percent(selection.validation_metrics.varying_factor_coverage)}
            />
            <Metric
              label="Two-sided exposure coverage"
              value={percent(selection.validation_metrics.two_sided_coverage)}
            />
            <Metric label="Validation oriented IC" value={decimal(validationOrientedIc)} />
            <Metric
              label="Validation sign consistency"
              value={percent(selection.validation_metrics.sign_consistency)}
            />
          </div>
        </section>
      )}

      {!finalized &&
        onFinalize &&
        extra?.session_id &&
        (extra.round_index ?? extra.segment) !== undefined && (
          <StrategyComparisonPanel
            sessionId={extra.session_id}
            roundIndex={(extra.round_index ?? extra.segment) as number}
            onFinalize={(strategyPlanId) => onFinalize(strategyPlanId)}
            finalizing={finalizing}
          />
        )}

      {!finalized &&
        (!onFinalize ||
          !extra?.session_id ||
          (extra.round_index ?? extra.segment) === undefined) && (
        <section className="dashboard-section finalization-callout" data-testid="validation-only">
          <div>
            <h3>Locked holdout is still unopened</h3>
            <p>
              Compare rounds using the validation evidence above. Finalizing evaluates this exact,
              immutable round once against the frozen holdout and does not change the checkpoint.
            </p>
          </div>
          {onFinalize && (
            <button
              type="button"
              className="primary-action"
              onClick={() => void onFinalize()}
              disabled={finalizing}
              data-testid="finalize-round"
            >
              {finalizing ? "Finalizing…" : "Finalize this round"}
            </button>
          )}
        </section>
        )}

      {report && (
        <section className="dashboard-section" data-testid="overfitting-controls">
          <header className="dashboard-section__head">
            <div>
              <h3>Overfitting controls</h3>
              <p>
                Adjusts the evidence for everything searched before the locked holdout was opened.
              </p>
            </div>
          </header>
          <div className="metric-grid metric-grid--secondary">
            <Metric
              label="Deflated Sharpe probability"
              value={percent(report.deflated_sharpe)}
              note={
                report.deflated_sharpe > 0.95
                  ? "Above the 95% pass threshold."
                  : "Below the 95% pass threshold."
              }
            />
            <Metric
              label="Probability of backtest overfitting"
              value={percent(report.pbo)}
              note={
                report.pbo < 0.5
                  ? "Below the 50% risk threshold."
                  : "At or above the 50% risk threshold."
              }
            />
            <Metric
              label="Trials searched"
              value={String(report.n_trials)}
              note="More trials raise the bar for statistical significance."
            />
          </div>
        </section>
      )}

      {report && (
        <section className="dashboard-section holdout-section" data-testid="holdout-performance">
          <header className="dashboard-section__head">
            <div>
              <h3>Locked holdout performance</h3>
              <p>Portfolio evidence calculated during explicit finalization, after costs.</p>
            </div>
            {holdout && (
              <span>
                {holdoutHealth?.calendar_observations ?? holdout.observations} calendar ·{" "}
                {holdoutHealth?.active_observations ?? holdout.observations} active
              </span>
            )}
          </header>
          {holdoutIntegrity && !holdoutIntegrity.valid && (
            <p className="oos-warning" role="alert" data-testid="holdout-integrity-warning">
              This performance path is invalid: {holdoutIntegrity.issues.join("; ")}. Sharpe and
              compounding are withheld after the first unusable observation.
            </p>
          )}
          {holdoutHealth && !holdoutHealth.valid && (
            <p className="strategy-no-exposure" role="alert" data-testid="no-exposure-warning">
              {holdoutHealth.reason === "no_exposure"
                ? "No tradable cross-sectional variation. The portfolio stayed in cash, so no performance curve, Sharpe, or drawdown evidence is available."
                : `Portfolio evidence is unavailable: ${holdoutHealth.reason ?? "failed health checks"}.`}
            </p>
          )}
          {strategyResults.length > 1 && (
            <FinalizedStrategyComparison
              results={strategyResults}
              primaryStrategyId={primaryStrategyId}
            />
          )}
          {usableBacktest(holdout) && (
            <BenchmarkComparison
              equity={holdout?.normalized_equity}
              returns={holdout?.returns}
              enabled={enableBenchmarks}
            />
          )}
          <div className="metric-grid metric-grid--secondary">
            <Metric
              label="Net Sharpe"
              value={decimal(holdoutPerformanceMetrics?.net_sharpe)}
            />
            <Metric
              label="Gross Sharpe"
              value={decimal(holdoutPerformanceMetrics?.gross_sharpe)}
            />
            <Metric
              label="Maximum drawdown"
              value={percent(holdoutPerformanceMetrics?.max_drawdown)}
            />
            <Metric label="Signed rank IC" value={decimal(holdoutMetrics?.signed_ic)} />
            <Metric
              label="Mean |rank IC| (diagnostic)"
              value={decimal(holdoutMetrics?.mean_abs_ic)}
            />
            <Metric label="IC information ratio" value={decimal(holdoutMetrics?.ic_ir)} />
            <Metric label="Average turnover" value={decimal(holdoutMetrics?.turnover)} />
            <Metric
              label="Average positions"
              value={decimal(holdoutMetrics?.avg_positions, 1)}
            />
            <Metric
              label="Average gross exposure"
              value={decimal(holdoutMetrics?.avg_gross)}
            />
            <Metric
              label="Average largest position"
              value={percent(holdoutMetrics?.max_position)}
            />
          </div>
        </section>
      )}

      {finalized &&
        onFinalize &&
        extra?.session_id &&
        (extra.round_index ?? extra.segment) !== undefined && (
          <details className="secondary dashboard-details strategy-repeat">
            <summary>Explore another strategy bundle</summary>
            <p className="hint">
              The holdout has already been viewed. Any new strategy choice and later finalization
              are explicitly exploratory repeated evidence.
            </p>
            <StrategyComparisonPanel
              sessionId={extra.session_id}
              roundIndex={(extra.round_index ?? extra.segment) as number}
              onFinalize={(strategyPlanId) => onFinalize(strategyPlanId)}
              finalizing={finalizing}
              postHoldout
            />
          </details>
        )}

      {extra?.session_id && (extra.round_index ?? extra.segment) !== undefined && (
        <FactorOverlapPanel
          sessionId={extra.session_id}
          roundIndex={(extra.round_index ?? extra.segment) as number}
        />
      )}

      <section className="dashboard-section history" data-testid="search-convergence">
        <header className="dashboard-section__head">
          <div>
            <h3>Search convergence</h3>
            <p>Fitness and IC use different units and are shown separately.</p>
          </div>
          <span>{history.length} generations recorded</span>
        </header>
        {lowDiversity && (
          <p className="oos-warning" role="alert" data-testid="diversity-warning">
            Population diversity is below 60%. Training continues, but repeated formulas may limit
            further improvement.
          </p>
        )}
        <div className="chart-grid-layout">
          <LineChart
            title="Population fitness"
            description="Best and mean fitness by generation. The scale includes zero to avoid exaggerating small improvements."
            baseline={0}
            series={[
              {
                label: "Best fitness",
                color: "#2563eb",
                points: history.map((point) => ({
                  x: point.generation,
                  value: point.best_fitness,
                })),
              },
              {
                label: "Mean fitness",
                color: "#9f1d20",
                points: history.map((point) => ({
                  x: point.generation,
                  value: point.mean_fitness,
                })),
              },
            ]}
          />
          <LineChart
            title="Best oriented rank IC"
            description="Best training-direction rank IC in each generation, shown on its own scale."
            baseline={0}
            series={[
              {
                label: "Best oriented rank IC",
                color: "#0f5b3d",
                points: history.map((point) => ({
                  x: point.generation,
                  value: point.best_ic,
                })),
              },
            ]}
          />
        </div>
        {lastHistory?.unique_tree_ratio !== undefined && (
          <div className="metric-grid metric-grid--secondary">
            <Metric label="Unique-tree ratio" value={percent(lastHistory.unique_tree_ratio)} />
            <Metric
              label="Novel offspring"
              value={String(lastHistory.novel_offspring_count ?? 0)}
            />
            <Metric
              label="Duplicate population entries"
              value={String(lastHistory.duplicate_count ?? 0)}
            />
            <Metric
              label="Champion age"
              value={`${lastHistory.champion_age ?? 0} generations`}
            />
            <Metric
              label="Generations since improvement"
              value={String(lastHistory.generations_since_improvement ?? 0)}
            />
          </div>
        )}
      </section>

      <details className="secondary dashboard-details" data-testid="secondary-metric">
        <summary>In-sample reference</summary>
        <p className="hint">
          Training metrics explain the search but are not evidence of generalization.
        </p>
        <div className="metric-grid metric-grid--secondary">
          <Metric
            label="Train oriented rank IC"
            value={decimal(selection?.training_metrics.oriented_ic ?? report?.train_ic)}
          />
          <Metric
            label="Train signed rank IC"
            value={decimal(selection?.training_metrics.signed_ic)}
          />
          <Metric
            label="Train mean |rank IC| (diagnostic)"
            value={decimal(selection?.training_metrics.mean_abs_ic)}
          />
          <Metric
            label="Train sign consistency"
            value={percent(selection?.training_metrics.sign_consistency)}
          />
          <Metric
            label="Average active names"
            value={decimal(selection?.training_metrics.avg_active_names, 1)}
          />
          {extra?.cumulative_trials !== undefined && (
            <Metric label="Cumulative trials" value={String(extra.cumulative_trials)} />
          )}
        </div>
      </details>

      <details className="secondary dashboard-details" data-testid="runtime-details">
        <summary>Runtime and reproducibility details</summary>
        <div className="metric-grid metric-grid--secondary">
          {testReads !== undefined && (
            <Metric label="Explicit locked-holdout reads" value={String(testReads)} />
          )}
          <Metric
            label="Termination"
            value={extra?.termination_reason?.replace(/_/g, " ") ?? "completed"}
          />
          <Metric label="Training time" value={seconds(extra?.timings?.training_seconds)} />
          <Metric label="Reporting time" value={seconds(extra?.timings?.reporting_seconds)} />
          <Metric label="Total time" value={seconds(extra?.timings?.total_seconds)} />
          {resources && (
            <Metric
              label="Compute resources"
              value={`${resources.effective_workers} workers · ${resources.percent}%`}
            />
          )}
          {extra?.formula_revisions && (
            <Metric label="Pinned formulas" value={String(extra.formula_revisions.length)} />
          )}
        </div>
      </details>
    </div>
  );
}
