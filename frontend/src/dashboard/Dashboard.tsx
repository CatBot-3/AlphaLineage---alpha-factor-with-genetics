import type { HistoryPoint, Report, RunResult } from "../api/types";
import { BenchmarkComparison } from "./BenchmarkComparison";
import { LineChart } from "./LineChart";

function decimal(value: number | null | undefined, digits = 3): string {
  return typeof value === "number" && Number.isFinite(value) ? value.toFixed(digits) : "—";
}

function percent(value: number | null | undefined, digits = 1): string {
  return typeof value === "number" && Number.isFinite(value) ? `${(value * 100).toFixed(digits)}%` : "—";
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
    boundaries?.test_start && boundaries?.test_end && `Locked holdout ${boundaries.test_start}–${boundaries.test_end}`,
    context.horizon != null && `Horizon ${context.horizon}d`,
    context.weighting_scheme && context.weighting_scheme.replace(/_/g, " "),
    context.commission_bps != null && context.slippage_bps != null && `${context.commission_bps + context.slippage_bps} bps costs`,
  ].filter(Boolean) as string[];
  return <div className="run-context" aria-label="Run context">{items.map((item) => <span key={item}>{item}</span>)}</div>;
}

export function Dashboard({
  report,
  history,
  extra,
  enableBenchmarks = false,
}: {
  report: Report;
  history: HistoryPoint[];
  extra?: RunResult;
  enableBenchmarks?: boolean;
}) {
  const testReads = extra?.test_reads;
  const repeatedOos = (testReads ?? 0) > 1;
  const holdout = extra?.oos_backtest ?? report.oos_backtest;
  const holdoutMetrics = holdout?.metrics;
  const resources = extra?.resources;

  return (
    <div className="dashboard">
      <RunContextLine result={extra} />

      {repeatedOos && (
        <p className="oos-warning" data-testid="oos-warning" role="alert">
          The locked holdout has been read {testReads} times in this lineage. Repeatedly
          selecting from those results makes the holdout behave like in-sample data.
        </p>
      )}

      <section className="primary verdict-panel" data-testid="primary-metric">
        <div className="primary-label">Research verdict</div>
        <div className="verdict-layout">
          <Metric
            emphasis
            label="Verdict"
            value={report.significant ? "Passes checks" : "Needs more evidence"}
            note="Pass requires DSR probability above 95% and PBO below 50%."
          />
          <Metric
            label="OOS mean |rank IC|"
            value={decimal(holdoutMetrics?.mean_abs_ic ?? report.oos_ic)}
            note="Absolute predictive association on the locked holdout."
          />
        </div>
      </section>

      <section className="dashboard-section" data-testid="overfitting-controls">
        <header className="dashboard-section__head">
          <div>
            <h3>Overfitting controls</h3>
            <p>Adjusts the evidence for everything searched before the locked holdout was opened.</p>
          </div>
        </header>
        <div className="metric-grid metric-grid--secondary">
          <Metric
            label="Deflated Sharpe probability"
            value={percent(report.deflated_sharpe)}
            note={report.deflated_sharpe > 0.95 ? "Above the 95% pass threshold." : "Below the 95% pass threshold."}
          />
          <Metric
            label="Probability of backtest overfitting"
            value={percent(report.pbo)}
            note={report.pbo < 0.5 ? "Below the 50% risk threshold." : "At or above the 50% risk threshold."}
          />
          <Metric
            label="Trials searched"
            value={String(report.n_trials)}
            note="More trials raise the bar for statistical significance."
          />
        </div>
      </section>

      <section className="dashboard-section holdout-section" data-testid="holdout-performance">
        <header className="dashboard-section__head">
          <div><h3>Locked holdout performance</h3><p>Portfolio evidence calculated once during finalization, after costs.</p></div>
          {holdout && <span>{holdout.observations} observations</span>}
        </header>
        <BenchmarkComparison
          equity={holdout?.normalized_equity}
          returns={holdout?.returns}
          enabled={enableBenchmarks}
        />
        <div className="metric-grid metric-grid--secondary">
          <Metric label="Net Sharpe" value={decimal(holdoutMetrics?.net_sharpe)} />
          <Metric label="Gross Sharpe" value={decimal(holdoutMetrics?.gross_sharpe)} />
          <Metric label="Maximum drawdown" value={percent(holdoutMetrics?.max_drawdown)} />
          <Metric label="Signed rank IC" value={decimal(holdoutMetrics?.signed_ic)} />
          <Metric label="IC information ratio" value={decimal(holdoutMetrics?.ic_ir)} />
          <Metric label="Average turnover" value={decimal(holdoutMetrics?.turnover)} />
          <Metric label="Average positions" value={decimal(holdoutMetrics?.avg_positions, 1)} />
          <Metric label="Average gross exposure" value={decimal(holdoutMetrics?.avg_gross)} />
          <Metric label="Average largest position" value={percent(holdoutMetrics?.max_position)} />
        </div>
      </section>

      <section className="dashboard-section history" data-testid="search-convergence">
        <header className="dashboard-section__head"><div><h3>Search convergence</h3><p>Fitness and IC use different units and are shown separately.</p></div><span>{history.length} generations recorded</span></header>
        <div className="chart-grid-layout">
          <LineChart
            title="Population fitness"
            description="Best and mean fitness by generation. The scale includes zero to avoid exaggerating small improvements."
            baseline={0}
            series={[
              { label: "Best fitness", color: "#2563eb", points: history.map((point) => ({ x: point.generation, value: point.best_fitness })) },
              { label: "Mean fitness", color: "#9f1d20", points: history.map((point) => ({ x: point.generation, value: point.mean_fitness })) },
            ]}
          />
          <LineChart
            title="Best |rank IC|"
            description="Best absolute rank IC found in each generation, shown on its own scale."
            baseline={0}
            series={[{ label: "Best |rank IC|", color: "#0f5b3d", points: history.map((point) => ({ x: point.generation, value: point.best_ic })) }]}
          />
        </div>
      </section>

      <details className="secondary dashboard-details" data-testid="secondary-metric">
        <summary>In-sample reference</summary>
        <p className="hint">Training metrics explain the search but are not evidence of generalization.</p>
        <div className="metric-grid metric-grid--secondary">
          <Metric label="Train mean |rank IC|" value={decimal(report.train_ic)} />
          {extra?.cumulative_trials !== undefined && <Metric label="Cumulative trials" value={String(extra.cumulative_trials)} />}
        </div>
      </details>

      <details className="secondary dashboard-details" data-testid="runtime-details">
        <summary>Runtime and reproducibility details</summary>
        <div className="metric-grid metric-grid--secondary">
          {testReads !== undefined && <Metric label="Locked-holdout reads" value={String(testReads)} />}
          <Metric label="Termination" value={extra?.termination_reason?.replace(/_/g, " ") ?? "completed"} />
          <Metric label="Training time" value={seconds(extra?.timings?.training_seconds)} />
          <Metric label="Reporting time" value={seconds(extra?.timings?.reporting_seconds)} />
          <Metric label="Total time" value={seconds(extra?.timings?.total_seconds)} />
          {resources && <Metric label="Compute resources" value={`${resources.effective_workers} workers · ${resources.percent}%`} />}
          {extra?.formula_revisions && <Metric label="Pinned formulas" value={String(extra.formula_revisions.length)} />}
        </div>
      </details>
    </div>
  );
}
