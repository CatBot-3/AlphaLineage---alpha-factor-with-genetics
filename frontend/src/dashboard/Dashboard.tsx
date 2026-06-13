// P6-T3: metrics dashboard. User-facing metrics default to OOS/deflated values.

import type { HistoryPoint, Report } from "../api/types";
import { Sparkline } from "./Sparkline";

function Metric({ label, value }: { label: string; value: string }) {
  return (
    <div className="metric">
      <span className="metric-label">{label}</span>
      <span className="metric-value">{value}</span>
    </div>
  );
}

export function Dashboard({ report, history }: { report: Report; history: HistoryPoint[] }) {
  return (
    <div className="dashboard">
      <section className="primary" data-testid="primary-metric">
        <div className="primary-label">Out-of-sample / deflated - the honest default</div>
        <div className="primary-grid">
          <Metric label="OOS rank IC" value={report.oos_ic.toFixed(3)} />
          <Metric label="Deflated Sharpe" value={report.deflated_sharpe.toFixed(3)} />
          <Metric label="PBO" value={report.pbo.toFixed(3)} />
          <Metric
            label="Verdict"
            value={report.significant ? "plausible" : "not significant"}
          />
        </div>
      </section>

      <details className="secondary" data-testid="secondary-metric">
        <summary>In-sample (train) metrics - for reference only</summary>
        <Metric label="Train rank IC" value={report.train_ic.toFixed(3)} />
        <Metric label="Trials searched" value={String(report.n_trials)} />
      </details>

      <section className="history">
        <div className="history-label">Best IC per generation</div>
        <Sparkline values={history.map((point) => point.best_ic)} />
      </section>
    </div>
  );
}
