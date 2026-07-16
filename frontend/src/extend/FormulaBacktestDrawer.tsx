import { useEffect, useMemo, useState } from "react";
import {
  clearFormulaTest,
  getFormulaTest,
  keepFormulaTest,
  listFormulaResults,
  listFormulaTests,
  listUniverses,
  startFormulaTest,
  stopFormulaTest,
} from "../api/client";
import type {
  FormulaInputSpec,
  FormulaResult,
  FormulaResultSeriesPoint,
  FormulaSpec,
  FormulaTestBinding,
  FormulaTestJob,
  FormulaTestSource,
  UniverseInfo,
} from "../api/types";

const DATA_FIELDS = ["open", "high", "low", "close", "volume", "vwap", "returns"];
const COLORS = ["#0f1c4a", "#a97700", "#0f5b3d", "#9f1d20"];

interface ComparableResult {
  key: string;
  label: string;
  metrics: Record<string, unknown>;
  series: FormulaResultSeriesPoint[];
  temporary: boolean;
  jobId?: string;
}

function compatible(actual: string | undefined, expected: string): boolean {
  if (!actual) return true;
  if (actual === expected) return true;
  return (actual === "series" || actual === "signal") &&
    (expected === "series" || expected === "signal");
}

function finite(value: unknown): number | null {
  const number = typeof value === "number" ? value : Number(value);
  return Number.isFinite(number) ? number : null;
}

function metric(value: unknown, digits = 3): string {
  const number = finite(value);
  return number == null ? "—" : number.toFixed(digits);
}

function coverageDates(universe: UniverseInfo | undefined): { start: string; end: string } {
  const rows = Object.values(universe?.cache_coverage?.symbol_coverage ?? {});
  const starts = rows.flatMap((row) => row.first_date ? [row.first_date] : []);
  const ends = rows.flatMap((row) => row.last_date ? [row.last_date] : []);
  return {
    start: starts.sort()[0] ?? "",
    end: ends.sort()[Math.max(ends.length - 1, 0)] ?? universe?.cache_coverage?.as_of ?? "",
  };
}

function EquityChart({ items }: { items: ComparableResult[] }) {
  const width = 640;
  const height = 210;
  const padding = 20;
  const values = items.flatMap((item) => item.series.map((point) => finite(point.equity)).filter((value): value is number => value != null));
  if (!values.length) return <p className="hint">Equity history is unavailable for the selected result.</p>;
  const min = Math.min(...values);
  const max = Math.max(...values);
  const span = Math.max(max - min, 1e-9);
  return (
    <div className="formula-backtest-chart" aria-label="Normalized net equity comparison">
      <svg viewBox={`0 0 ${width} ${height}`} role="img">
        <line x1={padding} y1={height - padding} x2={width - padding} y2={height - padding} />
        {items.map((item, itemIndex) => {
          const points = item.series
            .map((point, index) => ({ value: finite(point.equity), index }))
            .filter((point): point is { value: number; index: number } => point.value != null);
          if (points.length < 2) return null;
          const denominator = Math.max(item.series.length - 1, 1);
          const path = points.map((point) => {
            const x = padding + (point.index / denominator) * (width - padding * 2);
            const y = height - padding - ((point.value - min) / span) * (height - padding * 2);
            return `${x.toFixed(1)},${y.toFixed(1)}`;
          }).join(" ");
          return <polyline key={item.key} points={path} fill="none" stroke={COLORS[itemIndex % COLORS.length]} strokeWidth="2" />;
        })}
      </svg>
      <div className="formula-backtest-chart__legend">
        {items.map((item, index) => <span key={item.key}><i style={{ background: COLORS[index % COLORS.length] }} />{item.label}</span>)}
      </div>
    </div>
  );
}

function payloadSeries(
  returns: Array<{ date: string; gross: number | null; net: number | null }> = [],
  equity: Array<{ date: string; value: number | null }> = [],
): FormulaResultSeriesPoint[] {
  const values = new Map(equity.map((point) => [point.date, point.value]));
  return returns.map((point) => ({
    date: point.date,
    gross_return: point.gross,
    net_return: point.net,
    equity: values.get(point.date) ?? null,
  }));
}

export function FormulaBacktestDrawer({
  open,
  source,
  inputs,
  formulas,
  defaultName,
  defaultUniverse,
  onDataSync,
  onClose,
}: {
  open: boolean;
  source: FormulaTestSource | null;
  inputs: FormulaInputSpec[];
  formulas: FormulaSpec[];
  defaultName: string;
  defaultUniverse?: string;
  onDataSync?: () => void;
  onClose: () => void;
}) {
  const [universes, setUniverses] = useState<UniverseInfo[]>([]);
  const [jobs, setJobs] = useState<FormulaTestJob[]>([]);
  const [results, setResults] = useState<FormulaResult[]>([]);
  const [bindings, setBindings] = useState<Record<string, FormulaTestBinding>>({});
  const [universe, setUniverse] = useState("");
  const [start, setStart] = useState("");
  const [end, setEnd] = useState("");
  const [horizon, setHorizon] = useState(1);
  const [scheme, setScheme] = useState<"quantile_ls" | "rank_proportional">("quantile_ls");
  const [quantile, setQuantile] = useState(0.2);
  const [commission, setCommission] = useState(1);
  const [slippage, setSlippage] = useState(5);
  const [selected, setSelected] = useState<string[]>([]);
  const [keepName, setKeepName] = useState(defaultName);
  const [error, setError] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);

  function refresh() {
    Promise.all([listFormulaTests(), listFormulaResults()])
      .then(([nextJobs, nextResults]) => {
        setJobs(nextJobs);
        setResults(nextResults);
      })
      .catch((reason) => setError(String(reason)));
  }

  useEffect(() => {
    if (!open) return;
    setKeepName(defaultName);
    // Bindings are part of the experiment, not an implicit property of a formula.
    // Start each newly opened/source-switched test blank even when an exposed
    // numeric input has an authoring default.
    setBindings({});
    setSelected([]);
    listUniverses()
      .then((items) => {
        setUniverses(items);
        setUniverse((current) => {
          const preferred = items.find((item) => item.name === defaultUniverse)?.name;
          const retained = items.find((item) => item.name === current)?.name;
          return preferred || retained || items[0]?.name || "";
        });
      })
      .catch((reason) => setError(String(reason)));
    refresh();
  }, [defaultName, defaultUniverse, open, source]);

  useEffect(() => {
    if (!open || !jobs.some((job) => job.status === "queued" || job.status === "running")) return;
    const timer = window.setInterval(refresh, 800);
    return () => window.clearInterval(timer);
  }, [jobs, open]);

  useEffect(() => {
    const selectedUniverse = universes.find((item) => item.name === universe);
    const dates = coverageDates(selectedUniverse);
    setStart(dates.start);
    setEnd(dates.end);
  }, [universe, universes]);

  const comparable = useMemo<ComparableResult[]>(() => {
    const temporary = jobs.flatMap((job) => job.status === "done" && job.result ? [{
      key: `job:${job.job_id}`,
      label: `Backtest ${job.job_id.slice(0, 6)}`,
      metrics: { ...job.result.metrics },
      series: payloadSeries(job.result.returns, job.result.normalized_equity),
      temporary: true,
      jobId: job.job_id,
    }] : []);
    const kept = results.map((result) => ({
      key: `result:${result.id}`,
      label: result.name,
      metrics: result.metrics,
      series: result.series ?? payloadSeries(result.returns, result.normalized_equity),
      temporary: false,
    }));
    return [...temporary, ...kept];
  }, [jobs, results]);

  const selectedItems = selected.flatMap((key) => {
    const item = comparable.find((candidate) => candidate.key === key);
    return item ? [item] : [];
  });
  const active = jobs.find((job) => job.status === "queued" || job.status === "running");
  const missingBindings = inputs.filter((input) => !bindings[input.name]);
  const selectedUniverse = universes.find((item) => item.name === universe);
  const coverageComplete = selectedUniverse?.cache_coverage?.complete !== false;

  function updatePanelBinding(inputName: string, encoded: string) {
    if (!encoded) {
      setBindings((current) => {
        const next = { ...current };
        delete next[inputName];
        return next;
      });
      return;
    }
    const [kind, value] = encoded.split(":", 2);
    const binding: FormulaTestBinding = kind === "data"
      ? { kind: "field", field: value }
      : kind === "formula"
        ? { kind: "formula", runtime_name: value }
        : { kind: "result", result_id: value };
    setBindings((current) => ({ ...current, [inputName]: binding }));
  }

  async function run() {
    if (!source || !universe || !end || missingBindings.length) return;
    setSubmitting(true);
    setError(null);
    try {
      const job = await startFormulaTest({
        source,
        bindings,
        universe,
        start: start || null,
        end,
        horizon,
        weighting_scheme: scheme,
        quantile: scheme === "quantile_ls" ? quantile : undefined,
        commission_bps: commission,
        slippage_bps: slippage,
      });
      setJobs((items) => [job, ...items.filter((item) => item.job_id !== job.job_id)]);
      const loaded = await getFormulaTest(job.job_id).catch(() => job);
      setJobs((items) => [loaded, ...items.filter((item) => item.job_id !== loaded.job_id)]);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : String(reason));
    } finally {
      setSubmitting(false);
    }
  }

  async function keep(item: ComparableResult) {
    if (!item.jobId || !keepName.trim()) return;
    try {
      await keepFormulaTest(item.jobId, { name: keepName.trim() });
      refresh();
    } catch (reason) {
      setError(String(reason));
    }
  }

  if (!open) return null;
  return (
    <aside className="formula-backtest" aria-label="Formula backtest">
      <header>
        <div><span className="mode-chip">Exploratory research</span><h3>Backtest formula</h3></div>
        <button type="button" onClick={onClose} aria-label="Close backtest">Close</button>
      </header>
      <p className="hint">This test uses research data and can be repeated. It is not a locked out-of-sample verdict.</p>
      {error && <p className="error">{error}</p>}

      <div className="formula-backtest__settings">
        {inputs.length > 0 && <fieldset>
          <legend>Formula input bindings</legend>
          {inputs.map((input, index) => {
            const binding = bindings[input.name];
            const numeric = input.type === "scalar" || input.type === "window";
            return <label className="field" key={`${input.name}-${index}`}>
              <span className="field-label">{input.name} <small>{input.type}</small></span>
              {numeric ? <input
                type="number"
                step={input.type === "window" ? 1 : "any"}
                min={input.type === "window" ? 1 : undefined}
                placeholder={input.default == null ? "Enter a value" : `Default ${input.default}; enter to bind`}
                value={binding?.kind === "literal" ? binding.value : ""}
                onChange={(event) => {
                  const value = Number(event.target.value);
                  setBindings((current) => event.target.value && Number.isFinite(value)
                    ? { ...current, [input.name]: { kind: "literal", value } }
                    : Object.fromEntries(Object.entries(current).filter(([key]) => key !== input.name)));
                }}
              /> : <select value={binding?.kind === "field" ? `data:${binding.field}` : binding?.kind === "formula" ? `formula:${binding.runtime_name}` : binding?.kind === "result" ? `result:${binding.result_id}` : ""} onChange={(event) => updatePanelBinding(input.name, event.target.value)}>
                <option value="">Choose explicitly…</option>
                {(input.type === "series" || input.type === "signal") && DATA_FIELDS.map((field) => <option key={field} value={`data:${field}`}>Data · {field}</option>)}
                {formulas.filter((formula) => formula.arg_types.length === 0 && compatible(formula.out_type, input.type)).map((formula) => <option key={formula.runtime_name ?? formula.name} value={`formula:${formula.runtime_name ?? formula.name}`}>Formula · {formula.display_name || formula.name} v{formula.revision ?? 1}</option>)}
                {results.filter((result) => compatible(result.out_type, input.type)).map((result) => <option key={result.id} value={`result:${result.id}`}>Result · {result.name}</option>)}
              </select>}
              {input.description && <small>{input.description}</small>}
            </label>;
          })}
        </fieldset>}

        <div className="formula-backtest__grid">
          <label className="field"><span className="field-label">Universe</span><select value={universe} onChange={(event) => setUniverse(event.target.value)}>{universes.map((item) => <option key={item.name} value={item.name}>{item.display_name ?? item.name}</option>)}</select></label>
          <label className="field"><span className="field-label">Start</span><input type="date" value={start} onChange={(event) => setStart(event.target.value)} /></label>
          <label className="field"><span className="field-label">End</span><input type="date" value={end} onChange={(event) => setEnd(event.target.value)} /></label>
          <label className="field"><span className="field-label">Forward horizon</span><input type="number" min={1} value={horizon} onChange={(event) => setHorizon(Math.max(1, Number(event.target.value) || 1))} /></label>
          <label className="field"><span className="field-label">Portfolio</span><select value={scheme} onChange={(event) => setScheme(event.target.value as typeof scheme)}><option value="quantile_ls">Quantile long/short</option><option value="rank_proportional">Rank proportional</option></select></label>
          {scheme === "quantile_ls" && <label className="field"><span className="field-label">Tail quantile</span><input type="number" min={0.01} max={0.49} step={0.01} value={quantile} onChange={(event) => setQuantile(Number(event.target.value))} /></label>}
          <label className="field"><span className="field-label">Commission (bps)</span><input type="number" min={0} step={0.1} value={commission} onChange={(event) => setCommission(Number(event.target.value))} /></label>
          <label className="field"><span className="field-label">Slippage (bps)</span><input type="number" min={0} step={0.1} value={slippage} onChange={(event) => setSlippage(Number(event.target.value))} /></label>
        </div>
        {!coverageComplete && <div className="surface-message formula-backtest__coverage-warning">
          <span>This universe has incomplete cached data. Sync it before relying on the result.</span>
          <button type="button" onClick={onDataSync} disabled={!onDataSync}>Data Sync</button>
        </div>}
        {missingBindings.length > 0 && <p className="hint">Bind {missingBindings.map((input) => input.name).join(", ")} before running.</p>}
        <div className="actions">
          <button type="button" className="primary-action" onClick={run} disabled={submitting || Boolean(active) || !source || !universe || !end || missingBindings.length > 0}>Run backtest</button>
          {active && <button type="button" onClick={() => stopFormulaTest(active.job_id).then(refresh)}>Stop</button>}
          {active && <span className="hint">{active.status === "queued" ? "Queued" : "Evaluating…"}</span>}
        </div>
      </div>

      {comparable.length > 0 && <section className="formula-backtest__results">
        <h4>Compare results</h4>
        <p className="hint">Select up to four. Temporary runs are not saved automatically.</p>
        <div className="formula-backtest__result-list">
          {comparable.map((item) => <article key={item.key}>
            <label><input type="checkbox" checked={selected.includes(item.key)} disabled={!selected.includes(item.key) && selected.length >= 4} onChange={(event) => setSelected((current) => event.target.checked ? [...current, item.key] : current.filter((key) => key !== item.key))} /> <strong>{item.label}</strong></label>
            <span>{item.temporary ? "Temporary" : "Kept"}</span>
            <small>Net Sharpe {metric(item.metrics.net_sharpe)} · Drawdown {metric(item.metrics.max_drawdown)}</small>
            {item.temporary && <div className="actions"><input aria-label={`Name ${item.label}`} value={keepName} onChange={(event) => setKeepName(event.target.value)} /><button type="button" onClick={() => keep(item)}>Keep result</button><button type="button" onClick={() => item.jobId && clearFormulaTest(item.jobId).then(refresh)}>Clear</button></div>}
          </article>)}
        </div>
        {selectedItems.length > 0 && <>
          <div className="formula-backtest__metrics">
            <table><thead><tr><th>Result</th><th>IC</th><th>IC-IR</th><th>Gross Sharpe</th><th>Net Sharpe</th><th>Drawdown</th><th>Turnover</th></tr></thead><tbody>{selectedItems.map((item) => <tr key={item.key}><th>{item.label}</th><td>{metric(item.metrics.ic ?? item.metrics.oos_ic)}</td><td>{metric(item.metrics.ic_ir)}</td><td>{metric(item.metrics.gross_sharpe)}</td><td>{metric(item.metrics.net_sharpe)}</td><td>{metric(item.metrics.max_drawdown)}</td><td>{metric(item.metrics.turnover)}</td></tr>)}</tbody></table>
          </div>
          <EquityChart items={selectedItems} />
        </>}
      </section>}
    </aside>
  );
}
