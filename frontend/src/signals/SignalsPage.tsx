import { useEffect, useRef, useState } from "react";
import { getSignalSnapshotJob, getSignalSnapshot, getSignalSeries, getPortfolioPreview, listSignalSources, listUniverses, startSignalSnapshot, signalExportUrl, stopSignalJob } from "../api/client";
import type { FormulaTestBinding, PortfolioPreview, SignalSeries, SignalSnapshot, SignalSource, SignalsWorkspaceState, UniverseInfo } from "../api/types";
import { executionTimingLabel } from "../train/defaults";
import { UniversePicker } from "../extend/UniverseTree";
import { SignalChart } from "./SignalChart";

export function tradeSentence(snapshot: SignalSnapshot): string {
  return `Signal date ${snapshot.as_of} · Execution: ${executionTimingLabel(snapshot.execution)}${snapshot.execution_delay === 0 ? " (same-close research assumption)" : " after the signal close"}.`;
}
const number = (value: number | null | undefined, digits = 4) => value == null ? "—" : (Math.abs(value) < .5 * 10 ** -digits ? 0 : value).toFixed(digits);
const message = (error: unknown) => error instanceof Error ? error.message : String(error);
const sourceLabel = (source: SignalSource) => `${source.name} · ${source.kind === "saved_result" ? "Library" : source.kind === "round" ? "Training result" : source.kind === "evaluation" ? "Holdout evaluation" : `Formula r${source.revision ?? 1}`}`;

export function SignalsPage({ canRun = true, workspace, onWorkspaceChange }: {
  canRun?: boolean; workspace?: SignalsWorkspaceState; onWorkspaceChange?: (next: SignalsWorkspaceState) => void;
}) {
  const [local, setLocal] = useState<SignalsWorkspaceState>({range: "6M"});
  const state = workspace ?? local;
  const stateRef = useRef(state); stateRef.current = state;
  const update = (patch: Partial<SignalsWorkspaceState>) => {
    const next = {...stateRef.current, ...patch};
    stateRef.current = next; setLocal(next); onWorkspaceChange?.(next);
  };
  const [loadVersion, setLoadVersion] = useState(0);
  const [diagnostics, setDiagnostics] = useState<unknown>(null);
  const [sources, setSources] = useState<SignalSource[] | null>(null);
  const [universes, setUniverses] = useState<UniverseInfo[]>([]);
  const [snapshot, setSnapshot] = useState<SignalSnapshot | null>(null);
  const [series, setSeries] = useState<SignalSeries | null>(null);
  const [portfolio, setPortfolio] = useState<PortfolioPreview | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [progress, setProgress] = useState("");
  const [search, setSearch] = useState("");
  const [excluded, setExcluded] = useState(false);
  const [submitting, setSubmitting] = useState(false);
  const requestVersion = useRef(0);
  const selected = sources?.find(item => item.id === state.source);
  useEffect(() => {
    let active = true;
    Promise.all([listSignalSources(), listUniverses()]).then(([items, list]) => {
      if (!active) return;
      setSources(items); setUniverses(list);
      if (!stateRef.current.source && items[0]) update({source: items[0].id, universe: items[0].universe ?? list[0]?.name});
    }).catch(reason => { if (active) setError(message(reason)); });
    return () => { active = false; requestVersion.current++; };
  }, [loadVersion]);
  useEffect(() => {
    setSnapshot(null); setSeries(null); setPortfolio(null);
    if (!state.snapshotId) return;
    let active = true;
    const check = () => getSignalSnapshot(state.snapshotId!).then(result => { if (active) setSnapshot(result); })
      .catch(reason => { if (active) setError(message(reason)); });
    void check();
    const timer = setInterval(() => void check(), 60_000);
    window.addEventListener("focus", check);
    return () => {active = false; clearInterval(timer); window.removeEventListener("focus", check);};
  }, [state.snapshotId]);
  useEffect(() => {
    if (!state.jobId) return;
    let active = true; let timer: ReturnType<typeof setTimeout>;
    const poll = async () => {
      try {
        const job = await getSignalSnapshotJob(state.jobId!);
        if (!active) return;
        if (job.status === "done" && job.result) {
          const result = job.result;
          update({jobId: undefined, snapshotId: result.snapshot_id,
            symbol: stateRef.current.symbol ?? result.rows[0]?.symbol ?? result.excluded[0]?.symbol,
            strategy: stateRef.current.strategy ?? result.primary_strategy?.scheme ?? "quantile_ls"});
          setSnapshot(result); setProgress(""); return;
        }
        if (["failed", "stopped", "interrupted"].includes(job.status)) { setDiagnostics(job.error_info); throw new Error(job.error ?? `Preparation ${job.status}.`); }
        setProgress(job.progress?.phase ?? job.status);
        timer = setTimeout(poll, 800);
      } catch (reason) { if (active) { setError(message(reason)); update({jobId: undefined}); setProgress(""); } }
    };
    void poll();
    return () => { active = false; clearTimeout(timer); };
  }, [state.jobId]);
  useEffect(() => {
    setSeries(null);
    if (!snapshot?.snapshot_id || !state.symbol) return;
    let active = true;
    getSignalSeries(snapshot.snapshot_id, state.symbol).then(result => {if (active) setSeries(result);})
      .catch(reason => {if (active) setError(message(reason));});
    return () => {active = false;};
  }, [snapshot?.snapshot_id, state.symbol]);
  useEffect(() => {
    setPortfolio(null);
    if (!snapshot?.snapshot_id || state.view !== "portfolio" || snapshot.actionable === false) return;
    let active = true;
    const scheme = state.strategy ?? snapshot.primary_strategy?.scheme ?? "quantile_ls";
    const strategy = scheme === snapshot.primary_strategy?.scheme ? snapshot.primary_strategy
      : {id: "preview", scheme, ...(scheme === "quantile_ls" ? {quantile: .2} : {})};
    getPortfolioPreview(snapshot.snapshot_id, strategy, state.notional).then(result => {if (active) setPortfolio(result);})
      .catch(reason => {if (active) setError(message(reason));});
    return () => {active = false;};
  }, [snapshot?.snapshot_id, state.view, state.strategy, state.notional]);

  function changeCalculation(patch: Partial<SignalsWorkspaceState>) {
    requestVersion.current++; update({...patch, jobId: undefined, snapshotId: undefined});
    setSnapshot(null); setSeries(null); setPortfolio(null); setError(null); setProgress("");
  }
  async function refresh(data_mode: "refresh" | "cached") {
    if (!state.source) return;
    const version = ++requestVersion.current;
    setSubmitting(true); setError(null); setDiagnostics(null); setProgress("checking required data");
    try {
      const job = await startSignalSnapshot({source: state.source, universe: state.universe, data_mode,
        execution: state.execution, direction: state.direction, bindings: state.bindings?.[state.source],
        comparisons: (state.comparisons ?? []).map(source => ({source, bindings: state.bindings?.[source]}))});
      if (version === requestVersion.current) update({jobId: job.job_id});
    } catch (reason) {if (version === requestVersion.current) {setError(message(reason)); setProgress("");}}
    finally {setSubmitting(false);}
  }
  function bindingsFor(source: SignalSource) {
    return (source.inputs ?? []).map(input => {
      const binding = state.bindings?.[source.id]?.[input.name];
      const numeric = input.type === "scalar" || input.type === "window";
      const value = binding?.kind === "literal" ? binding.value : binding?.kind === "field" ? binding.field : input.default ?? "";
      function setBinding(binding: FormulaTestBinding) {
        changeCalculation({bindings: {...state.bindings, [source.id]: {...state.bindings?.[source.id], [input.name]: binding}}});
      }
      return <label className="field" key={`${source.id}:${input.name}`}><span>{source.name} · {input.name}</span>
        {numeric ? <input type="number" aria-label={`${source.name} ${input.name}`} value={value} onChange={e => setBinding({kind: "literal", value: Number(e.target.value)})}/>
          : <select aria-label={`${source.name} ${input.name}`} value={value} onChange={e => setBinding({kind: "field", field: e.target.value})}>
              <option value="">Choose input</option>{["close", "open", "high", "low", "volume", "returns"].map(field => <option key={field}>{field}</option>)}
            </select>}
      </label>;
    });
  }
  const rows = snapshot?.rows ?? [];
  const shown = (excluded ? snapshot?.excluded ?? [] : snapshot?.actionable === false ? snapshot.diagnostic_rows ?? [] : rows)
    .filter(row => row.symbol.toLowerCase().includes(search.toLowerCase()));
  const busy = submitting || Boolean(state.jobId);
  return <div className="signals-workspace" data-testid="signals-page">
    <div className="signals-toolbar">
      <label className="field"><span>Formula</span><select aria-label="Formula" value={state.source ?? ""} onChange={e => {
        const source = sources?.find(item => item.id === e.target.value);
        changeCalculation({source: e.target.value, universe: source?.universe ?? state.universe, execution: undefined, direction: undefined, symbol: undefined});
      }}><option value="">Choose a formula</option>{sources?.map(item => <option key={item.id} value={item.id}>{sourceLabel(item)} · {item.evidence === "validation" ? item.validation_passed ? "validation passed" : "validation not passed" : item.evidence}</option>)}</select></label>
      <div className="field"><span>Universe to rank</span><UniversePicker universes={universes} value={state.universe ?? ""} onChange={universe => changeCalculation({universe, symbol: undefined})}/></div>
      <button className="primary-action" disabled={!canRun || !state.source || busy} onClick={() => void refresh("refresh")}>Refresh &amp; rank</button>
      <button className="ghost" disabled={!canRun || !state.source || busy} onClick={() => void refresh("cached")}>Use cached data</button>
      {state.jobId && <button className="ghost" onClick={() => void stopSignalJob(state.jobId!).then(() => setProgress("stopping")).catch(reason => setError(message(reason)))}>Stop preparation</button>}
    </div>
    {selected && <p className="hint">{selected.evidence === "validation" ? selected.validation_passed ? "Validation passed. Holdout remains locked." : "Validation evaluated; passing criteria were not met. Holdout remains locked." : selected.evidence === "holdout" ? "Holdout opened; inspect performance before relying on this result." : selected.evidence === "unvalidated" ? "Reusable formula · no research validation." : "Saved research evidence"}</p>}
    {selected?.missing_context?.length ? <div className="signals-toolbar">
      <span>Complete the missing context for a portfolio preview:</span>
      {selected.missing_context.includes("execution") && <label>Execution <select aria-label="Signal execution" value={state.execution ?? ""} onChange={e => changeCalculation({execution: e.target.value as SignalsWorkspaceState["execution"]})}><option value="">Choose timing</option><option value="next_open">Next open</option><option value="next_close">Next close</option><option value="close">Same close (research)</option></select></label>}
      {selected.missing_context.includes("direction") && <label>Direction <select aria-label="Ranking direction" value={state.direction ?? ""} onChange={e => changeCalculation({direction: e.target.value as "higher" | "lower"})}><option value="">Choose direction</option><option value="higher">Higher values preferred</option><option value="lower">Lower values preferred</option></select></label>}
    </div> : null}
    <details className="signals-comparisons"><summary>Comparisons ({state.comparisons?.length ?? 0}/3) &amp; formula inputs</summary>
      <div className="signals-toolbar">{selected && bindingsFor(selected)}{(state.comparisons ?? []).map(id => sources?.find(item => item.id === id)).filter((item): item is SignalSource => !!item).flatMap(bindingsFor)}</div>
      <select aria-label="Add comparison" value="" disabled={(state.comparisons?.length ?? 0) >= 3} onChange={e => changeCalculation({comparisons: [...state.comparisons ?? [], e.target.value]})}>
        <option value="">Add a trained result or indicator</option>{sources?.filter(item => item.id !== state.source && !state.comparisons?.includes(item.id)).map(item => <option key={item.id} value={item.id}>{sourceLabel(item)}</option>)}
      </select>
      {state.comparisons?.map(id => <button className="ghost" key={id} onClick={() => changeCalculation({comparisons: state.comparisons?.filter(item => item !== id)})}>Remove {sources?.find(item => item.id === id)?.name ?? id}</button>)}
      <p className="hint">Refresh to calculate all comparisons together with their pinned revisions.</p>
    </details>
    {error && <div role="alert" className="error surface-message">{error} <button className="ghost" onClick={() => { if (sources === null) {setError(null); setLoadVersion(value => value + 1);} else void refresh("refresh"); }} disabled={busy}>{sources === null ? "Retry loading sources" : "Retry refresh"}</button>{diagnostics != null && <details><summary>Technical details</summary><pre>{JSON.stringify(diagnostics, null, 2)}</pre></details>}</div>}
    {progress && <p role="status">{progress.replace(/_/g, " ")}…</p>}
    {sources === null && !error && <p>Loading formula sources…</p>}
    {sources?.length === 0 && <p>No formulas available. Save a training result to Library or create a reusable formula in Build &amp; Data.</p>}
    {!snapshot && !busy && sources?.length ? <p className="surface-message">Choose a formula and universe, then refresh to inspect daily signals.</p> : null}
    {snapshot && <>
      <div className={`signals-freshness ${snapshot.status === "ready" && !snapshot.stale ? "" : "is-warning"}`}>
        <strong>{snapshot.status === "insufficient_coverage" ? "Insufficient coverage · diagnostic inspection only" : snapshot.excluded_count ? "Partial ranking" : "Ranking ready"}: {snapshot.ranked_count}/{snapshot.active_count ?? snapshot.ranked_count + snapshot.excluded_count} stocks</strong>
        <span>Expected completed session: {snapshot.current_expected_session ?? snapshot.expected_session ?? snapshot.as_of} · {tradeSentence(snapshot)} Fetched: {snapshot.fetched_at ?? "cached prices"} · Calculated: {snapshot.computed_at}</span>
        {snapshot.stale && <strong>Stale cached calculation — refresh to use the latest completed market session.</strong>}
        {snapshot.sync?.quota && <span>{snapshot.sync.quota.provider}: {snapshot.sync.quota.scope} allowance used. Retry after the provider quota resets.</span>}
        <span>{snapshot.price_basis ?? "Adjusted prices"} · Initialization: {snapshot.history_start ?? "recorded history"}{snapshot.approximate ? " · incomplete initialization; values are approximate" : ""}</span>
        {snapshot.snapshot_id && <div className="signals-toolbar">
          {snapshot.actionable !== false && <a href={signalExportUrl(snapshot.snapshot_id, "ranking")}>Export ranking CSV</a>}
          <a href={signalExportUrl(snapshot.snapshot_id, "excluded")}>Export exclusions</a>
        </div>}
      </div>
      <div className="signals-grid">
        <aside className="signals-ranking">
          <label>Search stocks<input aria-label="Search stocks" value={search} onChange={e => setSearch(e.target.value)}/></label>
          <button className="ghost" onClick={() => setExcluded(!excluded)}>{excluded ? "Show rankings" : `Show exclusions (${snapshot.excluded_count})`}</button>
          <div className="signals-ranking-scroll"><table><thead><tr><th>Stock</th><th>{excluded ? "Reason" : "Value"}</th>{!excluded && <><th>%ile</th><th>Δ rank</th></>}</tr></thead><tbody>
            {shown.map(row => <tr key={row.symbol} aria-selected={state.symbol === row.symbol}><td><button className="ghost" onClick={() => update({symbol: row.symbol, view: "chart"})}>{row.symbol}</button></td>
              {"reason" in row ? <td>{row.reason.replace(/_/g, " ")}</td> : <><td>{number(row.value)}</td><td>{number(row.percentile, 1)}</td><td>{row.rank_change == null ? "new" : row.rank_change > 0 ? `+${row.rank_change}` : row.rank_change}</td></>}
            </tr>)}
          </tbody></table></div>
        </aside>
        <section className="signals-detail">
          <div className="signals-toolbar"><button className="ghost" aria-pressed={state.view !== "portfolio"} onClick={() => update({view: "chart"})}>Stock inspection</button><button className="ghost" aria-pressed={state.view === "portfolio"} disabled={snapshot.actionable === false} onClick={() => update({view: "portfolio"})}>Portfolio preview</button></div>
          {state.view !== "portfolio" ? <>
            <div className="signals-toolbar">{(["1M", "3M", "6M", "1Y", "All"] as const).map(range => <button className="ghost" key={range} aria-pressed={(state.range ?? "6M") === range} onClick={() => update({range})}>{range}</button>)}<label><input type="checkbox" checked={state.percentile ?? false} onChange={e => update({percentile: e.target.checked})}/> Compare percentiles</label></div>
            {series ? <SignalChart data={series} range={state.range ?? "6M"} percentile={state.percentile ?? false}/> : <p>{state.symbol ? "Loading stock history…" : "Select a stock to inspect its daily candles and signals."}</p>}
          </> : <>
            <div className="signals-toolbar"><label>Strategy <select aria-label="Portfolio strategy" value={state.strategy ?? "quantile_ls"} onChange={e => update({strategy: e.target.value as "quantile_ls" | "rank_proportional"})}><option value="quantile_ls">{snapshot.primary_strategy?.scheme === "quantile_ls" ? `${(snapshot.primary_strategy.quantile ?? .2)*100}% quantile long/short (pinned)` : "20% quantile long/short (default)"}</option><option value="rank_proportional">Rank proportional</option></select></label><label>Indicative notional<input type="number" min="0" aria-label="Indicative notional" value={state.notional ?? ""} onChange={e => update({notional: Number(e.target.value) > 0 ? Number(e.target.value) : undefined})}/></label></div>
            <p>Hypothetical target allocations. No orders or fills are implied. Coverage changes the portfolio.</p>
            {portfolio && <><p>Gross {(portfolio.gross_exposure*100).toFixed(1)}% · Net {(portfolio.net_exposure*100).toFixed(1)}% · Largest position {(portfolio.largest_weight*100).toFixed(1)}% · {portfolio.excluded_count} excluded</p>
              <a href={signalExportUrl(portfolio.snapshot_id, "portfolio", portfolio.strategy)}>Export target weights CSV</a>
              <table><thead><tr><th>Stock</th><th>Side</th><th>Target weight</th><th>Indicative amount</th></tr></thead><tbody>{portfolio.rows.map(row => <tr key={row.symbol}><td>{row.symbol}</td><td>{row.side}</td><td>{(row.weight*100).toFixed(2)}%</td><td>{number(row.amount, 2)}</td></tr>)}</tbody></table></>}
          </>}
        </section>
      </div>
    </>}
  </div>;
}
