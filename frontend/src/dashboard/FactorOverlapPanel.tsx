// Overlap with known factors: is the selected formula new, or a known factor in disguise?
// Computed on the training window only, so running it never opens validation or the holdout.

import { useCallback, useEffect, useState } from "react";
import { getRoundOverlap, getRun, startRoundOverlap } from "../api/client";
import type { FactorOverlapReport, OverlapReference, OverlapVerdict } from "../api/types";

const VISIBLE_REFERENCES = 8;

const VERDICTS: Record<OverlapVerdict, { label: string; tone: string; summary: string }> = {
  strong_overlap: { label: "Strong overlap", tone: "is-warning", summary: "This formula shares substantial behavior with a known reference." },
  near_duplicate: {
    label: "Near-identical",
    tone: "is-danger",
    summary:
      "Ranks stocks almost the same way as a known factor. Its IC is real, but it adds little " +
      "to a portfolio that already holds that factor.",
  },
  mostly_explained: {
    label: "Mostly explained",
    tone: "is-warning",
    summary:
      "Several known factors together account for more than half of its IC. The new part is " +
      "the unique IC below.",
  },
  related: {
    label: "Related",
    tone: "is-warning",
    summary:
      "Shares some ranking with known factors, but most of its IC is not explained by them.",
  },
  novel: {
    label: "Novel",
    tone: "is-ok",
    summary: "No known factor ranks stocks in a similar way on the training window.",
  },
  unmeasured: {
    label: "Unmeasured",
    tone: "",
    summary: "No reference factor could be compared on shared dates and stocks.",
  },
};

const GROUP_LABELS: Record<string, string> = {
  catalog: "Starter",
  your_formulas: "Your formula",
  saved_results: "Saved result",
};

function decimal(value: number | null | undefined, digits = 3): string {
  if (typeof value !== "number" || !Number.isFinite(value)) return "—";
  const text = value.toFixed(digits);
  // A unique IC of -0.0004 rounds to the string "-0.000", which reads as a sign that is not
  // there. Anything that rounds to zero is reported as zero.
  return Number(text) === 0 ? (0).toFixed(digits) : text;
}

function percent(value: number | null | undefined): string {
  return typeof value === "number" && Number.isFinite(value)
    ? `${Math.round(value * 100)}%`
    : "—";
}

function CorrelationBar({ value }: { value: number | null | undefined }) {
  const magnitude = typeof value === "number" && Number.isFinite(value) ? Math.min(1, Math.abs(value)) : 0;
  const negative = (value ?? 0) < 0;
  return (
    <span className="overlap-bar" aria-hidden="true">
      <span
        className={`overlap-bar__fill${negative ? " is-negative" : ""}`}
        style={{ width: `${magnitude * 50}%`, [negative ? "right" : "left"]: "50%" }}
      />
    </span>
  );
}

export function FactorOverlapPanel({
  sessionId,
  roundIndex,
}: {
  sessionId: string;
  roundIndex: number;
}) {
  const [report, setReport] = useState<FactorOverlapReport | null>(null);
  const [loaded, setLoaded] = useState(false);
  const [jobId, setJobId] = useState<string | null>(null);
  const [progress, setProgress] = useState<{ done: number; total: number } | null>(null);
  const [includeSaved, setIncludeSaved] = useState(true);
  const [showAll, setShowAll] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    const stored = await getRoundOverlap(sessionId, roundIndex);
    setReport(stored);
    if (stored) setIncludeSaved(stored.include_saved_results);
    return stored;
  }, [roundIndex, sessionId]);

  useEffect(() => {
    let active = true;
    setReport(null);
    setLoaded(false);
    setJobId(null);
    setError(null);
    refresh()
      .catch((reason) => {
        if (active) setError(reason instanceof Error ? reason.message : String(reason));
      })
      .finally(() => {
        if (active) setLoaded(true);
      });
    return () => {
      active = false;
    };
  }, [refresh]);

  useEffect(() => {
    if (!jobId) return;
    let active = true;
    let timer: number | undefined;
    const poll = async () => {
      try {
        const job = await getRun(jobId);
        if (!active) return;
        const snapshot = job.progress;
        if (snapshot?.report_total) {
          setProgress({ done: snapshot.report_done ?? 0, total: snapshot.report_total });
        }
        if (job.status === "done") {
          await refresh();
          if (active) {
            setJobId(null);
            setProgress(null);
          }
          return;
        }
        if (job.status === "failed" || job.status === "stopped") {
          throw new Error(job.error ?? `Overlap check ${job.status}.`);
        }
        timer = window.setTimeout(() => void poll(), 1_000);
      } catch (reason) {
        if (active) {
          setJobId(null);
          setProgress(null);
          setError(reason instanceof Error ? reason.message : String(reason));
        }
      }
    };
    void poll();
    return () => {
      active = false;
      if (timer !== undefined) window.clearTimeout(timer);
    };
  }, [jobId, refresh]);

  async function run() {
    setError(null);
    setShowAll(false);
    try {
      const started = await startRoundOverlap(sessionId, roundIndex, {
        include_saved_results: includeSaved,
      });
      setProgress(null);
      setJobId(started.job_id);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : String(reason));
    }
  }

  const running = jobId !== null;
  const verdict = report ? VERDICTS[report.overlap_version < 2 && report.verdict === "near_duplicate" && (report.max_abs_rank_corr ?? 0) < .98 ? "strong_overlap" : report.verdict] ?? VERDICTS.unmeasured : null;
  const measured = report?.references.filter((item) => item.status === "ok") ?? [];
  const unavailable = report?.references.filter((item) => item.status !== "ok") ?? [];
  const explainedBy = new Set(report?.residual.explained_by ?? []);
  const byKey = new Map<string, OverlapReference>(
    (report?.references ?? []).map((item) => [item.key, item]),
  );
  const closest = measured[0];
  const visible = showAll ? measured : measured.slice(0, VISIBLE_REFERENCES);

  return (
    <section className="dashboard-section factor-overlap" data-testid="factor-overlap">
      <header className="dashboard-section__head">
        <div>
          <h3>Overlap with known factors</h3>
          <p>
            Compares this formula with starter formulas{includeSaved ? ", your formulas and saved results" : " and your formulas"} on
            the training window only. It never reads validation or the locked holdout.
          </p>
        </div>
        {report && (
          <span>
            {report.measured_count} of {report.reference_count} compared · {report.window.start} to{" "}
            {report.window.end}
          </span>
        )}
      </header>

      {report?.own_saved_copies?.length ? <p>{report.own_saved_copies.length} saved copies of this result are excluded from independent reference matches.</p> : null}
      <div className="factor-overlap__actions">
        <label className="seed-option">
          <input
            type="checkbox"
            checked={includeSaved}
            disabled={running}
            onChange={(event) => setIncludeSaved(event.target.checked)}
          />
          <span>Include my saved Formula Results</span>
        </label>
        <button
          type="button"
          className={report ? "ghost" : "primary-action"}
          data-testid="run-overlap"
          disabled={running || !loaded}
          onClick={() => void run()}
        >
          {running
            ? progress
              ? `Comparing… ${Math.round((progress.done / Math.max(1, progress.total)) * 100)}%`
              : "Comparing…"
            : report
              ? "Re-check"
              : "Check overlap"}
        </button>
      </div>

      {error && <p className="error" role="alert">{error}</p>}
      {report?.stale && (
        <p className="oos-warning" data-testid="overlap-stale">
          This check may be out of date: {report.stale_reasons?.join("; ")}. Re-check to refresh it.
        </p>
      )}

      {!report && loaded && !running && (
        <p className="hint">
          Genetic search often rediscovers a published anomaly or an indicator wrapped in a few
          extra operators. Its IC is still real, but it is not new information. This check
          measures how similarly the formula ranks stocks compared with each known factor, and how
          much of its IC is left after removing the most similar ones.
        </p>
      )}

      {report && verdict && (
        <>
          <div className={`factor-overlap__verdict ${verdict.tone}`} data-testid="overlap-verdict">
            <strong>{verdict.label}</strong>
            <span>
              {verdict.summary}
              {closest
                ? ` Closest: ${closest.display_name} (rank correlation ${decimal(closest.mean_rank_corr, 2)}).`
                : ""}
            </span>
          </div>

          <div className="metric-grid metric-grid--secondary">
            <div className="metric">
              <span className="metric-label">IC on comparable data</span>
              <span className="metric-value">{decimal(report.residual.decomposed_ic ?? report.candidate.ic)}</span>
              <small className="metric-note">
                Training-window rank IC where every removed factor has a value (their warm-up
                periods are excluded). Full training IC: {decimal(report.candidate.ic)}.
              </small>
            </div>
            <div className="metric">
              <span className="metric-label">Unique IC</span>
              <span className="metric-value" data-testid="overlap-unique-ic">{decimal(report.residual.ic)}</span>
              <small className="metric-note">
                The part of the IC the {report.residual.explained_by.length || "no"} most similar known
                factor{report.residual.explained_by.length === 1 ? "" : "s"} cannot account for
                {report.residual.ic_t != null ? ` (t = ${decimal(report.residual.ic_t, 1)})` : ""}.
              </small>
            </div>
            <div className="metric">
              <span className="metric-label">Unique share</span>
              <span className="metric-value" data-testid="overlap-unique-share">{percent(report.residual.unique_share)}</span>
              <small className="metric-note">Unique IC divided by the IC above.</small>
            </div>
            <div className="metric">
              <span className="metric-label">Ranking explained</span>
              <span className="metric-value">{percent(report.residual.mean_r_squared)}</span>
              <small className="metric-note">
                Average cross-sectional R² of the formula's ranks on the known factors' ranks.
              </small>
            </div>
          </div>

          {report.residual.explained_by.length > 0 && (
            <p className="hint">
              Removed before measuring unique IC:{" "}
              {report.residual.explained_by
                .map((key) => byKey.get(key)?.display_name ?? key)
                .join(", ")}{" "}
              (the most similar known factors with |rank correlation| ≥ {report.thresholds.related},
              up to {report.thresholds.top_k}, skipping near-copies of one already removed).
            </p>
          )}

          {measured.length > 0 && (
            <div className="factor-overlap__table-scroll">
              <table className="rows factor-overlap__table" data-testid="overlap-references">
                <thead>
                  <tr>
                    <th>Known factor</th>
                    <th>Source</th>
                    <th>Rank correlation</th>
                    <th>Its own training IC</th>
                  </tr>
                </thead>
                <tbody>
                  {visible.map((item) => (
                    <tr key={item.key} className={explainedBy.has(item.key) ? "is-explaining" : undefined}>
                      <td>
                        <strong>{item.display_name}</strong>
                        {explainedBy.has(item.key) && <small> · removed</small>}
                        {item.redundant_with && (
                          <small> · same as {byKey.get(item.redundant_with)?.display_name ?? item.redundant_with}</small>
                        )}
                      </td>
                      <td>{GROUP_LABELS[item.group] ?? item.group} · {item.family.replace(/_/g, " ")}</td>
                      <td>
                        <span className="factor-overlap__corr">
                          <CorrelationBar value={item.mean_rank_corr} />
                          <code>{decimal(item.mean_rank_corr, 2)}</code>
                        </span>
                      </td>
                      <td><code>{decimal(item.reference_ic)}</code></td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
          {measured.length > VISIBLE_REFERENCES && (
            <button type="button" className="show-more" onClick={() => setShowAll((value) => !value)}>
              {showAll ? "Show fewer" : `Show all ${measured.length} compared factors`}
            </button>
          )}
          {unavailable.length > 0 && (
            <details className="factor-overlap__unavailable">
              <summary>{unavailable.length} could not be compared</summary>
              <ul>
                {unavailable.map((item) => (
                  <li key={item.key}>
                    <strong>{item.display_name}</strong>: {item.error ?? item.status}
                  </li>
                ))}
              </ul>
            </details>
          )}
          <small className="hint">
            Rank correlation is the daily cross-sectional Spearman correlation averaged over the
            training window; a strongly negative value is the same idea inverted. Computed{" "}
            {new Date(report.computed_at).toLocaleString()} with the session's{" "}
            {report.horizon}-day horizon and {report.execution.replace("_", " ")} execution.
          </small>
        </>
      )}
    </section>
  );
}
