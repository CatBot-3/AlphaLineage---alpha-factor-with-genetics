import { useEffect, useMemo, useRef, useState, type CSSProperties } from "react";
import {
  getBenchmarkSeries,
  getDataSync,
  listBenchmarks,
  startDataSync,
} from "../api/client";
import type {
  BenchmarkDefinition,
  BenchmarkSeries,
  FormulaTestEquityPoint,
  FormulaTestReturnPoint,
} from "../api/types";
import { LineChart, type ChartSeries } from "./LineChart";

interface BenchmarkChoice {
  visible: boolean;
  loading?: boolean;
  syncing?: boolean;
  data?: BenchmarkSeries;
  error?: string;
}

interface EquityLevel {
  date: string;
  value: number;
}

function finite(value: number | null | undefined): value is number {
  return typeof value === "number" && Number.isFinite(value);
}

export function hasRealizationMetadata(
  returns: FormulaTestReturnPoint[] | undefined,
): boolean {
  return Boolean(
    returns?.length &&
      returns.every((point) => typeof point.signal_date === "string" && point.signal_date),
  );
}

/** Build an explicit pre-return baseline without discarding a legacy first return. */
export function deriveEquityLevels(
  equity: FormulaTestEquityPoint[] | undefined,
  returns: FormulaTestReturnPoint[] | undefined,
): EquityLevel[] {
  const stored = (equity ?? [])
    .filter((point): point is { date: string; value: number } => finite(point.value))
    .map((point) => ({ date: point.date, value: point.value }))
    .sort((left, right) => left.date.localeCompare(right.date));
  const datedReturns = [...(returns ?? [])]
    .sort((left, right) => left.date.localeCompare(right.date));
  const realized = hasRealizationMetadata(datedReturns);
  const hasExplicitBaseline = realized &&
    stored[0]?.date === datedReturns[0]?.signal_date &&
    stored[0]?.value === 1;
  if (hasExplicitBaseline) return stored;
  if (!datedReturns.length) {
    return stored.length
      ? [{ date: `Before ${stored[0].date}`, value: 1 }, ...stored]
      : [];
  }
  let level = 1;
  const baselineDate = realized
    ? datedReturns[0].signal_date!
    : `Before ${datedReturns[0].date}`;
  return [
    { date: baselineDate, value: 1 },
    ...datedReturns.map((point) => {
      // backtest_report treats a missing daily return as flat for the equity path
      if (finite(point.net)) level *= 1 + point.net;
      return { date: point.date, value: level };
    }),
  ].filter((point) => Number.isFinite(point.value));
}

function pointMap(points: Array<{ date: string; value: number | null }>): Map<string, number> {
  return new Map(
    points.flatMap((point) => finite(point.value) ? [[point.date, point.value]] : []),
  );
}

/**
 * Keep the complete factor path visible and join each benchmark at its own first
 * overlapping cached date.  Joining at the factor's level on that date avoids a
 * misleading jump while allowing independently cached benchmarks to cover
 * different portions of the holdout.
 */
export function comparisonChartSeries(
  alpha: EquityLevel[],
  benchmarks: BenchmarkSeries[],
): ChartSeries[] {
  const alphaValues = pointMap(alpha);
  const alphaBase = alpha[0]?.value;
  if (!finite(alphaBase) || alphaBase === 0) return [];

  const factor: ChartSeries = {
    label: "Factor - net after costs",
    color: "#2563eb",
    points: alpha.map((point) => ({
      x: point.date,
      value: point.value / alphaBase - 1,
    })),
  };
  const comparisons = benchmarks.flatMap((benchmark): ChartSeries[] => {
    const values = pointMap(benchmark.normalized_equity);
    const overlap = alpha
      .map((point) => point.date)
      .filter((date) => values.has(date));
    if (!overlap.length) return [];
    const firstDate = overlap[0];
    const benchmarkBase = values.get(firstDate);
    const alphaAtJoin = alphaValues.get(firstDate);
    if (!finite(benchmarkBase) || benchmarkBase === 0 || !finite(alphaAtJoin)) return [];
    const joinedLevel = alphaAtJoin / alphaBase;
    return [{
      label: `${benchmark.label} - index price return`,
      color: benchmark.color,
      points: overlap.map((date) => {
        const value = values.get(date);
        return {
          x: date,
          value: finite(value) ? joinedLevel * (value / benchmarkBase) - 1 : null,
        };
      }),
    }];
  });
  return [factor, ...comparisons];
}

function delay(milliseconds: number): Promise<void> {
  return new Promise((resolve) => window.setTimeout(resolve, milliseconds));
}

export function BenchmarkComparison({
  equity,
  returns,
  enabled = false,
}: {
  equity?: FormulaTestEquityPoint[];
  returns?: FormulaTestReturnPoint[];
  enabled?: boolean;
}) {
  const alpha = useMemo(() => deriveEquityLevels(equity, returns), [equity, returns]);
  const start = alpha[0]?.date;
  const end = alpha[alpha.length - 1]?.date;
  const hasRealizationDates = hasRealizationMetadata(returns);
  const canCompare = enabled && hasRealizationDates && Boolean(start && end);
  const rangeKey = `${canCompare ? "enabled" : "disabled"}:${start ?? ""}:${end ?? ""}`;
  const [catalog, setCatalog] = useState<BenchmarkDefinition[]>([]);
  const [catalogError, setCatalogError] = useState<string | null>(null);
  const [choices, setChoices] = useState<Record<string, BenchmarkChoice>>({});
  const mounted = useRef(true);
  const rangeEpoch = useRef(0);
  const activeRange = useRef(rangeKey);
  const requestTokens = useRef<Record<string, number>>({});
  activeRange.current = rangeKey;

  useEffect(() => {
    mounted.current = true;
    return () => {
      mounted.current = false;
    };
  }, []);

  useEffect(() => {
    const epoch = rangeEpoch.current + 1;
    rangeEpoch.current = epoch;
    requestTokens.current = {};
    setChoices({});
    setCatalogError(null);
    if (!canCompare) {
      setCatalog([]);
      return;
    }
    listBenchmarks()
      .then((items) => {
        if (
          mounted.current &&
          activeRange.current === rangeKey &&
          rangeEpoch.current === epoch
        ) {
          setCatalog(items);
        }
      })
      .catch((reason) => {
        if (
          mounted.current &&
          activeRange.current === rangeKey &&
          rangeEpoch.current === epoch
        ) {
          setCatalogError(String(reason));
        }
      });
  }, [canCompare, end, rangeKey, start]);

  async function loadCached(
    benchmarkId: string,
    makeVisible = true,
  ): Promise<void> {
    if (!start || !end) return;
    const epoch = rangeEpoch.current;
    const token = (requestTokens.current[benchmarkId] ?? 0) + 1;
    requestTokens.current[benchmarkId] = token;
    const isCurrent = () =>
      mounted.current &&
      activeRange.current === rangeKey &&
      rangeEpoch.current === epoch &&
      requestTokens.current[benchmarkId] === token;
    setChoices((current) => ({
      ...current,
      [benchmarkId]: {
        ...current[benchmarkId],
        visible: makeVisible ? true : (current[benchmarkId]?.visible ?? false),
        loading: true,
        error: undefined,
      },
    }));
    try {
      const data = await getBenchmarkSeries(benchmarkId, start, end);
      if (!isCurrent()) return;
      setChoices((current) => ({
        ...current,
        [benchmarkId]: {
          ...current[benchmarkId],
          visible: makeVisible ? true : (current[benchmarkId]?.visible ?? false),
          loading: false,
          data,
          error: undefined,
        },
      }));
    } catch (reason) {
      if (!isCurrent()) return;
      setChoices((current) => ({
        ...current,
        [benchmarkId]: {
          ...current[benchmarkId],
          visible: makeVisible ? true : (current[benchmarkId]?.visible ?? false),
          loading: false,
          error: String(reason),
        },
      }));
    }
  }

  async function syncBenchmark(benchmarkId: string): Promise<void> {
    const request = choices[benchmarkId]?.data?.sync_request;
    if (!request || !start || !end) return;
    const epoch = rangeEpoch.current;
    const token = (requestTokens.current[benchmarkId] ?? 0) + 1;
    requestTokens.current[benchmarkId] = token;
    const isCurrent = () =>
      mounted.current &&
      activeRange.current === rangeKey &&
      rangeEpoch.current === epoch &&
      requestTokens.current[benchmarkId] === token;
    setChoices((current) => ({
      ...current,
      [benchmarkId]: {
        ...current[benchmarkId],
        visible: true,
        syncing: true,
        error: undefined,
      },
    }));
    try {
      const submitted = await startDataSync(request);
      while (isCurrent()) {
        const job = await getDataSync(submitted.job_id);
        if (!isCurrent()) return;
        if (job.status === "done") {
          const data = await getBenchmarkSeries(benchmarkId, start, end);
          if (!isCurrent()) return;
          setChoices((current) => ({
            ...current,
            [benchmarkId]: {
              ...current[benchmarkId],
              visible: current[benchmarkId]?.visible ?? true,
              syncing: false,
              loading: false,
              data,
              error: undefined,
            },
          }));
          break;
        }
        if (job.status === "failed" || job.status === "stopped") {
          throw new Error(job.error ?? "Benchmark data sync did not complete.");
        }
        await delay(600);
      }
    } catch (reason) {
      if (!isCurrent()) return;
      setChoices((current) => ({
        ...current,
        [benchmarkId]: {
          ...current[benchmarkId],
          visible: true,
          syncing: false,
          error: String(reason),
        },
      }));
      return;
    }
  }

  const visibleBenchmarks = catalog.flatMap((benchmark) => {
    const choice = choices[benchmark.id];
    return choice?.visible && choice.data?.normalized_equity.length
      ? [choice.data]
      : [];
  });
  const chartSeries = comparisonChartSeries(alpha, visibleBenchmarks);
  const hasRequestedBenchmark = Object.values(choices).some((choice) => choice.visible);

  if (!alpha.length) {
    return <p className="chart-empty">Equity history is unavailable for this legacy result.</p>;
  }

  return (
    <div className="benchmark-comparison" data-testid="benchmark-comparison">
      {enabled && !hasRealizationDates && (
        <p className="benchmark-status">
          This legacy result has no signal-to-realization dates. Its factor curve keeps the
          first return, but benchmark comparison is available only after rerunning it.
        </p>
      )}
      {canCompare && (
        <div className="benchmark-picker">
          <div>
            <strong>Compare with a benchmark</strong>
            <span>
              Benchmarks are close-to-close price returns, not total returns. Data is loaded
              from the local cache and never downloaded until you choose Sync data.
            </span>
          </div>
          {catalogError && (
            <p className="benchmark-status benchmark-status--error" role="alert">
              Benchmark catalog unavailable: {catalogError}
            </p>
          )}
          <div className="benchmark-options" aria-label="Benchmark visibility">
            {catalog.map((benchmark) => {
              const choice = choices[benchmark.id];
              const checked = choice?.visible ?? false;
              return (
                <label key={benchmark.id}>
                  <input
                    type="checkbox"
                    checked={checked}
                    onChange={(event) => {
                      if (event.target.checked) {
                        void loadCached(benchmark.id);
                      } else {
                        requestTokens.current[benchmark.id] =
                          (requestTokens.current[benchmark.id] ?? 0) + 1;
                        setChoices((current) => ({
                          ...current,
                          [benchmark.id]: {
                            ...current[benchmark.id],
                            visible: false,
                            loading: false,
                            syncing: false,
                          },
                        }));
                      }
                    }}
                  />
                  <span style={{ "--benchmark-color": benchmark.color } as CSSProperties}>
                    {benchmark.label}
                  </span>
                </label>
              );
            })}
          </div>
          {catalog.map((benchmark) => {
            const choice = choices[benchmark.id];
            if (!choice?.visible) return null;
            const needsSync = choice.data?.status === "needs_sync" ||
              choice.data?.status === "partial";
            return (
              <div className="benchmark-status" key={`status:${benchmark.id}`}>
                <span>
                  {choice.loading
                    ? `Checking cached ${benchmark.label} prices…`
                    : choice.error
                      ? `${benchmark.label}: ${choice.error}`
                      : choice.data?.message ??
                        `${benchmark.label} index-level price return is aligned to the factor holdout.`}
                </span>
                {needsSync && (
                  <button
                    type="button"
                    className="ghost"
                    disabled={choice.syncing}
                    onClick={() => void syncBenchmark(benchmark.id)}
                  >
                    {choice.syncing ? "Syncing…" : "Sync data"}
                  </button>
                )}
              </div>
            );
          })}
        </div>
      )}
      <LineChart
        title="Cumulative percentage return"
        description={
          hasRequestedBenchmark
            ? "The factor starts at the holdout baseline. Each index joins it at that index's first cached overlap, so partial benchmark ranges never hide the factor path. The factor is net after costs; indexes are price returns."
            : "The factor's compounded locked-holdout return after commission and slippage, measured from the explicit pre-return baseline."
        }
        baseline={0}
        baselineAwarePadding
        domainFloor={-1}
        formatValue={(value) => `${(value * 100).toFixed(1)}%`}
        series={chartSeries}
      />
    </div>
  );
}
