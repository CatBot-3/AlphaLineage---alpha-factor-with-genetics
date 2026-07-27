import { useId, useMemo } from "react";

export interface ChartPoint {
  x: string | number;
  value: number | null | undefined;
}

export interface ChartSeries {
  label: string;
  color: string;
  points: ChartPoint[];
}

function finite(value: number | null | undefined): value is number {
  return typeof value === "number" && Number.isFinite(value);
}

function defaultFormat(value: number): string {
  const magnitude = Math.abs(value);
  if (magnitude >= 100) return value.toFixed(0);
  if (magnitude >= 10) return value.toFixed(1);
  return value.toFixed(3);
}

export function LineChart({
  title,
  description,
  series,
  baseline,
  domainFloor,
  baselineAwarePadding = false,
  height = 230,
  formatValue = defaultFormat,
}: {
  title: string;
  description: string;
  series: ChartSeries[];
  baseline?: number;
  domainFloor?: number;
  baselineAwarePadding?: boolean;
  height?: number;
  formatValue?: (value: number) => string;
}) {
  const titleId = useId();
  const descriptionId = useId();
  const width = 900;
  const padding = { left: 58, right: 18, top: 18, bottom: 34 };
  const labels = useMemo(() => {
    const ordered: Array<string | number> = [];
    const seen = new Set<string>();
    for (const item of series) {
      for (const point of item.points) {
        const key = String(point.x);
        if (!seen.has(key)) {
          seen.add(key);
          ordered.push(point.x);
        }
      }
    }
    return ordered;
  }, [series]);
  const labelIndex = new Map(labels.map((label, index) => [String(label), index]));
  const values = series.flatMap((item) => item.points.map((point) => point.value)).filter(finite);
  if (finite(baseline)) values.push(baseline);

  if (values.length === 0 || labels.length === 0) {
    return <p className="chart-empty">No chart data is available.</p>;
  }

  let minimum = Math.min(...values);
  let maximum = Math.max(...values);
  if (minimum === maximum) {
    const paddingValue = Math.max(Math.abs(minimum) * 0.05, 0.01);
    minimum -= paddingValue;
    maximum += paddingValue;
  } else {
    const range = maximum - minimum;
    if (baselineAwarePadding && finite(baseline) && minimum <= baseline && baseline <= maximum) {
      // Pad each side from its own distance to the baseline. A huge gain should not
      // manufacture an impossible negative tick merely because the full range is large.
      const lowerDistance = baseline - minimum;
      const upperDistance = maximum - baseline;
      minimum -= lowerDistance > 0 ? Math.max(lowerDistance * 0.08, 0.0025) : 0;
      maximum += upperDistance > 0 ? Math.max(upperDistance * 0.06, 0.0025) : 0;
    } else {
      const paddingValue = range * 0.06;
      minimum -= paddingValue;
      maximum += paddingValue;
    }
  }
  if (finite(domainFloor)) minimum = Math.max(domainFloor, minimum);
  if (minimum >= maximum) maximum = minimum + Math.max(Math.abs(minimum) * 0.05, 0.01);
  const xFor = (label: string | number) => {
    const index = labelIndex.get(String(label)) ?? 0;
    return padding.left + (index / Math.max(labels.length - 1, 1)) * (width - padding.left - padding.right);
  };
  const yFor = (value: number) =>
    padding.top + ((maximum - value) / (maximum - minimum)) * (height - padding.top - padding.bottom);
  const yTicks = [maximum, (maximum + minimum) / 2, minimum];
  const xTicks = [...new Set([0, Math.floor((labels.length - 1) / 2), labels.length - 1])];

  return (
    <figure className="line-chart">
      <figcaption>
        <strong id={titleId}>{title}</strong>
        <span id={descriptionId}>{description}</span>
      </figcaption>
      <svg
        viewBox={`0 0 ${width} ${height}`}
        role="img"
        aria-labelledby={`${titleId} ${descriptionId}`}
        data-domain-min={minimum}
        data-domain-max={maximum}
      >
        {yTicks.map((tick) => {
          const y = yFor(tick);
          return <g key={tick}><line className="chart-grid" x1={padding.left} x2={width - padding.right} y1={y} y2={y} /><text className="chart-axis-label" x={padding.left - 8} y={y + 4} textAnchor="end">{formatValue(tick)}</text></g>;
        })}
        {finite(baseline) && <line className="chart-baseline" x1={padding.left} x2={width - padding.right} y1={yFor(baseline)} y2={yFor(baseline)} />}
        <line className="chart-axis" x1={padding.left} x2={padding.left} y1={padding.top} y2={height - padding.bottom} />
        <line className="chart-axis" x1={padding.left} x2={width - padding.right} y1={height - padding.bottom} y2={height - padding.bottom} />
        {xTicks.map((index) => <text className="chart-axis-label" key={index} x={xFor(labels[index])} y={height - 10} textAnchor={index === 0 ? "start" : index === labels.length - 1 ? "end" : "middle"}>{String(labels[index])}</text>)}
        {series.map((item) => {
          let previous = false;
          const path = item.points.map((point) => {
            if (!finite(point.value)) {
              previous = false;
              return "";
            }
            const command = previous ? "L" : "M";
            previous = true;
            return `${command}${xFor(point.x).toFixed(1)},${yFor(point.value).toFixed(1)}`;
          }).join(" ");
          return <g key={item.label}>
            <path className="chart-series" d={path} style={{ stroke: item.color }} />
            {item.points.map((point) => finite(point.value) && <circle key={String(point.x)} className="chart-point" cx={xFor(point.x)} cy={yFor(point.value)} r="4" tabIndex={0} style={{ fill: item.color }} aria-label={`${item.label}, ${point.x}: ${formatValue(point.value)}`}><title>{item.label}: {formatValue(point.value)} at {point.x}</title></circle>)}
          </g>;
        })}
      </svg>
      <div className="chart-legend" aria-label="Chart legend">{series.map((item) => <span key={item.label}><i style={{ background: item.color }} />{item.label}</span>)}</div>
    </figure>
  );
}
