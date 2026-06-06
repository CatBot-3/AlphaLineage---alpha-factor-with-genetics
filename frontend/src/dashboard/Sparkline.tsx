// Inline-SVG sparkline (no chart library): best IC per generation.

export function Sparkline({
  values,
  width = 240,
  height = 48,
}: {
  values: number[];
  width?: number;
  height?: number;
}) {
  if (values.length === 0) {
    return <svg width={width} height={height} data-testid="sparkline" />;
  }
  const max = Math.max(...values);
  const min = Math.min(...values);
  const span = max - min || 1;
  const points = values
    .map((value, i) => {
      const x = (i / Math.max(values.length - 1, 1)) * (width - 4) + 2;
      const y = height - 2 - ((value - min) / span) * (height - 4);
      return `${x.toFixed(1)},${y.toFixed(1)}`;
    })
    .join(" ");
  return (
    <svg width={width} height={height} data-testid="sparkline" role="img" aria-label="best IC per generation">
      <polyline points={points} fill="none" stroke="#2563eb" strokeWidth={2} />
    </svg>
  );
}
