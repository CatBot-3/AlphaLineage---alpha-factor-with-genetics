import { useEffect, useRef, useState } from "react";
import { CandlestickSeries, ColorType, HistogramSeries, LineSeries, createChart, type Time } from "lightweight-charts";
import type { SignalSeries, SignalsWorkspaceState } from "../api/types";

const COLORS = ["#152451", "#bb8200", "#278775", "#9e557a"];
export function SignalChart({ data, range, percentile }: {
  data: SignalSeries; range: SignalsWorkspaceState["range"]; percentile: boolean;
}) {
  const host = useRef<HTMLDivElement>(null);
  const [date, setDate] = useState<string | null>(null);
  useEffect(() => {
    if (!host.current) return;
    const chart = createChart(host.current, {
      autoSize: true, height: 580,
      layout: { background: { type: ColorType.Solid, color: "#eeeae2" }, textColor: "#152451", attributionLogo: true,
        panes: { enableResize: true, separatorColor: "#c2bfb6", separatorHoverColor: "#bb8200" } },
      grid: { vertLines: { color: "#dedad2" }, horzLines: { color: "#dedad2" } },
      timeScale: { borderColor: "#99968e" },
    });
    const candles = chart.addSeries(CandlestickSeries, { upColor: "#278775", downColor: "#ac5252", borderVisible: false, wickUpColor: "#278775", wickDownColor: "#ac5252" });
    candles.setData(data.bars.map(bar => bar.open !== null && bar.high !== null && bar.low !== null && bar.close !== null
      ? { time: bar.time as Time, open: bar.open, high: bar.high, low: bar.low, close: bar.close }
      : {time: bar.time as Time}));
    const volume = chart.addSeries(HistogramSeries, { color: "#979eaf", priceFormat: { type: "volume" } }, 1);
    volume.setData(data.bars.map(bar => bar.volume === null ? {time: bar.time as Time} : {time: bar.time as Time, value: bar.volume}));
    data.signals.forEach((signal, i) => {
      const line = chart.addSeries(LineSeries, { color: COLORS[i], title: signal.name, lineWidth: 2,
        priceFormat: { type: "price", precision: percentile ? 1 : 4, minMove: percentile ? .1 : .0001 } }, percentile ? 2 : i + 2);
      line.setData(signal.points.map(point => {
        const value = percentile ? point.percentile : point.value;
        return value === null ? {time: point.time as Time} : {time: point.time as Time, value};
      }));
    });
    chart.panes()[0]?.setHeight(280);
    chart.panes()[1]?.setHeight(70);
    const last = data.bars[data.bars.length - 1]?.time;
    if (last && range !== "All") {
      const start = new Date(`${last}T12:00:00Z`);
      start.setUTCMonth(start.getUTCMonth() - ({ "1M": 1, "3M": 3, "6M": 6, "1Y": 12 }[range ?? "6M"]));
      chart.timeScale().setVisibleRange({from: start.toISOString().slice(0, 10) as Time, to: last as Time});
    } else chart.timeScale().fitContent();
    chart.subscribeCrosshairMove(event => {
      const time = event.time;
      setDate(typeof time === "string" ? time : typeof time === "object"
        ? `${time.year}-${String(time.month).padStart(2, "0")}-${String(time.day).padStart(2, "0")}`
        : typeof time === "number" ? new Date(time * 1000).toISOString().slice(0, 10) : null);
    });
    return () => chart.remove();
  }, [data, range, percentile]);
  const hovered = data.bars.find(bar => bar.time === date) ?? data.bars[data.bars.length - 1];
  return <>
    <div className="signals-inspection" aria-live="polite">
      <strong>{data.symbol} · {hovered?.time}</strong>
      {hovered && <span> O {hovered.open?.toFixed(2) ?? "—"} · H {hovered.high?.toFixed(2) ?? "—"} · L {hovered.low?.toFixed(2) ?? "—"} · C {hovered.close?.toFixed(2) ?? "—"} · Volume {hovered.volume?.toLocaleString() ?? "—"}</span>}
      {data.signals.map((signal, i) => {
        const point = signal.points.find(item => item.time === hovered?.time);
        return <div key={signal.source} style={{color: COLORS[i]}}>{signal.name}: {point?.value?.toFixed(4) ?? "gap"} · rank {point?.rank ?? "—"} · percentile {point?.percentile?.toFixed(1) ?? "—"} · coverage {point?.coverage == null ? "—" : `${(point.coverage * 100).toFixed(1)}%`}</div>;
      })}
    </div>
    <div ref={host} className="signal-chart" aria-label={`${data.symbol} daily candles and signals`} />
    <small><a href="https://www.tradingview.com/" target="_blank" rel="noreferrer">TradingView Lightweight Charts™</a> · Copyright © 2025 TradingView, Inc. Historical signals are retrospective inspection.</small>
  </>;
}
