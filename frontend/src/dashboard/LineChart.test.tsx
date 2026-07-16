import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { LineChart } from "./LineChart";

describe("LineChart", () => {
  it("renders a readable empty state for missing and non-finite data", () => {
    render(<LineChart title="Empty" description="No values" series={[{ label: "A", color: "blue", points: [{ x: 1, value: null }, { x: 2, value: Number.NaN }] }]} />);
    expect(screen.getByText("No chart data is available.")).toBeInTheDocument();
  });

  it("keeps constant series finite and exposes each point to keyboard users", () => {
    render(<LineChart title="Constant" description="Constant values" baseline={0} series={[{ label: "Fitness", color: "blue", points: [{ x: 0, value: 0.5 }, { x: 1, value: 0.5 }] }]} />);
    expect(screen.getByRole("img")).toBeInTheDocument();
    const points = screen.getAllByLabelText(/Fitness/);
    expect(points).toHaveLength(2);
    expect(points[0]).toHaveAttribute("tabindex", "0");
    expect(document.querySelector("path.chart-series")?.getAttribute("d")).not.toContain("NaN");
  });
});
