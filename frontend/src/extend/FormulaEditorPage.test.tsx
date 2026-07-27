import { act, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { useState } from "react";
import { afterEach, describe, expect, it, vi } from "vitest";
import type { FormulaDraft } from "../api/types";

const getPrimitives = vi.fn();
const listFormulas = vi.fn();
const listFormulaResults = vi.fn();
const getCategories = vi.fn();
const getFormula = vi.fn();
const getFormulaImpact = vi.fn();
const addFormula = vi.fn();
const updateFormula = vi.fn();
const deleteFormula = vi.fn();
const validateFormula = vi.fn();
const setPrimitiveCategory = vi.fn();
const putCategories = vi.fn();

vi.mock("../api/client", () => ({
  getPrimitives: () => getPrimitives(),
  listFormulas: () => listFormulas(),
  listFormulaResults: () => listFormulaResults(),
  listFormulaTests: vi.fn().mockResolvedValue([]),
  listUniverses: vi.fn().mockResolvedValue([]),
  getCategories: () => getCategories(),
  getFormula: (name: string) => getFormula(name),
  getFormulaImpact: (name: string, spec: unknown) => getFormulaImpact(name, spec),
  addFormula: (spec: unknown) => addFormula(spec),
  updateFormula: (name: string, spec: unknown, strategy: string) => updateFormula(name, spec, strategy),
  deleteFormula: (name: string) => deleteFormula(name),
  validateFormula: (spec: unknown) => validateFormula(spec),
  setPrimitiveCategory: (primitive: string, category: string) => setPrimitiveCategory(primitive, category),
  putCategories: (update: unknown) => putCategories(update),
}));

import { FormulaEditorPage } from "./FormulaEditorPage";

const PRIMS = [
  {
    name: "ts_mean",
    display_name: "Moving average",
    description: "Trailing arithmetic mean over a lookback window.",
    kind: "operator",
    arg_types: ["series", "window"],
    inputs: [
      { name: "series", type: "series", description: "Series to average." },
      { name: "lookback", type: "window", description: "Trailing period count." },
    ],
    out_type: "series",
    user: false,
    origin: "builtin",
    category: "time_series",
  },
  {
    name: "rank",
    display_name: "Cross-sectional rank",
    description: "Ranks symbols against one another on each date.",
    kind: "operator",
    arg_types: ["series"],
    inputs: [{ name: "series", type: "series", description: "Series to rank." }],
    out_type: "signal",
    user: false,
    origin: "builtin",
    category: "cross_sectional",
  },
  { name: "close", display_name: "Close", description: "Closing price.", kind: "operand", arg_types: [], inputs: [], out_type: "series", user: false, origin: "data", category: "data" },
  { name: "window", display_name: "Window", description: "Whole-number lookback.", kind: "ephemeral", arg_types: [], inputs: [], out_type: "window", user: false, origin: "value", category: "constant" },
];

function setup() {
  getPrimitives.mockResolvedValue(PRIMS);
  listFormulas.mockResolvedValue([]);
  listFormulaResults.mockResolvedValue([]);
  getCategories.mockResolvedValue({ order: ["data", "time_series", "custom"], overrides: {} });
  validateFormula.mockResolvedValue({ ok: true, out_type: "signal" });
  getFormulaImpact.mockResolvedValue({ name: "my_formula", runtime_name: "my_formula", change: "none", direct_formulas: [], transitive_formulas: [], factors: [], sessions: [], has_references: false });
  getFormula.mockResolvedValue({ revisions: [], impact: {} });
  addFormula.mockImplementation(async (spec) => ({ ...spec, revision: 1, registered: true }));
  updateFormula.mockImplementation(async (_name, spec) => ({ ...spec, revision: 1, registered: true }));
  putCategories.mockResolvedValue({ order: ["data", "time_series", "custom"], overrides: {} });
}

function controlledResizeObserver() {
  const observers: Array<{ callback: ResizeObserverCallback; targets: Set<Element>; instance: ResizeObserver }> = [];
  class ControlledResizeObserver {
    private readonly record: (typeof observers)[number];
    constructor(next: ResizeObserverCallback) {
      this.record = {
        callback: next,
        targets: new Set(),
        instance: this as unknown as ResizeObserver,
      };
      observers.push(this.record);
    }
    observe(target: Element): void {
      this.record.targets.add(target);
    }
    unobserve(target: Element): void {
      this.record.targets.delete(target);
    }
    disconnect(): void {
      this.record.targets.clear();
    }
  }
  vi.stubGlobal("ResizeObserver", ControlledResizeObserver);
  return (target: Element, width: number) => {
    const observer = observers.find((candidate) => candidate.targets.has(target));
    if (!observer) throw new Error("ResizeObserver was not initialized for the target.");
    const entry = {
      target,
      contentRect: { width } as DOMRectReadOnly,
    } as ResizeObserverEntry;
    act(() => observer.callback([entry], observer.instance));
  };
}

afterEach(() => {
  window.sessionStorage.clear();
  vi.clearAllMocks();
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
});

describe("FormulaEditorPage", () => {
  it("renders documented building blocks without per-block category selectors", async () => {
    setup();
    render(<FormulaEditorPage />);
    const library = await screen.findByTestId("formula-library");
    expect(within(library).getByText("Moving average")).toBeInTheDocument();
    expect(within(library).getByText("Trailing arithmetic mean over a lookback window.")).toBeInTheDocument();
    expect(within(library).queryByLabelText("category-ts_mean")).not.toBeInTheDocument();
  });

  it("resizes, snap-collapses, and drags the building-block panel open from the workspace edge", async () => {
    setup();
    render(<FormulaEditorPage />);
    const grid = screen.getByTestId("formula-editor-page").querySelector<HTMLElement>(".formula-workspace__grid")!;
    const handle = screen.getByRole("separator", { name: "Resize building blocks panel" });

    expect(screen.queryByRole("button", { name: /building blocks/i })).not.toBeInTheDocument();
    expect(handle).toHaveAttribute("aria-valuenow", "300");
    expect(grid.style.getPropertyValue("--formula-library-width")).toBe("300px");

    fireEvent(handle, new MouseEvent("pointerdown", { bubbles: true, button: 0, clientX: 300 }));
    fireEvent(window, new MouseEvent("pointermove", { bubbles: true, clientX: 360 }));
    fireEvent(window, new MouseEvent("pointerup", { bubbles: true, clientX: 360 }));
    await waitFor(() => expect(handle).toHaveAttribute("aria-valuenow", "360"));
    expect(grid.style.getPropertyValue("--formula-library-width")).toBe("360px");

    fireEvent(handle, new MouseEvent("pointerdown", { bubbles: true, button: 0, clientX: 360 }));
    fireEvent(window, new MouseEvent("pointermove", { bubbles: true, clientX: 0 }));
    fireEvent(window, new MouseEvent("pointerup", { bubbles: true, clientX: 0 }));
    await waitFor(() => expect(handle).toHaveAttribute("aria-valuetext", "Collapsed"));
    expect(grid).toHaveClass("is-library-collapsed");
    expect(grid.style.getPropertyValue("--formula-library-width")).toBe("0px");

    fireEvent(handle, new MouseEvent("pointerdown", { bubbles: true, button: 0, clientX: 0 }));
    fireEvent(window, new MouseEvent("pointermove", { bubbles: true, clientX: 100 }));
    fireEvent(window, new MouseEvent("pointerup", { bubbles: true, clientX: 100 }));
    await waitFor(() => expect(handle).toHaveAttribute("aria-valuenow", "292"));
    expect(grid).not.toHaveClass("is-library-collapsed");
    expect(grid.style.getPropertyValue("--formula-library-width")).toBe("292px");
  });

  it("supports keyboard resizing and restores the session panel layout", async () => {
    setup();
    const first = render(<FormulaEditorPage />);
    await screen.findByText("Moving average");
    const handle = screen.getByRole("separator", { name: "Resize inspector panel" });

    expect(handle).toHaveAttribute("aria-valuenow", "320");
    fireEvent.keyDown(handle, { key: "ArrowLeft" });
    expect(handle).toHaveAttribute("aria-valuenow", "340");
    fireEvent.keyDown(handle, { key: "Home" });
    expect(handle).toHaveAttribute("aria-valuetext", "Collapsed");
    fireEvent.keyDown(handle, { key: "Enter" });
    expect(handle).toHaveAttribute("aria-valuenow", "340");

    first.unmount();
    render(<FormulaEditorPage />);
    await screen.findByText("Moving average");
    expect(screen.getByRole("separator", { name: "Resize inspector panel" })).toHaveAttribute("aria-valuenow", "340");
  });

  it("fits restored panel preferences to the container and restores them when space returns", async () => {
    setup();
    vi.spyOn(window, "innerWidth", "get").mockReturnValue(1440);
    const resize = controlledResizeObserver();
    window.sessionStorage.setItem("alphalineage.formula-builder.panel-layout.v1", JSON.stringify({
      libraryWidth: 440,
      inspectorWidth: 480,
      libraryCollapsed: false,
      inspectorCollapsed: false,
    }));

    render(<FormulaEditorPage />);
    const grid = screen.getByTestId("formula-editor-page").querySelector<HTMLElement>(".formula-workspace__grid")!;
    const libraryHandle = screen.getByRole("separator", { name: "Resize building blocks panel" });
    const inspectorHandle = screen.getByRole("separator", { name: "Resize inspector panel" });

    resize(grid, 1000);
    await waitFor(() => {
      expect(libraryHandle).toHaveAttribute("aria-valuenow", "270");
      expect(inspectorHandle).toHaveAttribute("aria-valuenow", "293");
    });
    expect(1000 - 16 - 270 - 293).toBeGreaterThanOrEqual(420);
    expect(JSON.parse(window.sessionStorage.getItem("alphalineage.formula-builder.panel-layout.v1")!)).toMatchObject({
      libraryWidth: 440,
      inspectorWidth: 480,
      libraryCollapsed: false,
      inspectorCollapsed: false,
    });

    resize(grid, 900);
    await waitFor(() => {
      expect(libraryHandle).toHaveAttribute("aria-valuenow", "240");
      expect(inspectorHandle).toHaveAttribute("aria-valuetext", "Collapsed");
    });
    expect(grid).not.toHaveClass("is-library-collapsed");
    expect(grid).toHaveClass("is-inspector-collapsed");

    resize(grid, 650);
    await waitFor(() => {
      expect(libraryHandle).toHaveAttribute("aria-valuetext", "Collapsed");
      expect(inspectorHandle).toHaveAttribute("aria-valuetext", "Collapsed");
    });

    resize(grid, 1400);
    await waitFor(() => {
      expect(libraryHandle).toHaveAttribute("aria-valuenow", "440");
      expect(inspectorHandle).toHaveAttribute("aria-valuenow", "480");
    });
    expect(grid).not.toHaveClass("is-library-collapsed");
    expect(grid).not.toHaveClass("is-inspector-collapsed");
  });

  it("prioritizes a clicked inspector at 900px and restores both preferred panels at 1400px", async () => {
    setup();
    vi.spyOn(window, "innerWidth", "get").mockReturnValue(1440);
    const resize = controlledResizeObserver();

    render(<FormulaEditorPage />);
    const grid = screen.getByTestId("formula-editor-page")
      .querySelector<HTMLElement>(".formula-workspace__grid")!;
    const library = await screen.findByTestId("formula-library");
    const libraryHandle = screen.getByRole("separator", { name: "Resize building blocks panel" });
    const inspectorHandle = screen.getByRole("separator", { name: "Resize inspector panel" });

    resize(grid, 900);
    await waitFor(() => expect(inspectorHandle).toHaveAttribute("aria-valuetext", "Collapsed"));
    const marketData = within(library).getByText("Market Data").closest("details")!;
    fireEvent.click(within(marketData).getByText("Market Data"));
    fireEvent.click(within(library).getByRole("button", { name: "Inspect Close" }));

    await waitFor(() => {
      expect(libraryHandle).toHaveAttribute("aria-valuetext", "Collapsed");
      expect(inspectorHandle).toHaveAttribute("aria-valuenow", "320");
    });
    expect(within(screen.getByTestId("formula-inspector")).getByRole("heading", { name: "Close" }))
      .toBeInTheDocument();

    resize(grid, 1400);
    await waitFor(() => {
      expect(libraryHandle).toHaveAttribute("aria-valuenow", "300");
      expect(inspectorHandle).toHaveAttribute("aria-valuenow", "320");
    });
  });

  it("reopens the constrained inspector by dragging its workspace edge", async () => {
    setup();
    vi.spyOn(window, "innerWidth", "get").mockReturnValue(1440);
    const resize = controlledResizeObserver();

    render(<FormulaEditorPage />);
    const grid = screen.getByTestId("formula-editor-page")
      .querySelector<HTMLElement>(".formula-workspace__grid")!;
    const libraryHandle = screen.getByRole("separator", { name: "Resize building blocks panel" });
    const inspectorHandle = screen.getByRole("separator", { name: "Resize inspector panel" });

    resize(grid, 900);
    await waitFor(() => expect(inspectorHandle).toHaveAttribute("aria-valuetext", "Collapsed"));
    fireEvent(
      inspectorHandle,
      new MouseEvent("pointerdown", { bubbles: true, button: 0, clientX: 900 }),
    );
    fireEvent(window, new MouseEvent("pointermove", { bubbles: true, clientX: 800 }));
    fireEvent(window, new MouseEvent("pointerup", { bubbles: true, clientX: 800 }));

    await waitFor(() => {
      expect(libraryHandle).toHaveAttribute("aria-valuetext", "Collapsed");
      expect(inspectorHandle).toHaveAttribute("aria-valuenow", "312");
    });

    resize(grid, 1400);
    await waitFor(() => {
      expect(libraryHandle).toHaveAttribute("aria-valuenow", "300");
      expect(inspectorHandle).toHaveAttribute("aria-valuenow", "312");
    });
  });

  it("leaves tablet panel sizing to the responsive pane layout", () => {
    setup();
    vi.spyOn(window, "innerWidth", "get").mockReturnValue(1000);
    const resize = controlledResizeObserver();
    window.sessionStorage.setItem("alphalineage.formula-builder.panel-layout.v1", JSON.stringify({
      libraryWidth: 440,
      inspectorWidth: 480,
      libraryCollapsed: false,
      inspectorCollapsed: false,
    }));

    render(<FormulaEditorPage />);
    const grid = screen.getByTestId("formula-editor-page").querySelector<HTMLElement>(".formula-workspace__grid")!;
    resize(grid, 650);

    expect(screen.getByRole("separator", { name: "Resize building blocks panel" })).toHaveAttribute("aria-valuenow", "440");
    expect(screen.getByRole("separator", { name: "Resize inspector panel" })).toHaveAttribute("aria-valuenow", "480");
    expect(grid).not.toHaveClass("is-library-collapsed");
    expect(grid).not.toHaveClass("is-inspector-collapsed");
  });

  it("starts every palette section collapsed and restores manual state after search", async () => {
    setup();
    render(<FormulaEditorPage />);
    const library = await screen.findByTestId("formula-library");
    await waitFor(() => expect(within(library).getByText("Time Series")).toBeInTheDocument());

    expect([...library.querySelectorAll("details")]).toHaveLength(4);
    expect([...library.querySelectorAll("details")].every((details) => !details.open)).toBe(true);
    expect(within(library).getAllByText("Market Data")).toHaveLength(1);

    const timeSeries = within(library).getByText("Time Series").closest("details")!;
    fireEvent.click(within(timeSeries).getByText("Time Series"));
    await waitFor(() => expect(timeSeries.open).toBe(true));

    const search = within(library).getByPlaceholderText("Search functions and fields");
    fireEvent.change(search, { target: { value: "close" } });
    const marketData = within(library).getByText("Market Data").closest("details")!;
    expect(marketData.open).toBe(true);
    expect(within(library).queryByText("Time Series")).not.toBeInTheDocument();

    fireEvent.change(search, { target: { value: "" } });
    await waitFor(() => expect(within(library).getByText("Time Series").closest("details")).toHaveProperty("open", true));
    expect(within(library).getByText("Market Data").closest("details")).toHaveProperty("open", false);
  });

  it("uses card clicks for inspection and adds from the inspector", async () => {
    setup();
    const { container } = render(<FormulaEditorPage />);
    const library = await screen.findByTestId("formula-library");
    fireEvent.click(within(library).getByRole("button", { name: "Inspect Moving average", hidden: true }));

    expect(container.querySelectorAll(".formula-block--function")).toHaveLength(0);
    expect(within(library).queryByRole("button", { name: "Inspect" })).not.toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Add to formula" }));
    expect(container.querySelectorAll(".formula-block--function")).toHaveLength(1);
  });

  it("keeps legacy data-category operators out of Market Data", async () => {
    setup();
    getPrimitives.mockResolvedValue([
      ...PRIMS,
      { name: "zscore", display_name: "Cross-sectional z-score", description: "Standardizes each date.", kind: "operator", arg_types: ["series"], inputs: [{ name: "series", type: "series", description: "Series." }], out_type: "signal", user: false, origin: "builtin", category: "data" },
    ]);
    render(<FormulaEditorPage />);
    const library = await screen.findByTestId("formula-library");
    expect(within(library).getAllByText("Market Data")).toHaveLength(1);
    expect(within(library).getByText("Cross-sectional z-score").closest("details")?.querySelector("summary")).toHaveTextContent("Other");

    fireEvent.click(within(library).getByRole("button", { name: "Inspect Close", hidden: true }));
    expect(within(screen.getByTestId("formula-inspector")).queryByLabelText("Category")).not.toBeInTheDocument();
  });

  it("uses market data directly in the synchronized expression editor", async () => {
    setup();
    render(<FormulaEditorPage />);
    await waitFor(() => expect(getPrimitives).toHaveBeenCalled());
    fireEvent.click(screen.getByRole("tab", { name: "Expression" }));
    const expression = screen.getByLabelText("Formula expression");
    fireEvent.change(expression, { target: { value: "ts_mean(close, 10)" } });
    expect(expression).toHaveValue("ts_mean(close, 10)");
    expect(screen.queryByText(/not a declared formula input/)).not.toBeInTheDocument();
  });

  it("keeps built-ins locked and branches from them after confirmation", async () => {
    setup();
    vi.spyOn(window, "confirm").mockReturnValue(true);
    render(<FormulaEditorPage />);
    const library = await screen.findByTestId("formula-library");
    const item = within(library).getByText("Moving average").closest("article")!;
    fireEvent.click(within(item).getByRole("button", { name: "Inspect Moving average" }));
    expect(screen.getByText("Series to average.")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Add to formula" })).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Use as starting point" }));
    fireEvent.click(screen.getByRole("button", { name: "inspector" }));
    expect(screen.getByDisplayValue("ts_mean_custom")).toBeInTheDocument();
  });

  it("lists saved training factors as locked formula sources", async () => {
    setup();
    listFormulaResults.mockResolvedValue([{ id: "factor-1", kind: "training", name: "Momentum winner", saved_at: "2026-06-21", tree: { name: "rank", children: [{ name: "close" }] }, expanded_tree: { name: "rank", children: [{ name: "close" }] }, metrics: { oos_ic: 0.08 }, provenance: { universe: "sp500-lite" }, required_operators: [], notes: "Stable momentum candidate.", disclaimer: "Research only" }]);
    render(<FormulaEditorPage />);
    const library = await screen.findByTestId("formula-library");
    expect(within(library).getByText("Momentum winner")).toBeInTheDocument();
    expect(within(library).getByText("Stable momentum candidate.")).toBeInTheDocument();
  });

  it("separates managed starters, searches aliases, and opens a POSTable editable copy", async () => {
    setup();
    listFormulas.mockResolvedValue([
      {
        name: "ta_sma",
        runtime_name: "ta_sma",
        display_name: "SMA / Simple moving average",
        description: "Arithmetic mean over a trailing window.",
        arg_types: ["series", "window"],
        inputs: [
          { name: "series", type: "series", description: "Input series." },
          { name: "lookback", type: "window", description: "Period.", default: 20 },
        ],
        out_type: "series",
        body: { name: "ts_mean", children: [{ name: "$arg", value: 0 }, { name: "$arg", value: 1 }] },
        category: "technical_indicators",
        origin: "catalog_formula",
        editable: false,
        family: "moving_averages",
        aliases: ["sma", "ma"],
        catalog_revision: 1,
        revision: 1,
        registered: true,
      },
      {
        name: "ta_dea",
        runtime_name: "ta_dea",
        display_name: "DEA / MACD signal line",
        description: "EMA of DIF.",
        arg_types: ["series"],
        inputs: [{ name: "series", type: "series", description: "Input series." }],
        out_type: "series",
        body: { name: "$arg", value: 0 },
        category: "technical_indicators",
        origin: "catalog_formula",
        editable: false,
        family: "macd",
        aliases: ["dea", "mea", "macd signal"],
        catalog_revision: 1,
        revision: 1,
        registered: true,
      },
    ]);
    render(<FormulaEditorPage />);
    const library = await screen.findByTestId("formula-library");
    expect(within(library).getByText("Starter Formulas")).toBeInTheDocument();
    expect(within(library).queryByText("My Formulas")).not.toBeInTheDocument();

    fireEvent.change(within(library).getByPlaceholderText("Search functions and fields"), {
      target: { value: "mea" },
    });
    expect(within(library).getByText("DEA / MACD signal line")).toBeInTheDocument();

    fireEvent.change(within(library).getByPlaceholderText("Search functions and fields"), {
      target: { value: "sma" },
    });
    const sma = within(library).getByText("SMA / Simple moving average").closest("article")!;
    fireEvent.click(within(sma).getByRole("button", { name: "Open" }));
    fireEvent.click(screen.getByRole("button", { name: "inspector" }));
    expect(screen.getByDisplayValue("sma_copy")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Save" }));
    await waitFor(() => expect(addFormula).toHaveBeenCalled());
    expect(updateFormula).not.toHaveBeenCalled();
  });

  it("groups managed indicators stably and opens a concrete literal copy", async () => {
    setup();
    getPrimitives.mockResolvedValue([
      ...PRIMS,
      { name: "ts_ema", display_name: "Exponential moving average", description: "Trailing EMA.", kind: "operator", arg_types: ["series", "window"], inputs: [{ name: "series", type: "series", description: "Input." }, { name: "lookback", type: "window", description: "Period." }], out_type: "series", user: false, origin: "builtin", category: "time_series" },
      { name: "sub", display_name: "Subtract", description: "Left minus right.", kind: "operator", arg_types: ["series", "series"], inputs: [{ name: "left", type: "series", description: "Left." }, { name: "right", type: "series", description: "Right." }], out_type: "series", user: false, origin: "builtin", category: "arithmetic" },
    ]);
    listFormulas.mockResolvedValue([
      {
        name: "ta_macd_histogram",
        runtime_name: "ta_macd_histogram__r2",
        display_name: "MACD Histogram",
        description: "DIF minus DEA using canonical closing prices.",
        arg_types: ["window", "window"],
        inputs: [
          { name: "fast", type: "window", description: "Fast EMA period.", default: 12, role: "parameter", tuning: { enabled: true, min: 8, max: 20, step: 1, radius: 1 } },
          { name: "slow", type: "window", description: "Slow EMA period.", default: 26, role: "parameter", tuning: { enabled: true, min: 20, max: 40, step: 1, radius: 1 } },
        ],
        constraints: [{ left: "fast", operator: "lt", right: "slow" }],
        out_type: "series",
        body: {
          name: "sub",
          children: [
            { name: "ts_ema", children: [{ name: "close" }, { name: "$arg", value: 0 }] },
            { name: "ts_ema", children: [{ name: "close" }, { name: "$arg", value: 1 }] },
          ],
        },
        category: "technical_indicators",
        origin: "catalog_formula",
        editable: false,
        family: "macd",
        family_order: 3,
        status: "active",
        revision: 2,
        registered: true,
      },
      {
        name: "ta_dif",
        runtime_name: "ta_dif__r2",
        display_name: "DIF / MACD Line",
        description: "Fast EMA minus slow EMA.",
        arg_types: ["window", "window"],
        inputs: [
          { name: "fast", type: "window", description: "Fast EMA period.", default: 12 },
          { name: "slow", type: "window", description: "Slow EMA period.", default: 26 },
        ],
        constraints: [{ left: "fast", operator: "lt", right: "slow" }],
        out_type: "series",
        body: { name: "sub", children: [{ name: "close" }, { name: "close" }] },
        category: "technical_indicators",
        origin: "catalog_formula",
        editable: false,
        family: "macd",
        family_order: 1,
        status: "active",
        revision: 2,
        registered: true,
      },
      {
        name: "ta_sma",
        runtime_name: "ta_sma__r2",
        display_name: "SMA",
        description: "Simple moving average.",
        arg_types: ["window"],
        inputs: [{ name: "lookback", type: "window", description: "Period.", default: 20 }],
        out_type: "series",
        body: { name: "ts_mean", children: [{ name: "close" }, { name: "$arg", value: 0 }] },
        category: "technical_indicators",
        origin: "catalog_formula",
        editable: false,
        family: "moving_averages",
        family_order: 1,
        status: "active",
        revision: 2,
        registered: true,
      },
      {
        name: "ta_macd_histogram_2x",
        runtime_name: "ta_macd_histogram_2x",
        display_name: "MACD Histogram (2x)",
        description: "Retired.",
        arg_types: [],
        inputs: [],
        out_type: "series",
        body: { name: "close" },
        category: "technical_indicators",
        origin: "catalog_formula",
        editable: false,
        family: "macd",
        status: "retired",
        replacement: "ta_macd_histogram",
        revision: 1,
        registered: true,
      },
    ]);

    render(<FormulaEditorPage />);
    const library = await screen.findByTestId("formula-library");
    fireEvent.change(within(library).getByPlaceholderText("Search functions and fields"), { target: { value: "macd" } });

    expect(within(library).queryByText("MACD Histogram (2x)")).not.toBeInTheDocument();
    const macdFamily = within(library).getByText("MACD").closest("details")!;
    const indicatorNames = [...macdFamily.querySelectorAll("article strong")].map((element) => element.textContent);
    expect(indicatorNames).toEqual(["DIF / MACD Line", "MACD Histogram"]);

    const macd = within(library).getByRole("button", { name: "Inspect MACD Histogram" });
    expect(macd).toHaveTextContent("fast 12, slow 26 · fast < slow");
    fireEvent.click(macd);
    expect(screen.getByText("Default: 12")).toBeInTheDocument();
    expect(screen.getByText("Training range: 8–20, step 1, local radius 1")).toBeInTheDocument();
    expect(screen.getByText("fast < slow")).toBeInTheDocument();
    fireEvent.click(within(screen.getByTestId("formula-inspector")).getByRole("button", { name: "Open" }));
    fireEvent.click(screen.getByRole("button", { name: "inspector" }));
    expect(screen.queryByLabelText("Input 1 name")).not.toBeInTheDocument();
    fireEvent.click(screen.getByRole("tab", { name: "Expression" }));
    expect(screen.getByLabelText("Formula expression")).toHaveValue("sub(ts_ema(close, 12), ts_ema(close, 26))");
  });

  it("saves a market-data formula without artificial inputs", async () => {
    setup();
    render(<FormulaEditorPage />);
    await waitFor(() => expect(getPrimitives).toHaveBeenCalled());
    fireEvent.click(screen.getByRole("tab", { name: "Expression" }));
    fireEvent.change(screen.getByLabelText("Formula expression"), { target: { value: "rank(close)" } });
    fireEvent.click(screen.getByRole("button", { name: "Save" }));
    await waitFor(() => expect(addFormula).toHaveBeenCalled());
    expect(addFormula.mock.calls[0][0]).toMatchObject({
      name: "my_formula",
      inputs: [],
      body: { name: "rank", children: [{ name: "close" }] },
    });
  });

  it("seeds the starter graph when persisted draft props update during loading", async () => {
    setup();

    function PersistedEditor() {
      const [draft, setDraft] = useState<FormulaDraft>();
      return <FormulaEditorPage formulaDraft={draft} onFormulaDraftChange={setDraft} />;
    }

    render(<PersistedEditor />);
    await waitFor(() => expect(getPrimitives).toHaveBeenCalled());
    fireEvent.click(screen.getByRole("tab", { name: "Expression" }));
    fireEvent.change(screen.getByLabelText("Formula expression"), { target: { value: "rank(close)" } });
    fireEvent.click(screen.getByRole("button", { name: "Save" }));
    await waitFor(() => expect(addFormula).toHaveBeenCalled());
    expect(addFormula.mock.calls[0][0]).toMatchObject({
      body: { name: "rank", children: [{ name: "close" }] },
    });
  });

  it("offers reference upgrade when a calculation is shared", async () => {
    setup();
    const saved = {
      name: "shared_alpha",
      display_name: "Shared alpha",
      description: "Shared formula.",
      arg_types: ["series"],
      inputs: [{ name: "price", type: "series", description: "Price input." }],
      out_type: "signal",
      body: { name: "rank", children: [{ name: "$arg", value: 0 }] },
      category: "custom",
      revision: 1,
      runtime_name: "shared_alpha",
    };
    listFormulas.mockResolvedValue([saved]);
    getPrimitives.mockResolvedValue([...PRIMS, { ...saved, kind: "operator", user: true, origin: "user_formula", logical_name: "shared_alpha" }]);
    getFormula.mockResolvedValue({ ...saved, revisions: [saved], impact: {} });
    getFormulaImpact.mockResolvedValue({ name: "shared_alpha", runtime_name: "shared_alpha", change: "calculation", direct_formulas: ["caller"], transitive_formulas: ["caller"], factors: ["factor-1"], sessions: [], has_references: true });
    render(<FormulaEditorPage />);
    const library = await screen.findByTestId("formula-library");
    const item = within(library).getByText("Shared alpha").closest("article")!;
    fireEvent.click(within(item).getByRole("button", { name: "Open" }));
    fireEvent.click(screen.getByRole("tab", { name: "Expression" }));
    fireEvent.change(screen.getByLabelText("Formula expression"), { target: { value: "ts_mean($price, 10)" } });
    fireEvent.click(screen.getByRole("button", { name: "Save" }));
    expect(await screen.findByRole("dialog", { name: "This calculation is already in use" })).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Upgrade formula references" }));
    await waitFor(() => expect(updateFormula).toHaveBeenCalledWith("shared_alpha", expect.anything(), "upgrade_references"));
  });

  it("starts one Formula Builder with no dummy input", async () => {
    setup();
    render(<FormulaEditorPage />);
    await waitFor(() => expect(getPrimitives).toHaveBeenCalled());
    expect(screen.getByRole("heading", { name: "Formula Builder" })).toBeInTheDocument();
    expect(screen.queryByLabelText("Input 1 name")).not.toBeInTheDocument();
    expect(screen.getByText(/market data \/ saved formulas/)).toBeInTheDocument();
  });

  it("keeps broken saved formulas visible for repair", async () => {
    setup();
    listFormulas.mockResolvedValue([{
      name: "broken_alpha",
      display_name: "Broken alpha",
      description: "Needs migration.",
      arg_types: ["series"],
      inputs: [{ name: "series", type: "series", description: "Input." }],
      out_type: "signal",
      body: { name: "missing_operator", children: [{ name: "$arg", value: 0 }] },
      registered: false,
      error: "missing operator",
    }]);
    render(<FormulaEditorPage />);
    const library = await screen.findByTestId("formula-library");
    const item = within(library).getByText("Broken alpha").closest("article")!;
    expect(within(item).getByRole("button", { name: "Inspect Broken alpha" })).toBeEnabled();
    fireEvent.click(within(item).getByRole("button", { name: "Open" }));
    fireEvent.click(screen.getByRole("button", { name: "inspector" }));
    expect(screen.getByDisplayValue("broken_alpha")).toBeInTheDocument();
  });

  it("commits inline lookbacks as one undoable edit", async () => {
    setup();
    render(<FormulaEditorPage />);
    await waitFor(() => expect(getPrimitives).toHaveBeenCalled());
    fireEvent.click(screen.getByRole("tab", { name: "Expression" }));
    fireEvent.change(screen.getByLabelText("Formula expression"), { target: { value: "ts_mean(close, 10)" } });
    fireEvent.click(screen.getByRole("tab", { name: "Visual" }));

    const lookback = await screen.findByRole("spinbutton", { name: /moving average lookback/i });
    fireEvent.change(lookback, { target: { value: "12" } });
    fireEvent.blur(lookback);
    expect(lookback).toHaveValue(12);

    fireEvent.keyDown(window, { key: "z", ctrlKey: true });
    await waitFor(() => expect(screen.getByRole("spinbutton", { name: /moving average lookback/i })).toHaveValue(10));
    fireEvent.keyDown(window, { key: "z", ctrlKey: true, shiftKey: true });
    await waitFor(() => expect(screen.getByRole("spinbutton", { name: /moving average lookback/i })).toHaveValue(12));
  });

  it("opens a kept formula result as a clean editable copy", async () => {
    setup();
    listFormulaResults.mockResolvedValue([{
      id: "result-1",
      kind: "backtest",
      name: "Momentum Winner 2026",
      saved_at: "2026-07-14",
      tree: { name: "rank", children: [{ name: "close" }] },
      expanded_tree: { name: "rank", children: [{ name: "close" }] },
      out_type: "signal",
      metrics: { ic: 0.04 },
      provenance: { universe: "sp500-lite" },
      required_operators: [],
      notes: "Kept exploratory result.",
      disclaimer: "Research only",
    }]);
    render(<FormulaEditorPage />);
    const library = await screen.findByTestId("formula-library");
    const item = within(library).getByText("Momentum Winner 2026").closest("article")!;
    fireEvent.click(within(item).getByRole("button", { name: "Inspect Momentum Winner 2026" }));
    fireEvent.click(screen.getByRole("button", { name: "Open" }));
    fireEvent.click(screen.getByRole("button", { name: "inspector" }));
    expect(screen.getByDisplayValue("momentum_winner_2026_copy")).toBeInTheDocument();
    expect(screen.queryByLabelText("Input 1 name")).not.toBeInTheDocument();
    fireEvent.click(screen.getByRole("tab", { name: "Expression" }));
    expect(screen.getByLabelText("Formula expression")).toHaveValue("rank(close)");
  });

  it("updates a pinned nested formula node to the latest revision", async () => {
    setup();
    const latest = {
      name: "saved_alpha",
      display_name: "Saved alpha",
      description: "Reusable signal.",
      arg_types: ["series"],
      inputs: [{ name: "series", type: "series", description: "Series input." }],
      out_type: "signal",
      body: { name: "rank", children: [{ name: "$arg", value: 0 }] },
      category: "custom",
      revision: 2,
      runtime_name: "saved_alpha__r2",
      registered: true,
    };
    listFormulas.mockResolvedValue([latest]);
    const draft: FormulaDraft = {
      name: "wrapper",
      display_name: "Wrapper",
      description: "",
      inputs: [],
      out_type: "signal",
      graphNodes: [
        { id: "close", type: "formula", x: 40, y: 80, data: { kind: "data", label: "Close", primitiveName: "close", outType: "series" } },
        { id: "nested", type: "formula", x: 280, y: 80, data: { kind: "function", label: "Saved alpha", primitiveName: "saved_alpha__r1", logicalName: "saved_alpha", origin: "user_formula", inputTypes: ["series"], inputNames: ["series"], inputDescriptions: ["Series input."], outType: "signal", revision: 1 } },
        { id: "formula-output", type: "formula", x: 520, y: 80, data: { kind: "output", label: "Formula output", outType: "signal", locked: true } },
      ],
      graphEdges: [
        { id: "close-nested", source: "close", target: "nested", sourceHandle: "output", targetHandle: "input-0" },
        { id: "nested-output", source: "nested", target: "formula-output", sourceHandle: "output", targetHandle: "result" },
      ],
    };
    const { container } = render(<FormulaEditorPage formulaDraft={draft} />);
    await screen.findByTestId("formula-library");
    const nested = await waitFor(() => {
      const element = container.querySelector<HTMLElement>('[data-id="nested"]');
      expect(element).toBeInTheDocument();
      return element!;
    });
    expect(nested.querySelector(".formula-block__revision")).toHaveTextContent("v1");
    fireEvent.click(nested);
    fireEvent.click(await screen.findByRole("button", { name: "Update to v2" }));
    await waitFor(() => expect(container.querySelector('[data-id="nested"] .formula-block__revision')).toHaveTextContent("v2"));
  });

  it("can switch a numeric argument between an exposed input and an inline literal", async () => {
    setup();
    const draft: FormulaDraft = {
      name: "parameterized_mean",
      display_name: "Parameterized mean",
      description: "",
      inputs: [{ name: "lookback", type: "window", description: "External lookback." }],
      out_type: "series",
      body: { name: "ts_mean", children: [{ name: "close" }, { name: "$arg", value: 0 }] },
    };
    const { container } = render(<FormulaEditorPage formulaDraft={draft} />);
    await screen.findByText("Bound input");
    const mean = container.querySelector<HTMLElement>(".formula-block--function")!;
    fireEvent.click(mean);
    const source = await screen.findByLabelText("lookback source");
    expect(source).toHaveValue("formula-input-0");
    fireEvent.change(source, { target: { value: "" } });
    await waitFor(() => expect(screen.getByRole("spinbutton", { name: /moving average lookback/i })).toHaveValue(20));
  });

});
