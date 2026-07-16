import { fireEvent, render, screen, waitFor, within } from "@testing-library/react";
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

afterEach(() => {
  vi.clearAllMocks();
  vi.restoreAllMocks();
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
    fireEvent.click(within(item).getByRole("button", { name: "Inspect" }));
    expect(screen.getByText("Series to average.")).toBeInTheDocument();
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
    expect(within(library).getByText("Starter formulas")).toBeInTheDocument();
    expect(within(library).queryByText("My formulas")).not.toBeInTheDocument();

    fireEvent.change(within(library).getByPlaceholderText("Search functions and fields"), {
      target: { value: "mea" },
    });
    expect(within(library).getByText("DEA / MACD signal line")).toBeInTheDocument();

    fireEvent.change(within(library).getByPlaceholderText("Search functions and fields"), {
      target: { value: "sma" },
    });
    const sma = within(library).getByText("SMA / Simple moving average").closest("article")!;
    fireEvent.click(within(sma).getByRole("button", { name: "Open as editable copy" }));
    fireEvent.click(screen.getByRole("button", { name: "inspector" }));
    expect(screen.getByDisplayValue("sma_copy")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Save" }));
    await waitFor(() => expect(addFormula).toHaveBeenCalled());
    expect(updateFormula).not.toHaveBeenCalled();
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
    fireEvent.click(within(item).getByRole("button", { name: "Edit" }));
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
    expect(within(item).getByRole("button", { name: /Broken alpha/ })).toBeDisabled();
    fireEvent.click(within(item).getByRole("button", { name: "Repair" }));
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
    fireEvent.click(within(item).getByRole("button", { name: "Inspect" }));
    fireEvent.click(screen.getByRole("button", { name: "Open as editable copy" }));
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
