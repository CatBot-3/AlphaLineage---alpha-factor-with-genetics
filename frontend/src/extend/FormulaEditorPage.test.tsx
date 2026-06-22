import { fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { useState } from "react";
import { afterEach, describe, expect, it, vi } from "vitest";
import type { FormulaDraft } from "../api/types";

const getPrimitives = vi.fn();
const listFormulas = vi.fn();
const listFactors = vi.fn();
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
  listFactors: () => listFactors(),
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
  listFactors.mockResolvedValue([]);
  getCategories.mockResolvedValue({ order: ["data", "time_series", "custom"], overrides: {} });
  validateFormula.mockResolvedValue({ ok: true, out_type: "signal" });
  getFormulaImpact.mockResolvedValue({ name: "my_formula", runtime_name: "my_formula", change: "none", direct_formulas: [], transitive_formulas: [], factors: [], sessions: [], has_references: false });
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

  it("uses named inputs in the synchronized expression editor", async () => {
    setup();
    render(<FormulaEditorPage />);
    await waitFor(() => expect(getPrimitives).toHaveBeenCalled());
    fireEvent.click(screen.getByRole("tab", { name: "Expression" }));
    const expression = screen.getByLabelText("Formula expression");
    fireEvent.change(expression, { target: { value: "ts_mean($price, 10)" } });
    expect(expression).toHaveValue("ts_mean($price, 10)");
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
    listFactors.mockResolvedValue([{ id: "factor-1", name: "Momentum winner", saved_at: "2026-06-21", tree: { name: "rank", children: [{ name: "close" }] }, expanded_tree: { name: "rank", children: [{ name: "close" }] }, metrics: { oos_ic: 0.08 }, provenance: { universe: "sp500-lite" }, required_operators: [], notes: "Stable momentum candidate.", disclaimer: "Research only" }]);
    render(<FormulaEditorPage />);
    const library = await screen.findByTestId("formula-library");
    expect(within(library).getByText("Momentum winner")).toBeInTheDocument();
    expect(within(library).getByText("Stable momentum candidate.")).toBeInTheDocument();
  });

  it("saves a new visual formula with named input metadata", async () => {
    setup();
    render(<FormulaEditorPage />);
    await waitFor(() => expect(getPrimitives).toHaveBeenCalled());
    await waitFor(() => expect(screen.getAllByText("Cross-sectional rank").length).toBeGreaterThan(1));
    fireEvent.click(screen.getByRole("button", { name: "Save" }));
    await waitFor(() => expect(addFormula).toHaveBeenCalled());
    expect(addFormula.mock.calls[0][0]).toMatchObject({
      name: "my_formula",
      inputs: [{ name: "price", type: "series" }],
      body: { name: "rank", children: [{ name: "$arg", value: 0 }] },
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
    fireEvent.click(screen.getByRole("button", { name: "Save" }));
    await waitFor(() => expect(addFormula).toHaveBeenCalled());
    expect(addFormula.mock.calls[0][0]).toMatchObject({
      body: { name: "rank", children: [{ name: "$arg", value: 0 }] },
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
    fireEvent.click(screen.getByRole("button", { name: "Open formula" }));
    fireEvent.click(screen.getByRole("tab", { name: "Expression" }));
    fireEvent.change(screen.getByLabelText("Formula expression"), { target: { value: "ts_mean($price, 10)" } });
    fireEvent.click(screen.getByRole("button", { name: "Save" }));
    expect(await screen.findByRole("dialog", { name: "This calculation is already in use" })).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Upgrade formula references" }));
    await waitFor(() => expect(updateFormula).toHaveBeenCalledWith("shared_alpha", expect.anything(), "upgrade_references"));
  });
});
