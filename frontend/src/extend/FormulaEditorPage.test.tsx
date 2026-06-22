import { fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

const getPrimitives = vi.fn();
const listFormulas = vi.fn();
const getCategories = vi.fn();
const addFormula = vi.fn();
const updateFormula = vi.fn();
const deleteFormula = vi.fn();
const validateFormula = vi.fn();
const setPrimitiveCategory = vi.fn();
const putCategories = vi.fn();

vi.mock("../api/client", () => ({
  getPrimitives: () => getPrimitives(),
  listFormulas: () => listFormulas(),
  getCategories: () => getCategories(),
  addFormula: (s: unknown) => addFormula(s),
  updateFormula: (n: string, s: unknown) => updateFormula(n, s),
  deleteFormula: (n: string) => deleteFormula(n),
  validateFormula: (s: unknown) => validateFormula(s),
  setPrimitiveCategory: (p: string, c: string) => setPrimitiveCategory(p, c),
  putCategories: (u: unknown) => putCategories(u),
}));

import { FormulaEditorPage } from "./FormulaEditorPage";

const PRIMS = [
  { name: "ts_mean", kind: "operator", arg_types: ["series", "window"], out_type: "series", user: false, category: "time_series" },
  { name: "rank", kind: "operator", arg_types: ["series"], out_type: "signal", user: false, category: "cross_sectional" },
  { name: "close", kind: "operand", arg_types: [], out_type: "series", user: false, category: "data" },
];

function setup() {
  getPrimitives.mockResolvedValue(PRIMS);
  listFormulas.mockResolvedValue([]);
  getCategories.mockResolvedValue({ order: ["data", "time_series", "custom"], overrides: {} });
  addFormula.mockImplementation(async (s) => ({ ...s, registered: true }));
  updateFormula.mockImplementation(async (_n, s) => ({ ...s, registered: true }));
}

afterEach(() => {
  vi.clearAllMocks();
});

describe("FormulaEditorPage", () => {
  it("renders the node display from the default tree", async () => {
    setup();
    render(<FormulaEditorPage />);
    await waitFor(() => expect(getPrimitives).toHaveBeenCalled());
    expect(screen.getByTestId("formula-tree")).toHaveTextContent("rank");
  });

  it("editing a window param updates the serialized text", async () => {
    setup();
    render(<FormulaEditorPage />);
    await waitFor(() => expect(getPrimitives).toHaveBeenCalled());
    // load a tree with a window leaf via text
    fireEvent.change(screen.getByLabelText("Formula text"), {
      target: { value: "ts_mean($0, 12)" },
    });
    const param = await screen.findByLabelText("param-0");
    fireEvent.change(param, { target: { value: "20" } });
    expect(screen.getByLabelText("Formula text")).toHaveValue("ts_mean($0, 20)");
  });

  it("promotes a window constant to an argument", async () => {
    setup();
    render(<FormulaEditorPage />);
    await waitFor(() => expect(getPrimitives).toHaveBeenCalled());
    fireEvent.change(screen.getByLabelText("Formula text"), {
      target: { value: "ts_mean($0, 12)" },
    });
    fireEvent.click(await screen.findByTestId("promote-0"));
    // the window leaf becomes $1 and arg types grow
    expect(screen.getByLabelText("Formula text")).toHaveValue("ts_mean($0, $1)");
    expect(screen.getByLabelText("Formula arg types")).toHaveValue("series, window");
  });

  it("updates an existing formula via PUT and creates a new one via POST", async () => {
    getPrimitives.mockResolvedValue(PRIMS);
    getCategories.mockResolvedValue({ order: ["custom"], overrides: {} });
    listFormulas.mockResolvedValue([
      {
        name: "my_alpha",
        display_name: "my alpha",
        description: "",
        arg_types: ["series"],
        out_type: "signal",
        body: { name: "rank", children: [{ name: "$arg", value: 0 }] },
        category: "custom",
      },
    ]);
    addFormula.mockImplementation(async (s) => ({ ...s, registered: true }));
    updateFormula.mockImplementation(async (_n, s) => ({ ...s, registered: true }));

    render(<FormulaEditorPage />);
    const list = await screen.findByTestId("formula-list");
    fireEvent.click(within(list).getByText("Edit"));

    // editing the loaded formula -> Update -> PUT
    fireEvent.click(screen.getByTestId("save-formula"));
    await waitFor(() => expect(updateFormula).toHaveBeenCalledWith("my_alpha", expect.anything()));

    // branch -> a new name -> Save -> POST
    fireEvent.click(screen.getByTestId("branch-formula"));
    fireEvent.click(screen.getByTestId("save-formula"));
    await waitFor(() => expect(addFormula).toHaveBeenCalled());
  });
});
