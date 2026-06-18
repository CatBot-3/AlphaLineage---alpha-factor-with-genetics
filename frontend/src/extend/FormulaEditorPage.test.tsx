import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

const addFormula = vi.fn();
const deleteFormula = vi.fn();
const listFormulas = vi.fn();
const getPrimitives = vi.fn();

vi.mock("../api/client", () => ({
  addFormula: (payload: unknown) => addFormula(payload),
  deleteFormula: (name: string) => deleteFormula(name),
  listFormulas: () => listFormulas(),
  getPrimitives: () => getPrimitives(),
}));

import { FormulaEditorPage } from "./FormulaEditorPage";

afterEach(() => {
  vi.clearAllMocks();
});

describe("FormulaEditorPage", () => {
  it("saves a formula template through the formula API without needing to expand a section first", async () => {
    listFormulas.mockResolvedValue([]);
    getPrimitives.mockResolvedValue([]);
    addFormula.mockImplementation(async (payload) => ({ ...payload, registered: true, error: null }));

    render(<FormulaEditorPage />);

    fireEvent.click(screen.getByText("Macd-style spread"));
    fireEvent.change(screen.getByLabelText("Formula name"), { target: { value: "macd_signal" } });
    fireEvent.click(screen.getByTestId("save-formula"));

    await waitFor(() =>
      expect(addFormula).toHaveBeenCalledWith(
        expect.objectContaining({
          name: "macd_signal",
          arg_types: ["series"],
          out_type: "series",
          body: expect.objectContaining({ name: "sub" }),
        }),
      ),
    );
  });

  it("also renders the operator composer on the same page", async () => {
    listFormulas.mockResolvedValue([]);
    getPrimitives.mockResolvedValue([]);
    render(<FormulaEditorPage />);
    await waitFor(() => expect(getPrimitives).toHaveBeenCalled());
    expect(screen.getByTestId("operator-composer")).toBeInTheDocument();
  });
});
