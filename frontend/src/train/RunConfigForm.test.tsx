import { fireEvent, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import type { RunRequestForm } from "./RunConfigForm";

vi.mock("../api/client", () => ({
  listUniverses: () => Promise.resolve([]),
  listFactors: () => Promise.resolve([]),
  getPrimitives: () =>
    Promise.resolve([
      { name: "ts_mean", kind: "operator", arg_types: ["series", "window"], out_type: "series", user: false, category: "time_series" },
      { name: "gt", kind: "operator", arg_types: ["series", "series"], out_type: "bool", user: false, category: "condition" },
      { name: "close", kind: "operand", arg_types: [], out_type: "series", user: false, category: "data" },
    ]),
}));

import { RunConfigForm } from "./RunConfigForm";

afterEach(() => vi.clearAllMocks());

describe("RunConfigForm function space", () => {
  it("lists operator categories and fires the Edit-functions callback", async () => {
    const onOpenFormulaEditor = vi.fn();
    render(<RunConfigForm onStart={vi.fn()} onOpenFormulaEditor={onOpenFormulaEditor} />);

    expect(await screen.findByTestId("function-cat-time_series")).toBeInTheDocument();
    expect(screen.getByTestId("function-cat-condition")).toBeInTheDocument();
    // operands (data) are not operators, so they don't appear as a toggle
    expect(screen.queryByTestId("function-cat-data")).not.toBeInTheDocument();

    fireEvent.click(screen.getByTestId("edit-functions"));
    expect(onOpenFormulaEditor).toHaveBeenCalled();
  });

  it("excludes the condition category from enabled_categories by default", async () => {
    const onStart = vi.fn();
    render(<RunConfigForm onStart={onStart} />);
    await screen.findByTestId("function-cat-condition");

    fireEvent.submit(screen.getByTestId("run-config-form"));
    const req = onStart.mock.calls[0][0] as RunRequestForm;
    expect(req.config.enabled_categories).toContain("time_series");
    expect(req.config.enabled_categories).not.toContain("condition");
  });

  it("includes condition once the user enables it", async () => {
    const onStart = vi.fn();
    render(<RunConfigForm onStart={onStart} />);
    await screen.findByTestId("function-cat-condition");

    fireEvent.click(screen.getByLabelText("enable condition"));
    fireEvent.submit(screen.getByTestId("run-config-form"));
    const req = onStart.mock.calls[0][0] as RunRequestForm;
    expect(req.config.enabled_categories).toContain("condition");
  });
});
