import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import type { FactorNode } from "../api/types";
import { BestFormulaResultPage } from "./BestFormulaResultPage";

vi.mock("../api/client", () => ({
  getPrimitives: () => Promise.resolve([
    {
      name: "ts_mean",
      display_name: "Moving average",
      description: "Rolling mean.",
      kind: "operator",
      arg_types: ["series", "window"],
      inputs: [
        { name: "series", type: "series" },
        { name: "lookback", type: "window", default: 20 },
      ],
      out_type: "series",
      user: false,
    },
    {
      name: "close",
      display_name: "Close",
      kind: "operand",
      arg_types: [],
      out_type: "series",
      user: false,
    },
  ]),
}));

const factor: FactorNode = {
  name: "ts_mean",
  children: [
    { name: "close" },
    { name: "window", value: 20 },
  ],
};

describe("BestFormulaResultPage", () => {
  it("uses the shared read-only formula canvas and exposes explicit snapshot actions", () => {
    const onSave = vi.fn();
    const onOpenCopy = vi.fn();
    render(
      <BestFormulaResultPage
        factor={factor}
        canSave
        saved={false}
        onSave={onSave}
        onOpenCopy={onOpenCopy}
      />,
    );

    expect(screen.getByRole("region", { name: "Best formula result diagram" })).toHaveAttribute(
      "aria-readonly",
      "true",
    );
    expect(screen.queryByRole("spinbutton")).not.toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: "Save" }));
    fireEvent.click(screen.getByRole("button", { name: "Open copy" }));
    expect(onSave).toHaveBeenCalledOnce();
    expect(onOpenCopy).toHaveBeenCalledOnce();

    fireEvent.click(screen.getByRole("tab", { name: "Expression" }));
    expect(screen.getByRole("region", { name: "Formula expression" })).toHaveTextContent(
      "ts_mean(close, 20)",
    );
  });
});
