import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

vi.mock("../api/client", () => ({
  getPrimitives: vi.fn(async () => [
    { name: "sub", kind: "operator", arg_types: ["series", "series"], out_type: "series", user: false },
  ]),
  registerOperator: vi.fn(async () => ({
    name: "x",
    kind: "operator",
    arg_types: [],
    out_type: "series",
    user: true,
  })),
}));

import { OperatorComposer } from "./OperatorComposer";

describe("OperatorComposer (P7-T2)", () => {
  it("renders the palette and signature controls", () => {
    render(<OperatorComposer />);
    expect(screen.getByTestId("operator-composer")).toBeInTheDocument();
    expect(screen.getByTestId("palette")).toBeInTheDocument();
    expect(screen.getByTestId("register-operator")).toBeInTheDocument();
    expect(screen.getByLabelText("arg-types")).toBeInTheDocument();
  });
});
