import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { UniverseBuilder } from "./UniverseBuilder";

describe("UniverseBuilder (P7-T1)", () => {
  it("renders the point-in-time form and adds rows", () => {
    render(<UniverseBuilder />);
    expect(screen.getByTestId("universe-builder")).toBeInTheDocument();
    expect(screen.getByLabelText("symbol-0")).toBeInTheDocument();
    expect(screen.getByLabelText("entry-0")).toBeInTheDocument();
    expect(screen.getByLabelText("exit-0")).toBeInTheDocument();

    fireEvent.click(screen.getByText("+ Add symbol"));
    expect(screen.getByLabelText("symbol-1")).toBeInTheDocument();
  });
});
