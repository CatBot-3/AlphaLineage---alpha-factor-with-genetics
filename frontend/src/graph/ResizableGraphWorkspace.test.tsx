import { fireEvent, render, screen } from "@testing-library/react";
import { beforeEach, describe, expect, it } from "vitest";
import { ResizableGraphWorkspace } from "./ResizableGraphWorkspace";

beforeEach(() => window.localStorage.clear());

describe("ResizableGraphWorkspace", () => {
  it("resizes both side panes with keyboard-accessible physical edge movement", () => {
    render(
      <ResizableGraphWorkspace
        storageKey="graph-test"
        left={{ id: "left-pane", label: "Generations", content: <p>Left</p> }}
        center={{ id: "center-pane", label: "Graph", content: <p>Center</p> }}
        right={{ id: "right-pane", label: "Inspector", content: <p>Right</p> }}
      />,
    );

    const left = screen.getByRole("separator", { name: "Resize Generations" });
    const right = screen.getByRole("separator", { name: "Resize Inspector" });
    expect(left).toHaveAttribute("aria-valuenow", "280");
    expect(right).toHaveAttribute("aria-valuenow", "340");

    fireEvent.keyDown(left, { key: "ArrowRight" });
    fireEvent.keyDown(right, { key: "ArrowLeft" });

    expect(left).toHaveAttribute("aria-valuenow", "304");
    expect(right).toHaveAttribute("aria-valuenow", "364");
    expect(window.localStorage.getItem("graph-test:left")).toBe("304");
    expect(window.localStorage.getItem("graph-test:right")).toBe("364");

    fireEvent.keyDown(left, { key: "Enter" });
    expect(left).toHaveAttribute("aria-valuenow", "0");
    expect(screen.getByLabelText("Generations")).toHaveAttribute("aria-hidden", "true");

    fireEvent.keyDown(left, { key: "ArrowRight" });
    expect(left).toHaveAttribute("aria-valuenow", "328");
    expect(screen.getByLabelText("Generations")).toHaveAttribute("aria-hidden", "false");
  });
});
