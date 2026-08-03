import { describe, expect, it, vi } from "vitest";
import { fireEvent, render, screen } from "@testing-library/react";
import type { AgentToolCatalog } from "../api/types";
import { Composer } from "./Composer";

const catalog: AgentToolCatalog = {
  tools: [
    {
      name: "get_current_factor",
      description: "The session's current best factor.",
      parameters: {},
      safety: "read",
    },
    {
      name: "evaluate_expression",
      description: "Score a candidate on an inner holdout inside the training window.",
      parameters: {},
      safety: "evaluate",
    },
  ],
  tunable_config_keys: ["population_size"],
  protected_config_keys: { horizon: "defines what the metric means", seed: "seed-hacking" },
  defaults: { max_tool_calls: 12, max_evaluations: 8, max_seconds: 300 },
  unavailable_reason: "",
  disclaimer: "Not investment advice.",
};

describe("Composer", () => {
  it("sends the trimmed message with the chosen budget and clears the box", () => {
    const onSend = vi.fn();
    render(<Composer catalog={catalog} onSend={onSend} onClear={vi.fn()} />);

    const input = screen.getByTestId("agent-input");
    fireEvent.change(input, { target: { value: "  try a volume term  " } });
    fireEvent.change(screen.getByTestId("agent-max-evaluations"), { target: { value: "3" } });
    fireEvent.click(screen.getByTestId("agent-send"));

    expect(onSend).toHaveBeenCalledWith("try a volume term", {
      max_tool_calls: 12,
      max_evaluations: 3,
    });
    expect(input).toHaveValue("");
  });

  it("sends on Enter and inserts a newline on Shift+Enter", () => {
    const onSend = vi.fn();
    render(<Composer catalog={catalog} onSend={onSend} onClear={vi.fn()} />);
    const input = screen.getByTestId("agent-input");

    fireEvent.change(input, { target: { value: "first line" } });
    fireEvent.keyDown(input, { key: "Enter", shiftKey: true });
    expect(onSend).not.toHaveBeenCalled();

    fireEvent.keyDown(input, { key: "Enter" });
    expect(onSend).toHaveBeenCalledOnce();
  });

  it("refuses to send an empty message or a second one while the first is running", () => {
    render(<Composer catalog={catalog} onSend={vi.fn()} onClear={vi.fn()} />);
    expect(screen.getByTestId("agent-send")).toBeDisabled();

    fireEvent.change(screen.getByTestId("agent-input"), { target: { value: "hello" } });
    expect(screen.getByTestId("agent-send")).toBeEnabled();
  });

  it("disables sending while a turn is in flight", () => {
    render(<Composer catalog={catalog} busy onSend={vi.fn()} onClear={vi.fn()} />);
    fireEvent.change(screen.getByTestId("agent-input"), { target: { value: "hello" } });
    expect(screen.getByTestId("agent-send")).toBeDisabled();
    expect(screen.getByTestId("agent-send")).toHaveTextContent("Working");
  });

  // The capability list is the honest answer to "what can this thing do to my project?" — it is
  // rendered from the backend registry rather than restated in the UI, so it cannot drift.
  it("discloses every tool with its safety class and names the settings it cannot touch", () => {
    render(<Composer catalog={catalog} onSend={vi.fn()} onClear={vi.fn()} />);
    expect(screen.getByText("What it can do (2 tools)")).toBeInTheDocument();
    expect(screen.getByText("get_current_factor")).toBeInTheDocument();
    expect(screen.getByText("evaluate")).toBeInTheDocument();
    expect(screen.getByText(/Settings it cannot change: horizon, seed/)).toBeInTheDocument();
  });

  it("explains that the split is absent rather than merely forbidden", () => {
    render(<Composer catalog={catalog} onSend={vi.fn()} onClear={vi.fn()} />);
    expect(screen.getByText(/absent from the data it is given, not merely/)).toBeInTheDocument();
  });

  it("offers to clear the thread only once there is something to clear", () => {
    const { rerender } = render(
      <Composer catalog={catalog} onSend={vi.fn()} onClear={vi.fn()} />,
    );
    expect(screen.queryByText("Clear thread")).not.toBeInTheDocument();

    const onClear = vi.fn();
    rerender(<Composer catalog={catalog} hasHistory onSend={vi.fn()} onClear={onClear} />);
    fireEvent.click(screen.getByText("Clear thread"));
    expect(onClear).toHaveBeenCalledOnce();
  });
});
