import { describe, expect, it, vi } from "vitest";
import { fireEvent, render, screen, within } from "@testing-library/react";
import type { AgentTurn, Conversation } from "../api/types";
import { ConversationView } from "./ConversationView";

function assistant(overrides: Partial<AgentTurn> = {}): AgentTurn {
  return {
    role: "assistant",
    text: "It leans on realized volatility.",
    at: "2026-01-02T00:00:00Z",
    calls: [],
    labels: [],
    proposals: [],
    usage: {},
    stop_reason: "finished",
    error: "",
    ...overrides,
  };
}

function conversation(turns: AgentTurn[]): Conversation {
  return {
    session_id: "sess-1",
    session_name: "momentum-hunt",
    created_at: "2026-01-01T00:00:00Z",
    updated_at: "2026-01-02T00:00:00Z",
    turns,
    summary: "",
    summarised_through: 0,
    disclaimer: "Not investment advice.",
  };
}

const noop = () => {};

describe("ConversationView", () => {
  it("names the session in the empty state so it is obvious what the agent is looking at", () => {
    render(
      <ConversationView
        conversation={conversation([])}
        sessionName="momentum-hunt"
        onPromote={noop}
        onApplyConfig={noop}
      />,
    );
    expect(screen.getByText("momentum-hunt")).toBeInTheDocument();
  });

  it("keeps the work collapsed until asked, then shows every call", () => {
    render(
      <ConversationView
        conversation={conversation([
          assistant({
            calls: [
              {
                tool: "get_current_factor",
                arguments: {},
                result: { expression: "rank(x)" },
                safety: "read",
                ok: true,
              },
              {
                tool: "evaluate_expression",
                arguments: { expression: "rank(volume)" },
                result: { inner_holdout: { oriented_ic: 0.031 }, generalization_gap: 0.012 },
                safety: "evaluate",
                ok: true,
              },
            ],
          }),
        ])}
        onPromote={noop}
        onApplyConfig={noop}
      />,
    );

    expect(screen.queryByTestId("tool-calls")).not.toBeInTheDocument();
    fireEvent.click(screen.getByTestId("toggle-work"));
    expect(screen.getAllByTestId("tool-call")).toHaveLength(2);
  });

  // The count of *evaluations* is the number that costs something, so it is surfaced separately
  // from the number of steps rather than buried in the expanded view.
  it("separates evaluations from plain reads in the work summary", () => {
    render(
      <ConversationView
        conversation={conversation([
          assistant({
            calls: [
              { tool: "list_operators", arguments: {}, result: {}, safety: "read", ok: true },
              {
                tool: "evaluate_expression",
                arguments: { expression: "a" },
                result: { inner_holdout: { oriented_ic: 0.02 } },
                safety: "evaluate",
                ok: true,
              },
            ],
          }),
        ])}
        onPromote={noop}
        onApplyConfig={noop}
      />,
    );
    expect(screen.getByTestId("toggle-work")).toHaveTextContent("Worked for 2 steps");
    expect(screen.getByTestId("toggle-work")).toHaveTextContent("1 evaluation");
  });

  it("shows the inner-holdout IC on the evaluation line without opening the raw result", () => {
    render(
      <ConversationView
        conversation={conversation([
          assistant({
            calls: [
              {
                tool: "evaluate_expression",
                arguments: { expression: "rank(volume)" },
                result: {
                  inner_holdout: { oriented_ic: 0.0312 },
                  generalization_gap: 0.0125,
                  correlation_with_current_factor: 0.4,
                },
                safety: "evaluate",
                ok: true,
              },
            ],
          }),
        ])}
        onPromote={noop}
        onApplyConfig={noop}
      />,
    );
    fireEvent.click(screen.getByTestId("toggle-work"));
    const call = screen.getByTestId("tool-call");
    expect(within(call).getByText(/inner-holdout IC 0\.0312/)).toBeInTheDocument();
    expect(within(call).getByText(/gap 0\.0125/)).toBeInTheDocument();
  });

  it("marks a failed call so a wrong answer can be traced to it", () => {
    render(
      <ConversationView
        conversation={conversation([
          assistant({
            calls: [
              {
                tool: "validate_expression",
                arguments: { expression: "rank(" },
                result: { error: "unbalanced parenthesis" },
                safety: "read",
                ok: false,
              },
            ],
          }),
        ])}
        onPromote={noop}
        onApplyConfig={noop}
      />,
    );
    fireEvent.click(screen.getByTestId("toggle-work"));
    expect(screen.getByTestId("tool-call")).toHaveAttribute("data-ok", "false");
    expect(screen.getByText("unbalanced parenthesis")).toBeInTheDocument();
  });

  // Labels come from the model, not from a fixed dictionary, so the view must render whatever
  // names it invents rather than mapping them onto a known set.
  it("renders model-authored labels verbatim", () => {
    render(
      <ConversationView
        conversation={conversation([
          assistant({
            labels: [
              {
                name: "liquidity-conditioned reversal",
                evidence: "ts_rank(volume, 20)",
                reading: "It fades moves that happen on thin volume.",
                risk: "Crowded in small caps.",
              },
            ],
          }),
        ])}
        onPromote={noop}
        onApplyConfig={noop}
      />,
    );
    const labels = screen.getByTestId("agent-labels");
    expect(within(labels).getByText("liquidity-conditioned reversal")).toBeInTheDocument();
    expect(within(labels).getByText("ts_rank(volume, 20)")).toBeInTheDocument();
    expect(within(labels).getByText(/Crowded in small caps/)).toBeInTheDocument();
  });

  it("routes a factor proposal to promotion and a config proposal to apply", () => {
    const onPromote = vi.fn();
    const onApplyConfig = vi.fn();
    render(
      <ConversationView
        conversation={conversation([
          assistant({
            proposals: [
              {
                id: "p1",
                kind: "factor",
                rationale: "Adds a volume term.",
                payload: { name: "vol-tilt", expression: "rank(volume)" },
                applied: false,
              },
              {
                id: "p2",
                kind: "config",
                rationale: "More diversity.",
                payload: { diff: [{ key: "population_size", from: 200, to: 400 }] },
                applied: false,
              },
            ],
          }),
        ])}
        onPromote={onPromote}
        onApplyConfig={onApplyConfig}
      />,
    );

    fireEvent.click(screen.getByTestId("promote"));
    fireEvent.click(screen.getByTestId("apply-config"));
    expect(onPromote).toHaveBeenCalledWith("p1");
    expect(onApplyConfig).toHaveBeenCalledWith("p2");
  });

  it("says the turn ran out of budget rather than silently truncating", () => {
    render(
      <ConversationView
        conversation={conversation([assistant({ stop_reason: "budget_exhausted" })])}
        onPromote={noop}
        onApplyConfig={noop}
      />,
    );
    expect(screen.getByText(/ran out of budget/)).toBeInTheDocument();
  });

  it("echoes the pending message and a working indicator before the reply lands", () => {
    render(
      <ConversationView
        conversation={conversation([])}
        pending="What does this measure?"
        busy
        onPromote={noop}
        onApplyConfig={noop}
      />,
    );
    expect(screen.getByText("What does this measure?")).toBeInTheDocument();
    expect(screen.getByRole("status")).toHaveTextContent("Working");
  });
});
