import { describe, expect, it, vi } from "vitest";
import { fireEvent, render, screen } from "@testing-library/react";
import type { AgentProposal } from "../api/types";
import { ProposalCard } from "./ProposalCard";

function factorProposal(overrides: Partial<AgentProposal> = {}): AgentProposal {
  return {
    id: "p1",
    kind: "factor",
    rationale: "Conditions the reversal on volume.",
    payload: { name: "vol-tilt", expression: "rank(ts_mean(volume, 20))" },
    applied: false,
    ...overrides,
  };
}

describe("ProposalCard", () => {
  it("shows the expression so the user approves a formula, not a description of one", () => {
    render(<ProposalCard proposal={factorProposal()} onPromote={vi.fn()} onApplyConfig={vi.fn()} />);
    expect(screen.getByText("rank(ts_mean(volume, 20))")).toBeInTheDocument();
    expect(screen.getByText("Conditions the reversal on volume.")).toBeInTheDocument();
  });

  // The two surprising consequences of promotion are that real validation runs and that the
  // agent's spend is folded into the session's trial count. Both belong next to the button.
  it("states that promotion runs real validation and costs trials", () => {
    render(<ProposalCard proposal={factorProposal()} onPromote={vi.fn()} onApplyConfig={vi.fn()} />);
    const hint = screen.getByText(/real validation pass/);
    expect(hint).toHaveTextContent(/inner holdout/);
    expect(hint).toHaveTextContent(/trial count/);
  });

  it("calls back on promote", () => {
    const onPromote = vi.fn();
    render(<ProposalCard proposal={factorProposal()} onPromote={onPromote} onApplyConfig={vi.fn()} />);
    fireEvent.click(screen.getByTestId("promote"));
    expect(onPromote).toHaveBeenCalledOnce();
  });

  it("replaces the gate with the resulting round once applied", () => {
    render(
      <ProposalCard
        proposal={factorProposal({ applied: true, round_index: 4 })}
        onPromote={vi.fn()}
        onApplyConfig={vi.fn()}
      />,
    );
    expect(screen.queryByTestId("promote")).not.toBeInTheDocument();
    expect(screen.getByRole("status")).toHaveTextContent("round 4");
  });

  it("renders a config patch as a before/after diff rather than raw JSON", () => {
    render(
      <ProposalCard
        proposal={{
          id: "p2",
          kind: "config",
          rationale: "Diversity collapsed by generation 20.",
          payload: {
            diff: [
              { key: "population_size", from: 200, to: 400 },
              { key: "mutation_rate", from: 0.1, to: 0.2 },
            ],
          },
          applied: false,
        }}
        onPromote={vi.fn()}
        onApplyConfig={vi.fn()}
      />,
    );
    expect(screen.getByText("population_size")).toBeInTheDocument();
    expect(screen.getByText("400")).toBeInTheDocument();
    expect(screen.getByTestId("apply-config")).toHaveTextContent("Apply to the session");
    // Applying a patch changes what the *next* segment does; it must not imply a run started.
    expect(screen.getByText(/Nothing runs yet/)).toBeInTheDocument();
  });
});
