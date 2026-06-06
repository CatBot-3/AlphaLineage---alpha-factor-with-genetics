import { render, screen } from "@testing-library/react";
import { ReactFlowProvider } from "@xyflow/react";
import { describe, expect, it } from "vitest";
import type { FactorNode } from "../api/types";
import { FactorTree } from "./FactorTree";
import { NodeCard } from "./NodeCard";
import { treeToFlow } from "./treeToFlow";

const factor: FactorNode = {
  name: "rank",
  children: [{ name: "ts_mean", children: [{ name: "returns" }, { name: "window", value: 20 }] }],
};

describe("factor tree (P6-T1)", () => {
  it("transforms a stored factor into a structurally correct tree", () => {
    const { nodes, edges } = treeToFlow(factor);
    expect(nodes).toHaveLength(4); // one node per primitive
    expect(edges).toHaveLength(3); // nodes - 1
    expect(nodes.map((n) => n.data.name).sort()).toEqual([
      "rank",
      "returns",
      "ts_mean",
      "window",
    ]);
  });

  it("renders a primitive node with its name and parameter", () => {
    render(
      <ReactFlowProvider>
        {/* eslint-disable-next-line @typescript-eslint/no-explicit-any */}
        <NodeCard {...({ data: { name: "window", value: 20 } } as any)} />
      </ReactFlowProvider>,
    );
    const node = screen.getByTestId("factor-node");
    expect(node).toHaveTextContent("window");
    expect(node).toHaveTextContent("20");
  });

  it("mounts the full tree view", () => {
    render(<FactorTree factor={factor} />);
    expect(screen.getByTestId("factor-tree")).toBeInTheDocument();
  });
});
