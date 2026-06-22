import { describe, expect, it } from "vitest";
import type { FactorNode } from "../api/types";
import { paramLeaves, promoteToArg, setLeafValue } from "./treeEdit";

const MACD: FactorNode = {
  name: "mul_scalar",
  children: [
    {
      name: "sub",
      children: [
        { name: "ts_mean", children: [{ name: "$arg", value: 0 }, { name: "window", value: 12 }] },
        { name: "ts_mean", children: [{ name: "$arg", value: 0 }, { name: "window", value: 26 }] },
      ],
    },
    { name: "const", value: 2 },
  ],
};

describe("treeEdit", () => {
  it("collects every window/const/$arg leaf with its path", () => {
    const leaves = paramLeaves(MACD);
    const names = leaves.map((l) => l.node.name).sort();
    expect(names).toEqual(["$arg", "$arg", "const", "window", "window"]);
    // the scalar 2 lives at [1]
    expect(leaves.find((l) => l.node.name === "const")?.path).toEqual([1]);
  });

  it("sets a leaf value immutably", () => {
    const next = setLeafValue(MACD, [0, 0, 1], 9); // first ts_mean's window 12 -> 9
    expect(next.children?.[0].children?.[0].children?.[1]).toEqual({ name: "window", value: 9 });
    expect(MACD.children?.[0].children?.[0].children?.[1]).toEqual({ name: "window", value: 12 }); // original untouched
  });

  it("promotes a constant leaf to an argument placeholder", () => {
    const next = promoteToArg(MACD, [1], 1); // const 2 -> $arg 1
    expect(next.children?.[1]).toEqual({ name: "$arg", value: 1 });
  });
});
