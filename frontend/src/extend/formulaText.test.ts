import { describe, expect, it } from "vitest";
import type { FactorNode, PrimitiveInfo } from "../api/types";
import { parseFormula, serializeFormula } from "./formulaText";

const PRIMS: PrimitiveInfo[] = [
  { name: "sub", kind: "operator", arg_types: ["series", "series"], out_type: "series", user: false },
  { name: "ts_mean", kind: "operator", arg_types: ["series", "window"], out_type: "series", user: false },
  { name: "mul_scalar", kind: "operator", arg_types: ["series", "scalar"], out_type: "series", user: false },
  { name: "close", kind: "operand", arg_types: [], out_type: "series", user: false },
];

const MACD: FactorNode = {
  name: "sub",
  children: [
    { name: "ts_mean", children: [{ name: "$arg", value: 0 }, { name: "window", value: 12 }] },
    { name: "ts_mean", children: [{ name: "$arg", value: 0 }, { name: "window", value: 26 }] },
  ],
};

describe("formulaText", () => {
  it("round-trips a composed body through serialize -> parse", () => {
    const text = serializeFormula(MACD);
    expect(text).toBe("sub(ts_mean($0, 12), ts_mean($0, 26))");
    const { tree, errors } = parseFormula(text, ["series", "window"], PRIMS);
    expect(errors).toEqual([]);
    expect(tree).toEqual(MACD);
  });

  it("types a bare number by its parent slot (window vs scalar)", () => {
    const win = parseFormula("ts_mean(close, 20)", [], PRIMS);
    expect(win.tree?.children?.[1]).toEqual({ name: "window", value: 20 });
    const sca = parseFormula("mul_scalar(close, 2)", [], PRIMS);
    expect(sca.tree?.children?.[1]).toEqual({ name: "const", value: 2 });
  });

  it("rejects an unknown identifier with a position", () => {
    const { tree, errors } = parseFormula("bogus(close)", [], PRIMS);
    expect(tree).toBeUndefined();
    expect(errors[0].msg).toMatch(/unknown function/);
    expect(errors[0].pos).toBe(0);
  });

  it("rejects an out-of-range argument and arity mismatch", () => {
    expect(parseFormula("ts_mean($5, 10)", ["series"], PRIMS).errors[0].msg).toMatch(/out of range/);
    expect(parseFormula("ts_mean(close)", [], PRIMS).errors[0].msg).toMatch(/expects 2 args/);
  });

  it("rejects a non-integer window", () => {
    expect(parseFormula("ts_mean(close, 2.5)", [], PRIMS).errors[0].msg).toMatch(/whole number/);
  });
});
