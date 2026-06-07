import { describe, expect, it } from "vitest";
import { toUniversePayload } from "./toUniversePayload";

describe("toUniversePayload (P7-T1)", () => {
  it("builds a point-in-time payload, dropping empty rows and upper-casing symbols", () => {
    const spec = toUniversePayload("u", [
      { symbol: "aapl", entry: "2020-01-01", exit: "" },
      { symbol: "leh", entry: "2000-01-01", exit: "2008-09-15" },
      { symbol: "", entry: "", exit: "" },
    ]);
    expect(spec).toEqual({
      name: "u",
      memberships: [
        { symbol: "AAPL", entry: "2020-01-01", exit: null },
        { symbol: "LEH", entry: "2000-01-01", exit: "2008-09-15" },
      ],
    });
  });
});
