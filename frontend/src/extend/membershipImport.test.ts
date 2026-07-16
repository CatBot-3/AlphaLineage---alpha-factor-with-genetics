import { describe, expect, it } from "vitest";
import { parseMembershipImport } from "./membershipImport";

describe("membership CSV/TSV import", () => {
  it("parses a header and normalizes CSV symbols", () => {
    expect(parseMembershipImport([
      "symbol,entry,exit",
      "aapl,2000-01-03,",
      "LEH,2000-01-03,2008-09-15",
    ].join("\n"))).toMatchObject({
      headerSkipped: true,
      errors: [],
      rows: [
        { symbol: "AAPL", entry: "2000-01-03", exit: "" },
        { symbol: "LEH", entry: "2000-01-03", exit: "2008-09-15" },
      ],
    });
  });

  it("accepts TSV and non-overlapping re-entry intervals", () => {
    const parsed = parseMembershipImport([
      "IBM\t2000-01-01\t2010-01-01",
      "IBM\t2011-01-01\t",
    ].join("\n"));
    expect(parsed.rows).toHaveLength(2);
    expect(parsed.errors).toHaveLength(0);
  });

  it("reports invalid dates and overlapping intervals without fabricating rows", () => {
    const parsed = parseMembershipImport([
      "AAPL,not-a-date,",
      "MSFT,2020-01-01,2022-01-01",
      "MSFT,2021-01-01,",
    ].join("\n"));
    expect(parsed.rows).toEqual([{ symbol: "MSFT", entry: "2020-01-01", exit: "2022-01-01" }]);
    expect(parsed.errors.map((error) => error.message)).toEqual([
      "Entry must be a valid YYYY-MM-DD date.",
      "Overlapping membership interval for MSFT.",
    ]);
  });
});
