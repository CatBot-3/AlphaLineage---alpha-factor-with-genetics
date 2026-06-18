// Turn universe-updater rows into a point-in-time universe payload, and back.

import type { UniverseInfo, UniverseSpec } from "../api/types";

export interface UniverseRow {
  symbol: string;
  entry: string;
  exit: string; // empty string = still active
}

export function toUniversePayload(name: string, rows: UniverseRow[]): UniverseSpec {
  const memberships = rows
    .filter((row) => row.symbol.trim() !== "" && row.entry.trim() !== "")
    .map((row) => ({
      symbol: row.symbol.trim().toUpperCase(),
      entry: row.entry,
      exit: row.exit.trim() ? row.exit : null,
    }));
  return { name: name.trim(), memberships };
}

export function rowsFromUniverse(universe: UniverseInfo): UniverseRow[] {
  const rows = universe.memberships.map((membership) => ({
    symbol: membership.symbol,
    entry: membership.entry,
    exit: membership.exit ?? "",
  }));
  return rows.length > 0 ? rows : [{ symbol: "", entry: "", exit: "" }];
}

export function uniqueSymbols(rows: UniverseRow[]): string[] {
  return Array.from(new Set(rows.map((row) => row.symbol.trim().toUpperCase()).filter(Boolean)));
}
