// Turn universe-updater rows into a point-in-time universe payload.

import type { UniverseSpec } from "../api/types";

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
