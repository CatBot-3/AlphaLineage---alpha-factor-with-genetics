import type { UniverseRow } from "./toUniversePayload";

export interface MembershipImportError {
  line: number;
  message: string;
  value: string;
}

export interface MembershipImportPreview {
  rows: UniverseRow[];
  errors: MembershipImportError[];
  headerSkipped: boolean;
}

const SYMBOL_PATTERN = /^[A-Z0-9^][A-Z0-9._^=-]{0,63}$/;

function unquote(value: string): string {
  const trimmed = value.trim();
  return trimmed.length >= 2 && trimmed.startsWith('"') && trimmed.endsWith('"')
    ? trimmed.slice(1, -1).replace(/""/g, '"').trim()
    : trimmed;
}

function validIsoDate(value: string): boolean {
  if (!/^\d{4}-\d{2}-\d{2}$/.test(value)) return false;
  const parsed = new Date(`${value}T00:00:00Z`);
  return !Number.isNaN(parsed.valueOf()) && parsed.toISOString().slice(0, 10) === value;
}

function columns(line: string): string[] {
  const delimiter = line.includes("\t") ? "\t" : ",";
  return line.split(delimiter).map(unquote);
}

export function parseMembershipImport(text: string): MembershipImportPreview {
  const rows: UniverseRow[] = [];
  const errors: MembershipImportError[] = [];
  let headerSkipped = false;
  const intervals = new Map<string, Array<{ entry: string; exit: string }>>();

  for (const [index, raw] of text.replace(/^\uFEFF/, "").split(/\r?\n/).entries()) {
    const line = raw.trim();
    if (!line) continue;
    const lineNumber = index + 1;
    const parts = columns(line);
    const first = parts[0]?.toLowerCase();
    if (rows.length === 0 && errors.length === 0 && (first === "symbol" || first === "ticker")) {
      headerSkipped = true;
      continue;
    }
    if (parts.length < 2 || parts.length > 3) {
      errors.push({ line: lineNumber, message: "Expected symbol, entry, and optional exit.", value: raw });
      continue;
    }

    const symbol = parts[0].toUpperCase();
    const entry = parts[1];
    const exit = parts[2] ?? "";
    if (!SYMBOL_PATTERN.test(symbol)) {
      errors.push({ line: lineNumber, message: "Invalid market symbol.", value: raw });
      continue;
    }
    if (!validIsoDate(entry)) {
      errors.push({ line: lineNumber, message: "Entry must be a valid YYYY-MM-DD date.", value: raw });
      continue;
    }
    if (exit && !validIsoDate(exit)) {
      errors.push({ line: lineNumber, message: "Exit must be blank or a valid YYYY-MM-DD date.", value: raw });
      continue;
    }
    if (exit && exit <= entry) {
      errors.push({ line: lineNumber, message: "Exit must be after entry.", value: raw });
      continue;
    }

    const existing = intervals.get(symbol) ?? [];
    const overlaps = existing.some((interval) => (
      (!interval.exit || entry < interval.exit) && (!exit || interval.entry < exit)
    ));
    if (overlaps) {
      errors.push({ line: lineNumber, message: `Overlapping membership interval for ${symbol}.`, value: raw });
      continue;
    }
    existing.push({ entry, exit });
    intervals.set(symbol, existing);
    rows.push({ symbol, entry, exit });
  }

  return { rows, errors, headerSkipped };
}
