// Picks the data source by build target: `demo` reads a static JSON snapshot (no backend),
// `app` calls the local FastAPI backend. Default (dev) is the static demo snapshot.

import { fetchRun } from "./client";
import type { RunResult } from "./types";

export async function loadRun(): Promise<RunResult> {
  if (import.meta.env.VITE_DATA_SOURCE === "app") {
    return fetchRun();
  }
  const res = await fetch(`${import.meta.env.BASE_URL}demo-run.json`);
  if (!res.ok) throw new Error("failed to load demo-run.json");
  return (await res.json()) as RunResult;
}
