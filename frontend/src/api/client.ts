// Typed client for the `app` build: submit a GP run and poll it to completion.

import type { RunResult } from "./types";

const BASE = import.meta.env.VITE_API_BASE ?? "http://localhost:8000";

export interface RunConfig {
  population_size?: number;
  generations?: number;
  seed?: number;
  [key: string]: unknown;
}

interface JobResponse {
  job_id: string;
  status: string;
}

interface RunStatus {
  job_id: string;
  status: string;
  result: RunResult | null;
  error: string | null;
}

export async function submitRun(config: RunConfig = {}): Promise<string> {
  const res = await fetch(`${BASE}/runs`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ config }),
  });
  if (!res.ok) throw new Error(`submit failed: ${res.status}`);
  return ((await res.json()) as JobResponse).job_id;
}

export async function getRun(jobId: string): Promise<RunStatus> {
  const res = await fetch(`${BASE}/runs/${jobId}`);
  if (!res.ok) throw new Error(`status failed: ${res.status}`);
  return (await res.json()) as RunStatus;
}

export async function fetchRun(
  config: RunConfig = { population_size: 60, generations: 8 },
): Promise<RunResult> {
  const jobId = await submitRun(config);
  for (let i = 0; i < 600; i++) {
    const status = await getRun(jobId);
    if (status.status === "done" && status.result) return status.result;
    if (status.status === "failed") throw new Error(status.error ?? "run failed");
    await new Promise((resolve) => setTimeout(resolve, 1000));
  }
  throw new Error("run timed out");
}
