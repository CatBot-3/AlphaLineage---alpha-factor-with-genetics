// Typed client for the `app` build: submit a GP run and poll it to completion.

import type {
  OperatorSpec,
  PrimitiveInfo,
  RunResult,
  UniverseInfo,
  UniverseSpec,
  WorkspaceSnapshot,
  WorkspaceSummary,
} from "./types";

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

// --- extensibility (Phase 7) ---------------------------------------------------
export async function getPrimitives(): Promise<PrimitiveInfo[]> {
  const res = await fetch(`${BASE}/primitives`);
  if (!res.ok) throw new Error(`primitives failed: ${res.status}`);
  return (await res.json()) as PrimitiveInfo[];
}

export async function registerOperator(spec: OperatorSpec): Promise<PrimitiveInfo> {
  const res = await fetch(`${BASE}/operators`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(spec),
  });
  if (!res.ok) {
    const detail = (await res.json().catch(() => ({}))) as { detail?: string };
    throw new Error(detail.detail ?? `register failed: ${res.status}`);
  }
  return (await res.json()) as PrimitiveInfo;
}

export async function defineUniverse(
  spec: UniverseSpec,
): Promise<{ name: string; symbols: string[] }> {
  const res = await fetch(`${BASE}/universes`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(spec),
  });
  if (!res.ok) throw new Error(`define universe failed: ${res.status}`);
  return (await res.json()) as { name: string; symbols: string[] };
}

export async function listUniverses(): Promise<UniverseInfo[]> {
  const res = await fetch(`${BASE}/universes`);
  if (!res.ok) throw new Error(`universes failed: ${res.status}`);
  return (await res.json()) as UniverseInfo[];
}

export async function listWorkspaces(): Promise<WorkspaceSummary[]> {
  const res = await fetch(`${BASE}/workspaces`);
  if (!res.ok) throw new Error(`workspaces failed: ${res.status}`);
  return (await res.json()) as WorkspaceSummary[];
}

export async function saveWorkspace(snapshot: WorkspaceSnapshot): Promise<WorkspaceSnapshot> {
  const res = await fetch(`${BASE}/workspaces`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(snapshot),
  });
  if (!res.ok) {
    const detail = (await res.json().catch(() => ({}))) as { detail?: string };
    throw new Error(detail.detail ?? `save workspace failed: ${res.status}`);
  }
  return (await res.json()) as WorkspaceSnapshot;
}

export async function getWorkspace(id: string): Promise<WorkspaceSnapshot> {
  const res = await fetch(`${BASE}/workspaces/${encodeURIComponent(id)}`);
  if (!res.ok) throw new Error(`workspace load failed: ${res.status}`);
  return (await res.json()) as WorkspaceSnapshot;
}

export async function deleteWorkspace(id: string): Promise<void> {
  const res = await fetch(`${BASE}/workspaces/${encodeURIComponent(id)}`, { method: "DELETE" });
  if (!res.ok) throw new Error(`workspace delete failed: ${res.status}`);
}
