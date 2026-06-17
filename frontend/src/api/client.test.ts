import { afterEach, describe, expect, it, vi } from "vitest";
import {
  continueSession,
  createSession,
  getSession,
  putSettings,
  saveFactor,
} from "./client";

function mockFetch(payload: unknown, ok = true) {
  const fn = vi.fn(async () => ({
    ok,
    status: ok ? 200 : 400,
    json: async () => payload,
  })) as unknown as typeof fetch;
  globalThis.fetch = fn;
  return fn as unknown as ReturnType<typeof vi.fn>;
}

afterEach(() => {
  vi.restoreAllMocks();
});

describe("session client", () => {
  it("POSTs the form config verbatim to /sessions", async () => {
    const fetchSpy = mockFetch({ session_id: "s1", job_id: "j1" });
    const handle = await createSession({
      name: "run",
      universe: "tech",
      config: { population_size: 42, generations: 7 },
      seed_factor_ids: ["f1"],
    });

    expect(handle).toEqual({ session_id: "s1", job_id: "j1" });
    const [url, init] = fetchSpy.mock.calls[0];
    expect(String(url)).toMatch(/\/sessions$/);
    expect(init.method).toBe("POST");
    const body = JSON.parse(init.body as string);
    expect(body.config.population_size).toBe(42);
    expect(body.seed_factor_ids).toEqual(["f1"]);
  });

  it("POSTs overrides to the continue endpoint", async () => {
    const fetchSpy = mockFetch({ session_id: "s1", job_id: "j2" });
    await continueSession("s1", { generations: 3, universe: "other" });

    const [url, init] = fetchSpy.mock.calls[0];
    expect(String(url)).toMatch(/\/sessions\/s1\/continue$/);
    const body = JSON.parse(init.body as string);
    expect(body).toEqual({ generations: 3, universe: "other" });
  });

  it("parses session state including the locked boundary", async () => {
    mockFetch({
      id: "s1",
      boundaries: { test_start: "2025-01-01", train_end: "2024-01-01" },
      cumulative_trials: 100,
      test_reads: 2,
      segments: [],
    });
    const state = await getSession("s1");
    expect(state.boundaries.test_start).toBe("2025-01-01");
    expect(state.test_reads).toBe(2);
  });

  it("surfaces backend error detail", async () => {
    mockFetch({ detail: "a segment is already running" }, false);
    await expect(continueSession("s1", { generations: 1 })).rejects.toThrow(
      "a segment is already running",
    );
  });
});

describe("factor + settings client", () => {
  it("POSTs the exact tree and provenance to /factors", async () => {
    const fetchSpy = mockFetch({ id: "f1", name: "f", disclaimer: "x" });
    const tree = { name: "rank", children: [{ name: "close" }] };
    await saveFactor({ name: "f", tree, provenance: { session_id: "s1" } });

    const [, init] = fetchSpy.mock.calls[0];
    const body = JSON.parse(init.body as string);
    expect(body.tree).toEqual(tree);
    expect(body.provenance).toEqual({ session_id: "s1" });
  });

  it("PUTs the factors_dir and returns the resolved path", async () => {
    const fetchSpy = mockFetch({ factors_dir: "/tmp/factors" });
    const settings = await putSettings("/tmp/factors");
    expect(settings.factors_dir).toBe("/tmp/factors");
    const [, init] = fetchSpy.mock.calls[0];
    expect(init.method).toBe("PUT");
    expect(JSON.parse(init.body as string)).toEqual({ factors_dir: "/tmp/factors" });
  });
});
