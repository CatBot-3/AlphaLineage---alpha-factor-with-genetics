import { act, renderHook } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { ApiError } from "../api/client";
import type { SessionState } from "../api/types";
import { useSession } from "./useSession";

const { getSession, continueSession, createSession } = vi.hoisted(() => ({
  getSession: vi.fn(), continueSession: vi.fn(), createSession: vi.fn(),
}));
vi.mock("../api/client", async (importOriginal) => ({
  ...await importOriginal<typeof import("../api/client")>(),
  getSession, continueSession, createSession, stopSession: vi.fn(),
}));
const snapshot = (status = "running"): SessionState => ({
  id: "overnight", segments: [], cumulative_trials: 21916,
  job: { id: "job", status, progress: {
    generation: 15, target_generations: 50, history: [], best: null,
  } }, result: null,
} as unknown as SessionState);
async function flush() { await act(async () => { await Promise.resolve(); }); }
async function advance(ms: number) { await act(async () => { await vi.advanceTimersByTimeAsync(ms); }); }
beforeEach(() => { vi.useFakeTimers(); vi.resetAllMocks(); });
afterEach(() => { vi.useRealTimers(); });

describe("session connection recovery", () => {
  it("shows a pending continuation immediately and suppresses a repeated click", async () => {
    getSession.mockResolvedValue(snapshot("interrupted"));
    const { result } = renderHook(() => useSession(undefined, "overnight"));
    await flush();
    let finish!: (value: { session_id: string; job_id: string }) => void;
    continueSession.mockImplementation(() => new Promise((resolve) => { finish = resolve; }));
    act(() => { void result.current.cont({ generations: 25 }); });
    expect(result.current.submitting).toBe("continue");
    act(() => { void result.current.cont({ generations: 35 }); });
    expect(continueSession).toHaveBeenCalledTimes(1);
    getSession.mockResolvedValue(snapshot());
    await act(async () => { finish({ session_id: "overnight", job_id: "new-job" }); });
    await flush();
    expect(result.current.submitting).toBeNull();
    expect(result.current.phase).toBe("running");
  });

  it("reconciles an already-running continuation without sending another POST", async () => {
    getSession.mockResolvedValue(snapshot("interrupted"));
    const { result } = renderHook(() => useSession(undefined, "overnight"));
    await flush();
    continueSession.mockRejectedValue(new ApiError("a segment is already running", 409));
    getSession.mockResolvedValue(snapshot());
    await act(async () => { await result.current.cont({ generations: 25 }); });
    await flush();
    expect(result.current.phase).toBe("running");
    expect(continueSession).toHaveBeenCalledTimes(1);
  });

  it("preserves running progress and backs off through 15 seconds, then reconnects", async () => {
    const running = snapshot();
    getSession.mockResolvedValueOnce(running);
    const { result } = renderHook(() => useSession(undefined, "overnight"));
    await flush();
    getSession.mockRejectedValue(new TypeError("Failed to fetch"));
    await advance(1000);
    expect(result.current.connection).toBe("reconnecting");
    expect(result.current.phase).toBe("running");
    expect(result.current.state).toBe(running);
    expect(result.current.error).toBeNull();
    for (const delay of [1000, 2000, 4000, 8000, 15000, 15000]) {
      const calls = getSession.mock.calls.length;
      await advance(delay - 1);
      expect(getSession).toHaveBeenCalledTimes(calls);
      await advance(1);
      expect(getSession).toHaveBeenCalledTimes(calls + 1);
    }
    getSession.mockResolvedValue(running);
    await advance(15000);
    expect(result.current.connection).toBe("connected");
    const calls = getSession.mock.calls.length;
    await advance(1000);
    expect(getSession).toHaveBeenCalledTimes(calls + 1);
    expect(continueSession).not.toHaveBeenCalled();
    expect(createSession).not.toHaveBeenCalled();
  });

  it("shows a confirmed training error and stops status polling", async () => {
    const failed = snapshot("failed");
    failed.job!.error = "checkpoint storage failed";
    getSession.mockResolvedValue(failed);
    const { result } = renderHook(() => useSession(undefined, "overnight"));
    await flush();
    expect(result.current.phase).toBe("failed");
    expect(result.current.error).toBe("checkpoint storage failed");
    await advance(30000);
    expect(getSession).toHaveBeenCalledTimes(1);
  });

  it("keeps interrupted sessions attached and removes genuinely missing sessions", async () => {
    getSession.mockResolvedValue(snapshot("interrupted"));
    const { result } = renderHook(() => useSession(undefined, "overnight"));
    await flush();
    expect(result.current.phase).toBe("interrupted");
    expect(result.current.sessionId).toBe("overnight");
    getSession.mockRejectedValue(new ApiError("unknown session", 404));
    act(() => result.current.refresh());
    await flush();
    expect(result.current.sessionId).toBeNull();
    expect(result.current.phase).toBe("idle");
  });

  it("reconciles a lost continuation response with GET without replaying the POST", async () => {
    getSession.mockResolvedValue(snapshot("interrupted"));
    const { result } = renderHook(() => useSession(undefined, "overnight"));
    await flush();
    continueSession.mockRejectedValue(new TypeError("Failed to fetch"));
    getSession.mockResolvedValue(snapshot());
    await act(async () => { await result.current.cont({ generations: 35 }); });
    await flush();
    expect(continueSession).toHaveBeenCalledTimes(1);
    expect(result.current.phase).toBe("running");
    await advance(30000);
    expect(continueSession).toHaveBeenCalledTimes(1);
  });

  it("a low-memory submission error preserves the interrupted checkpoint state", async () => {
    const interrupted = snapshot("interrupted");
    getSession.mockResolvedValue(interrupted);
    const { result } = renderHook(() => useSession(undefined, "overnight"));
    await flush();
    continueSession.mockRejectedValue(new ApiError("Free memory and retry", 503, { retryable: true }));
    await act(async () => { await result.current.cont({ generations: 35 }); });
    expect(result.current.phase).toBe("interrupted");
    expect(result.current.error).toContain("Free memory");
    expect(result.current.state).toBe(interrupted);
    await advance(30000);
    expect(continueSession).toHaveBeenCalledTimes(1);
  });
});
