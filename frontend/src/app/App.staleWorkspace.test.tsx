/**
 * A workspace saved by an older build must not be able to blank the app.
 *
 * This reproduces a real failure: the "Explain" tab was renamed to "Agent", and anyone whose
 * localStorage still said `selectedTab: "explain"` got a white screen on next launch. The restored
 * value flowed straight into `PAGE_COPY[tab].title`, which threw during App's own render — above
 * the ErrorBoundary, so there was not even an error to read. The server logged four clean 200s.
 *
 * The general point is that persisted UI state is untrusted input from a previous version of the
 * program. TypeScript's belief that `ui.selectedTab` is a `Tab` stops at the JSON boundary.
 */
import { render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { WORKSPACE_KEY } from "./workspace";

vi.mock("./mode", () => ({
  getAppMode: () => "app",
  modeLabel: () => "Local Backend",
}));

/**
 * The whole HTTP surface, stubbed at `fetch` rather than at the client module.
 *
 * Mocking `../api/client` would mean enumerating every export App's tree happens to touch, and
 * the test would then fail for a missing stub rather than for the thing it is checking. Stubbing
 * the transport keeps the real client in play, so a render crash here is the app's own.
 */
beforeEach(() => {
  globalThis.fetch = vi.fn(async (input: RequestInfo | URL) => {
    const url = String(typeof input === "string" ? input : (input as Request).url ?? input);
    const list = /\/(sessions|formulas|primitives|universes|factors|workspaces|finalizations|rounds)(\?|$)/;
    const body: unknown = list.test(url) ? [] : {};
    return {
      ok: true,
      status: 200,
      headers: new Headers({ "content-type": "application/json" }),
      json: async () => body,
      text: async () => JSON.stringify(body),
    } as unknown as Response;
  }) as unknown as typeof fetch;
});

/**
 * `readLocalWorkspace` rejects anything without `version: 1` and a string `name`, so both are
 * required for this fixture to survive the read and actually reach the tab logic. It notably does
 * *not* validate `ui.selectedTab` — which is the hole this file exists to cover.
 */
function seedWorkspace(selectedTab: unknown) {
  window.localStorage.setItem(
    WORKSPACE_KEY,
    JSON.stringify({
      version: 1,
      name: "saved-by-an-older-build",
      run: null,
      ui: { selectedTab, selectedFactorNode: null, selectedLineage: null, sessionId: null },
    }),
  );
}

afterEach(() => {
  window.localStorage.clear();
  vi.resetModules();
});

describe("restoring a workspace from an older build", () => {
  it("still renders when the saved tab no longer exists", async () => {
    seedWorkspace("explain"); // renamed to "agent" in P11

    const { App } = await import("./App");
    render(<App />);

    // The shell rendering at all is the assertion: before the fix this threw during render.
    expect(await screen.findByTestId("app-shell")).toBeInTheDocument();
    await waitFor(() => expect(screen.getByTestId("main-nav")).toBeInTheDocument());
  });

  it("falls back to a real tab rather than showing an empty header", async () => {
    seedWorkspace("explain");

    const { App } = await import("./App");
    render(<App />);

    // App mode with no loaded run starts at the launcher, which is where a fresh user lands too.
    expect(await screen.findByRole("heading", { name: "Train", level: 1 })).toBeInTheDocument();
  });

  it("keeps honouring a saved tab that does still exist", async () => {
    seedWorkspace("genealogy");

    const { App } = await import("./App");
    render(<App />);

    expect(await screen.findByRole("heading", { name: "Results · Lineage", level: 1 })).toBeInTheDocument();
  });

  it("survives a saved tab that is not a string at all", async () => {
    seedWorkspace(7);

    const { App } = await import("./App");
    render(<App />);

    expect(await screen.findByTestId("app-shell")).toBeInTheDocument();
  });
});
