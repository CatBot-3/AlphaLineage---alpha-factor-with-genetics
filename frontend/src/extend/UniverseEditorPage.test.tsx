import { act, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

const defineUniverse = vi.fn();
const deleteUniverse = vi.fn();
const getDataCoverage = vi.fn();
const getDataSync = vi.fn();
const getMembershipSync = vi.fn();
const getUniverse = vi.fn();
const listUniversePresets = vi.fn();
const listUniverses = vi.fn();
const listDataSyncs = vi.fn();
const searchSymbols = vi.fn();
const startDataSync = vi.fn();
const startMembershipSync = vi.fn();
const stopDataSync = vi.fn();
const updateUniverse = vi.fn();
const validateSymbol = vi.fn();

vi.mock("../api/client", () => ({
  defineUniverse: (payload: unknown) => defineUniverse(payload),
  deleteUniverse: (name: string) => deleteUniverse(name),
  getDataCoverage: (...args: unknown[]) => getDataCoverage(...args),
  getDataSync: (jobId: string) => getDataSync(jobId),
  getMembershipSync: (jobId: string) => getMembershipSync(jobId),
  getUniverse: (name: string) => getUniverse(name),
  listUniversePresets: () => listUniversePresets(),
  listUniverses: () => listUniverses(),
  listDataSyncs: (...args: unknown[]) => listDataSyncs(...args),
  searchSymbols: (...args: unknown[]) => searchSymbols(...args),
  startDataSync: (payload: unknown) => startDataSync(payload),
  startMembershipSync: (payload: unknown) => startMembershipSync(payload),
  stopDataSync: (jobId: string) => stopDataSync(jobId),
  updateUniverse: (name: string, payload: unknown) => updateUniverse(name, payload),
  validateSymbol: (payload: unknown) => validateSymbol(payload),
}));

import { UniverseEditorPage } from "./UniverseEditorPage";

function candidate(symbol: string) {
  return {
    symbol,
    name: `${symbol} Inc.`,
    exchange: "Nasdaq",
    quote_type: "Equity",
    currency: "USD",
    source: "yfinance",
  };
}

function setupMocks() {
  getDataCoverage.mockResolvedValue([]);
  listDataSyncs.mockResolvedValue([]);
  stopDataSync.mockResolvedValue({ stopping: true });
  listUniversePresets.mockResolvedValue([
    {
      id: "sp500-lite",
      display_name: "Popular US stocks sample",
      benchmark: "US large-cap demonstration basket",
      status: "bundled_sample",
      available: true,
      mode: "static_snapshot",
      membership_history: "static_snapshot",
      coverage: "15 recognizable, liquid US stocks in a dated offline sample",
      research_ready: false,
      provenance: "Bundled current-stock demonstration data",
      warning: "This is a current demonstration basket, not historical index membership.",
      definition: {
        id: "sp500-lite",
        display_name: "Popular US stocks sample",
        snapshot_date: "2026-07-15",
        interval_semantics: "Fixed demonstration basket",
        member_count: 15,
      },
    },
    {
      id: "sp500",
      display_name: "S&P 500",
      benchmark: "S&P 500",
      status: "bundled_static",
      available: true,
      snapshot_universe: "builtin-sp500-current",
      pit_import_name: "sp500",
      mode: "static_snapshot",
      membership_history: "static_snapshot",
      coverage: "Bundled current constituent snapshot",
      research_ready: false,
      provenance: "Bundled index snapshot",
      warning: "A current snapshot is not valid historical membership data.",
      definition: {
        id: "builtin-sp500-current",
        display_name: "S&P 500 static snapshot",
        snapshot_date: "2026-07-14",
        interval_semantics: "All members active from the snapshot date",
        member_count: 2,
      },
      fingerprint: "sha256:sp500",
      provenance_detail: {
        provider: "Wikipedia",
        retrieved_at: "2026-07-14",
        attribution: "Wikipedia contributors",
        license: "CC BY-SA 4.0",
        license_url: "https://creativecommons.org/licenses/by-sa/4.0/",
        terms_url: "https://foundation.wikimedia.org/wiki/Policy:Terms_of_Use",
      },
      readiness: {
        membership_ready: true,
        price_ready: false,
        research_ready: false,
        issues: ["Static membership only"],
      },
    },
  ]);
  listUniverses.mockResolvedValue([
    {
      name: "sp500-lite",
      symbols: ["AAPL"],
      source: "sample",
      memberships: [{ symbol: "AAPL", entry: "2020-01-01", exit: null }],
      cache_coverage: {
        as_of: "2026-07-14",
        eligible_symbols: ["AAPL"],
        cached_symbols: [],
        missing_symbols: ["AAPL"],
        incomplete_symbols: ["AAPL"],
        complete: false,
      },
    },
    {
      name: "builtin-sp500-current",
      display_name: "S&P 500 static snapshot",
      symbols: [],
      source: "bundled",
      mode: "static_snapshot",
      memberships: [],
      definition: {
        id: "builtin-sp500-current",
        display_name: "S&P 500 static snapshot",
        snapshot_date: "2026-07-14",
        interval_semantics: "Static membership",
        member_count: 2,
      },
      fingerprint: "sha256:sp500",
    },
  ]);
  getUniverse.mockImplementation(async (name: string) => name === "builtin-sp500-current" ? {
    name,
    display_name: "S&P 500 static snapshot",
    symbols: ["AAPL", "BRK.B"],
    source: "bundled",
    mode: "static_snapshot",
    memberships: [
      { symbol: "AAPL", entry: "2026-07-14", exit: null },
      { symbol: "BRK.B", entry: "2026-07-14", exit: null },
    ],
    definition: {
      id: name,
      display_name: "S&P 500 static snapshot",
      snapshot_date: "2026-07-14",
      interval_semantics: "Static membership",
      member_count: 2,
    },
    fingerprint: "sha256:sp500",
    provenance: {
      provider: "Wikipedia",
      retrieved_at: "2026-07-14",
      attribution: "Wikipedia contributors",
      license: "CC BY-SA 4.0",
      license_url: "https://creativecommons.org/licenses/by-sa/4.0/",
      terms_url: "https://foundation.wikimedia.org/wiki/Policy:Terms_of_Use",
    },
    readiness: {
      membership_ready: true,
      price_ready: false,
      research_ready: false,
      issues: ["Static membership only"],
    },
    aliases: { "BRK.B": "BRK-B" },
  } : {
    name: "sp500-lite",
    symbols: ["AAPL"],
    source: "sample",
    memberships: [{ symbol: "AAPL", entry: "2020-01-01", exit: null }],
  });
  searchSymbols.mockResolvedValue([candidate("AAPL")]);
  validateSymbol.mockResolvedValue({
    symbol: "AAPL",
    valid: true,
    rows: 100,
    first_date: "2020-01-02",
    last_date: "2026-06-16",
    provider: "yfinance",
    error: null,
  });
  defineUniverse.mockResolvedValue({ name: "my-universe", symbols: ["AAPL"] });
  updateUniverse.mockResolvedValue({ name: "my-universe", symbols: ["AAPL"] });
  deleteUniverse.mockResolvedValue(undefined);
  startDataSync.mockResolvedValue({ job_id: "sync-1", status: "queued" });
  getDataSync.mockResolvedValue({
    job_id: "sync-1",
    status: "done",
    result: { mode: "incremental", start: "2020-01-01", end: null, results: [] },
    error: null,
    progress: { done: 1, total: 1, current_symbol: "AAPL" },
  });
  startMembershipSync.mockResolvedValue({ job_id: "m-1", status: "queued" });
}

beforeEach(() => {
  vi.spyOn(window, "confirm").mockReturnValue(false);
});

afterEach(() => {
  vi.clearAllMocks();
  vi.restoreAllMocks();
});

describe("UniverseEditorPage", () => {
  it("searches ticker candidates, validates one, and adds it to the draft", async () => {
    setupMocks();
    render(<UniverseEditorPage />);

    fireEvent.click(screen.getByRole("button", { name: /Add ticker/ }));
    fireEvent.change(screen.getByLabelText("Ticker query"), { target: { value: "AAPL" } });
    fireEvent.click(screen.getByTestId("search-symbols"));

    const candidates = await screen.findByTestId("symbol-candidates");
    fireEvent.click(within(candidates).getByText("AAPL"));
    expect(await screen.findByTestId("symbol-validation")).toHaveTextContent("validated");

    fireEvent.click(screen.getByText("Add selected ticker"));
    expect(screen.getByLabelText("symbol-0")).toHaveValue("AAPL");
  });

  it("triggers search on Enter in the ticker query box", async () => {
    setupMocks();
    render(<UniverseEditorPage />);

    fireEvent.click(screen.getByRole("button", { name: /Add ticker/ }));
    fireEvent.change(screen.getByLabelText("Ticker query"), { target: { value: "AAPL" } });
    fireEvent.keyDown(screen.getByLabelText("Ticker query"), { key: "Enter" });

    await screen.findByTestId("symbol-candidates");
    expect(searchSymbols).toHaveBeenCalledWith("AAPL", 15);
  });

  it("marks the selected candidate and shows a verified badge once validated", async () => {
    setupMocks();
    render(<UniverseEditorPage />);

    fireEvent.click(screen.getByRole("button", { name: /Add ticker/ }));
    fireEvent.change(screen.getByLabelText("Ticker query"), { target: { value: "AAPL" } });
    fireEvent.click(screen.getByTestId("search-symbols"));

    const candidates = await screen.findByTestId("symbol-candidates");
    const candidateButton = within(candidates).getByText("AAPL").closest("button");
    expect(candidateButton).not.toBeNull();
    fireEvent.click(candidateButton as HTMLButtonElement);

    await screen.findByTestId("symbol-validation");
    expect(candidateButton).toHaveAttribute("aria-pressed", "true");
    expect(within(candidates).getByTestId("verified-badge")).toBeInTheDocument();
  });

  it("offers to pull data right after adding a ticker", async () => {
    setupMocks();
    vi.spyOn(window, "confirm").mockReturnValue(true);
    render(<UniverseEditorPage />);

    fireEvent.click(screen.getByRole("button", { name: /Add ticker/ }));
    fireEvent.change(screen.getByLabelText("Ticker query"), { target: { value: "AAPL" } });
    fireEvent.click(screen.getByTestId("search-symbols"));
    const candidates = await screen.findByTestId("symbol-candidates");
    fireEvent.click(within(candidates).getByText("AAPL"));
    await screen.findByTestId("symbol-validation");

    fireEvent.click(screen.getByText("Add selected ticker"));

    expect(window.confirm).toHaveBeenCalledWith(expect.stringContaining("Pull price data for AAPL"));
    await waitFor(() =>
      expect(startDataSync).toHaveBeenCalledWith(
        expect.objectContaining({ symbols: ["AAPL"], mode: "incremental" }),
      ),
    );
  });

  it("declines the pull when the user says no", async () => {
    setupMocks();
    vi.spyOn(window, "confirm").mockReturnValue(false);
    render(<UniverseEditorPage />);

    fireEvent.click(screen.getByRole("button", { name: /Add ticker/ }));
    fireEvent.change(screen.getByLabelText("Ticker query"), { target: { value: "AAPL" } });
    fireEvent.click(screen.getByTestId("search-symbols"));
    const candidates = await screen.findByTestId("symbol-candidates");
    fireEvent.click(within(candidates).getByText("AAPL"));
    await screen.findByTestId("symbol-validation");
    fireEvent.click(screen.getByText("Add selected ticker"));

    expect(window.confirm).toHaveBeenCalled();
    expect(startDataSync).not.toHaveBeenCalled();
  });

  it("reveals more than 5 search results via a show-more control", async () => {
    setupMocks();
    searchSymbols.mockResolvedValue(
      Array.from({ length: 8 }, (_, i) => candidate(`SYM${i}`)),
    );
    render(<UniverseEditorPage />);

    fireEvent.click(screen.getByRole("button", { name: /Add ticker/ }));
    fireEvent.change(screen.getByLabelText("Ticker query"), { target: { value: "SYM" } });
    fireEvent.click(screen.getByTestId("search-symbols"));

    const candidates = await screen.findByTestId("symbol-candidates");
    expect(within(candidates).getAllByRole("button")).toHaveLength(5);

    fireEvent.click(screen.getByTestId("show-more-candidates"));
    expect(within(candidates).getAllByRole("button")).toHaveLength(8);
    expect(screen.queryByTestId("show-more-candidates")).not.toBeInTheDocument();
  });

  it("wraps the symbols table in a scrollable container", async () => {
    setupMocks();
    const { container } = render(
      <UniverseEditorPage
        draft={{
          name: "u",
          rows: Array.from({ length: 12 }, (_, i) => ({
            symbol: `S${i}`,
            entry: "2020-01-01",
            exit: "",
          })),
        }}
      />,
    );
    await waitFor(() => expect(listUniverses).toHaveBeenCalled());
    expect(container.querySelector(".universe-rows-scroll table")).not.toBeNull();
    expect(screen.getByLabelText("symbol-11")).toHaveValue("S11");
  });

  it("resets the draft via New universe", async () => {
    setupMocks();
    render(
      <UniverseEditorPage draft={{ name: "loaded-u", rows: [{ symbol: "AAPL", entry: "2020-01-01", exit: "" }] }} />,
    );
    await waitFor(() => expect(listUniverses).toHaveBeenCalled());
    expect(screen.getByLabelText("symbol-0")).toHaveValue("AAPL");

    fireEvent.click(screen.getByTestId("new-universe"));
    expect(screen.getByDisplayValue("my-universe")).toBeInTheDocument();
    expect(screen.getByLabelText("symbol-0")).toHaveValue("");
  });

  it("warns before overwriting a different existing custom universe", async () => {
    setupMocks();
    listUniverses.mockResolvedValue([
      { name: "other-universe", symbols: ["MSFT"], source: "custom", memberships: [] },
    ]);
    vi.spyOn(window, "confirm").mockReturnValue(false);
    render(
      <UniverseEditorPage
        draft={{ name: "other-universe", rows: [{ symbol: "AAPL", entry: "2020-01-01", exit: "" }] }}
      />,
    );

    await waitFor(() => expect(listUniverses).toHaveBeenCalled());
    fireEvent.click(screen.getByTestId("save-universe"));

    await waitFor(() => expect(window.confirm).toHaveBeenCalled());
    expect(defineUniverse).not.toHaveBeenCalled();
  });

  it("renders the rename/recolor/reposition of the delete button", async () => {
    setupMocks();
    render(<UniverseEditorPage />);
    await waitFor(() => expect(listUniverses).toHaveBeenCalled());
    const deleteButton = screen.getByTestId("delete-universe");
    expect(deleteButton).toHaveTextContent("Delete this universe");
    expect(deleteButton).toHaveClass("danger-navy");
  });

  it("preserves memberships when stale prices cannot verify a delisting", async () => {
    setupMocks();
    getMembershipSync.mockResolvedValue({
      job_id: "m-1",
      status: "done",
      result: {
        expected_start: "2020-01-01",
        results: [
          {
            symbol: "AAPL",
            status: "resolved",
            entry: null,
            exit: null,
            delisted: false,
            first_date: "1999-11-01",
            last_date: "2026-07-24",
            note: "Observed price coverage only. Membership entry and exit were left unchanged.",
          },
          {
            symbol: "STALE",
            status: "unverified_stale",
            entry: null,
            exit: null,
            delisted: false,
            review_needed: true,
            last_date: "2008-09-12",
            note: "Membership was left unchanged; confirm any exit independently.",
          },
          {
            symbol: "BAD",
            status: "failed",
            entry: null,
            exit: null,
            delisted: false,
            error: "Provider rejected symbol",
          },
        ],
      },
      error: null,
      progress: { done: 2, total: 2, current_symbol: "STALE" },
    });
    render(
      <UniverseEditorPage
        draft={{
          name: "u",
          rows: [
            { symbol: "AAPL", entry: "2019-01-01", exit: "2024-01-01" },
            { symbol: "STALE", entry: "2000-01-01", exit: "" },
            { symbol: "BAD", entry: "2010-01-01", exit: "" },
          ],
        }}
      />,
    );

    fireEvent.click(screen.getByTestId("sync-membership-dates"));

    await waitFor(() => expect(screen.getByTestId("stale-membership-list")).toBeInTheDocument());
    expect(screen.getByLabelText("entry-0")).toHaveValue("2019-01-01");
    expect(screen.getByLabelText("exit-0")).toHaveValue("2024-01-01");
    expect(screen.getByLabelText("entry-1")).toHaveValue("2000-01-01");
    expect(screen.getByLabelText("exit-1")).toHaveValue("");
    expect(screen.getByLabelText("entry-2")).toHaveValue("2010-01-01");

    const review = await screen.findByTestId("stale-membership-list");
    expect(review).toHaveTextContent("1999-11-01 to 2026-07-24");
    expect(within(review).getByText("STALE")).toBeInTheDocument();
    expect(review).toHaveTextContent("Membership was left unchanged");
    expect(within(review).getByText("BAD")).toBeInTheDocument();
    expect(review).toHaveTextContent("Provider rejected symbol");
    expect(screen.queryByTestId("delisted-list")).not.toBeInTheDocument();
  });

  it("keeps snapshot bias disclosure compact while preparing a separate PIT import", async () => {
    setupMocks();
    render(<UniverseEditorPage />);
    const presets = await screen.findByTestId("universe-presets");
    const sp500 = within(presets).getByText("S&P 500", { selector: "strong" }).closest("article")!;
    expect(within(sp500).getByText("Current members ⓘ")).toBeInTheDocument();
    expect(within(sp500).queryByText(/current snapshot is not valid/i, {
      selector: ".oos-warning",
    })).not.toBeInTheDocument();
    fireEvent.click(within(sp500).getByText("Source and research notes"));
    expect(within(sp500).getByText(/current snapshot is not valid/i)).toBeInTheDocument();

    fireEvent.click(within(sp500).getByRole("button", { name: "Import point-in-time history" }));

    expect(screen.getByLabelText("Universe name")).toHaveValue("sp500");
    expect(screen.getByLabelText("symbol-0")).toHaveValue("");
    expect(screen.getByTestId("universe-result")).toHaveTextContent(/No memberships were generated/i);
    expect(screen.getByTestId("universe-result")).toHaveTextContent(/separate/i);
    expect(screen.getByRole("button", { name: /Import membership history/ })).toHaveAttribute("aria-expanded", "true");
  });

  it("loads a bundled static snapshot separately and shows its definition metadata", async () => {
    setupMocks();
    render(<UniverseEditorPage />);
    const presets = await screen.findByTestId("universe-presets");
    const sp500 = within(presets).getByText("S&P 500", { selector: "strong" }).closest("article")!;

    fireEvent.click(within(sp500).getByRole("button", { name: "Load current snapshot" }));

    await waitFor(() => expect(getUniverse).toHaveBeenCalledWith("builtin-sp500-current"));
    const metadata = await screen.findByTestId("universe-integrity");
    expect(metadata).toHaveTextContent("Mode: static snapshot");
    expect(metadata).toHaveTextContent("Source: bundled");
    expect(metadata).toHaveTextContent("Fingerprint: sha256:sp500");
    expect(metadata).toHaveTextContent("Attribution: Wikipedia contributors · CC BY-SA 4.0");
    expect(within(metadata).getByRole("link", { name: "License terms" })).toHaveAttribute(
      "href",
      "https://creativecommons.org/licenses/by-sa/4.0/",
    );
    expect(within(metadata).getByRole("link", { name: "Wikimedia terms" })).toHaveAttribute(
      "href",
      "https://foundation.wikimedia.org/wiki/Policy:Terms_of_Use",
    );
    expect(metadata).toHaveTextContent("BRK.B → BRK-B");
  });

  it("previews and imports pasted point-in-time membership CSV", async () => {
    setupMocks();
    render(<UniverseEditorPage />);
    await screen.findByTestId("universe-presets");
    fireEvent.click(screen.getByRole("button", { name: /Import membership history/ }));
    fireEvent.change(screen.getByLabelText("Point-in-time membership CSV or TSV"), {
      target: { value: "symbol,entry,exit\naapl,2000-01-03,\nLEH,2000-01-03,2008-09-15" },
    });
    expect(screen.getByTestId("membership-import-summary")).toHaveTextContent("2 valid interval(s); 0 error(s)");

    fireEvent.click(screen.getByRole("button", { name: "Replace draft with valid rows" }));
    expect(screen.getByLabelText("symbol-0")).toHaveValue("AAPL");
    expect(screen.getByLabelText("exit-1")).toHaveValue("2008-09-15");
    fireEvent.click(screen.getByTestId("save-universe"));
    await waitFor(() => expect(defineUniverse).toHaveBeenCalledWith({
      name: "my-universe",
      memberships: [
        { symbol: "AAPL", entry: "2000-01-03", exit: null },
        { symbol: "LEH", entry: "2000-01-03", exit: "2008-09-15" },
      ],
    }));
  });

  it("surfaces paste errors and only imports rows explicitly marked valid", async () => {
    setupMocks();
    render(<UniverseEditorPage />);
    await screen.findByTestId("universe-presets");
    fireEvent.click(screen.getByRole("button", { name: /Import membership history/ }));
    fireEvent.change(screen.getByLabelText("Point-in-time membership CSV or TSV"), {
      target: { value: "BAD,not-a-date,\nMSFT,2020-01-01," },
    });
    expect(screen.getByTestId("membership-import-summary")).toHaveTextContent("1 valid interval(s); 1 error(s)");
    expect(screen.getByTestId("membership-import-errors")).toHaveTextContent("Line 1");
    fireEvent.click(screen.getByRole("button", { name: "Replace draft with valid rows" }));
    expect(screen.getByLabelText("symbol-0")).toHaveValue("MSFT");
    expect(screen.getByTestId("universe-result")).toHaveTextContent(/skipped 1 invalid line/i);
  });

  it("surfaces incomplete price-cache history before training", async () => {
    setupMocks();
    render(<UniverseEditorPage />);
    await screen.findByTestId("universe-presets");
    fireEvent.change(screen.getByLabelText("Load universe"), {
      target: { value: "sp500-lite" },
    });

    const coverage = await screen.findByTestId("universe-cache-coverage");
    expect(coverage).toHaveTextContent("Price history: needs attention");
    expect(coverage).toHaveTextContent("AAPL");
  });

  it("ignores a slower universe response after the user selects another definition", async () => {
    setupMocks();
    const slowUniverse = {
      name: "slow-universe",
      symbols: ["OLD"],
      source: "custom",
      mode: "point_in_time",
      memberships: [{ symbol: "OLD", entry: "2000-01-01", exit: null }],
    };
    const fastUniverse = {
      name: "fast-universe",
      symbols: ["NEW"],
      source: "custom",
      mode: "point_in_time",
      memberships: [{ symbol: "NEW", entry: "2020-01-01", exit: null }],
    };
    listUniverses.mockResolvedValue([slowUniverse, fastUniverse]);
    let resolveSlow: (value: typeof slowUniverse) => void = () => undefined;
    const slowResponse = new Promise<typeof slowUniverse>((resolve) => {
      resolveSlow = resolve;
    });
    getUniverse.mockImplementation((universeName: string) => (
      universeName === "slow-universe" ? slowResponse : Promise.resolve(fastUniverse)
    ));

    render(<UniverseEditorPage />);
    await screen.findByRole("option", { name: "slow-universe (custom)" });
    fireEvent.change(screen.getByLabelText("Load universe"), {
      target: { value: "slow-universe" },
    });
    fireEvent.change(screen.getByLabelText("Load universe"), {
      target: { value: "fast-universe" },
    });

    await waitFor(() => expect(screen.getByLabelText("Universe name")).toHaveValue("fast-universe"));
    expect(screen.getByLabelText("symbol-0")).toHaveValue("NEW");
    await act(async () => {
      resolveSlow(slowUniverse);
    });
    expect(screen.getByLabelText("Universe name")).toHaveValue("fast-universe");
    expect(screen.getByLabelText("symbol-0")).toHaveValue("NEW");
  });

  it("does not let saved-draft hydration replace a universe selected after reopen", async () => {
    setupMocks();
    const savedUniverse = {
      name: "saved-universe",
      symbols: ["OLD"],
      source: "custom",
      mode: "point_in_time",
      memberships: [{ symbol: "OLD", entry: "2010-01-01", exit: null }],
    };
    const selectedUniverse = {
      name: "selected-universe",
      symbols: ["NEW"],
      source: "custom",
      mode: "point_in_time",
      memberships: [{ symbol: "NEW", entry: "2020-01-01", exit: null }],
    };
    listUniverses.mockResolvedValue([savedUniverse, selectedUniverse]);
    let resolveHydration!: (value: typeof savedUniverse) => void;
    const hydration = new Promise<typeof savedUniverse>((resolve) => {
      resolveHydration = resolve;
    });
    getUniverse.mockImplementation((universeName: string) => (
      universeName === savedUniverse.name
        ? hydration
        : Promise.resolve(selectedUniverse)
    ));

    render(
      <UniverseEditorPage
        draft={{
          name: savedUniverse.name,
          selectedUniverse: savedUniverse.name,
          expectedStart: "2010-01-01",
          rows: [{ symbol: "OLD", entry: "2010-01-01", exit: "" }],
        }}
      />,
    );
    await screen.findByRole("option", { name: "selected-universe (custom)" });
    fireEvent.change(screen.getByLabelText("Load universe"), {
      target: { value: selectedUniverse.name },
    });
    await waitFor(() => expect(screen.getByLabelText("Universe name"))
      .toHaveValue(selectedUniverse.name));

    await act(async () => {
      resolveHydration(savedUniverse);
    });

    expect(screen.getByLabelText("Universe name")).toHaveValue(selectedUniverse.name);
    expect(screen.getByLabelText("symbol-0")).toHaveValue("NEW");
  });

  it("stops direct membership polling when the selected universe changes", async () => {
    setupMocks();
    const first = {
      name: "first-universe",
      symbols: ["AAA"],
      source: "custom",
      mode: "point_in_time",
      memberships: [{ symbol: "AAA", entry: "2020-01-01", exit: null }],
    };
    const second = {
      name: "second-universe",
      symbols: ["BBB"],
      source: "custom",
      mode: "point_in_time",
      memberships: [{ symbol: "BBB", entry: "2020-01-01", exit: null }],
    };
    listUniverses.mockResolvedValue([first, second]);
    getUniverse.mockImplementation(async (universeName: string) => (
      universeName === first.name ? first : second
    ));
    let resolveMembership!: (value: {
      job_id: string;
      status: string;
      result: null;
      error: null;
      progress: { done: number; total: number; current_symbol: string };
    }) => void;
    getMembershipSync.mockReturnValue(new Promise((resolve) => {
      resolveMembership = resolve;
    }));
    const onPullProgress = vi.fn();

    render(<UniverseEditorPage onPullProgress={onPullProgress} />);
    await screen.findByRole("option", { name: "first-universe (custom)" });
    fireEvent.change(screen.getByLabelText("Load universe"), {
      target: { value: first.name },
    });
    await waitFor(() => expect(screen.getByLabelText("symbol-0")).toHaveValue("AAA"));
    fireEvent.click(screen.getByTestId("sync-membership-dates"));
    await waitFor(() => expect(getMembershipSync).toHaveBeenCalledWith("m-1"));

    fireEvent.change(screen.getByLabelText("Load universe"), {
      target: { value: second.name },
    });
    await waitFor(() => expect(screen.getByLabelText("symbol-0")).toHaveValue("BBB"));
    await act(async () => {
      resolveMembership({
        job_id: "m-1",
        status: "running",
        result: null,
        error: null,
        progress: { done: 1, total: 2, current_symbol: "AAA" },
      });
    });

    expect(onPullProgress).not.toHaveBeenCalledWith(
      expect.objectContaining({ current_symbol: "AAA" }),
    );
    expect(getMembershipSync).toHaveBeenCalledTimes(1);
  });

  it("syncs from the earliest PIT membership and refreshes the stale readiness card", async () => {
    setupMocks();
    const incomplete = {
      name: "sp500-lite",
      symbols: ["AAPL", "LEH"],
      source: "sample",
      mode: "point_in_time",
      memberships: [
        { symbol: "AAPL", entry: "2000-01-01", exit: null },
        { symbol: "LEH", entry: "2000-01-01", exit: "2008-09-15" },
      ],
      cache_coverage: {
        as_of: "2026-07-16",
        required_start: "2000-01-01",
        required_end: "2026-07-16",
        eligible_symbols: ["AAPL", "LEH"],
        active_symbols: ["AAPL"],
        exited_symbols: ["LEH"],
        cached_symbols: ["AAPL", "LEH"],
        missing_symbols: [],
        incomplete_symbols: ["AAPL", "LEH"],
        complete: false,
      },
    };
    const complete = {
      ...incomplete,
      cache_coverage: {
        ...incomplete.cache_coverage,
        incomplete_symbols: [],
        complete: true,
      },
    };
    getUniverse.mockReset();
    getUniverse.mockResolvedValueOnce(incomplete).mockResolvedValueOnce(complete);
    startDataSync.mockResolvedValue({ job_id: "sync-pit", status: "queued", reused: false });
    getDataSync.mockResolvedValue({
      job_id: "sync-pit",
      status: "done",
      request: { universe: "sp500-lite", start: "2000-01-01", mode: "incremental" },
      result: {
        universe: "sp500-lite",
        mode: "incremental",
        start: "2000-01-01",
        results: [],
        failed_count: 0,
        succeeded_count: 2,
      },
      error: null,
    });

    render(<UniverseEditorPage />);
    await screen.findByTestId("universe-presets");
    fireEvent.change(screen.getByLabelText("Load universe"), {
      target: { value: "sp500-lite" },
    });
    await waitFor(() => expect(screen.getByLabelText("Sync start date")).toHaveValue("2000-01-01"));
    expect(screen.getByTestId("universe-cache-coverage")).toHaveTextContent(
      "1 exited membership requires prices only through",
    );
    fireEvent.click(screen.getByTestId("sync-universe"));

    await waitFor(() => expect(startDataSync).toHaveBeenCalledWith({
      universe: "sp500-lite",
      start: "2000-01-01",
      end: undefined,
      mode: "incremental",
    }));
    await waitFor(() => expect(screen.getByTestId("universe-cache-coverage")).toHaveTextContent(
      "Price history: complete",
    ));
  });

  it("does not sync a saved identity while its displayed definition has unsaved edits", async () => {
    setupMocks();
    render(<UniverseEditorPage />);
    await screen.findByTestId("universe-presets");
    fireEvent.change(screen.getByLabelText("Load universe"), {
      target: { value: "sp500-lite" },
    });
    await waitFor(() => expect(getUniverse).toHaveBeenCalledWith("sp500-lite"));
    fireEvent.change(screen.getByLabelText("Universe name"), { target: { value: "edited-name" } });

    expect(await screen.findByText(/Save this universe before syncing/i)).toBeInTheDocument();
    expect(screen.getByTestId("sync-universe")).toBeDisabled();
  });
});
