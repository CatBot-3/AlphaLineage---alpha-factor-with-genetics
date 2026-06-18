import { fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

const defineUniverse = vi.fn();
const deleteUniverse = vi.fn();
const getDataSync = vi.fn();
const getMembershipSync = vi.fn();
const getUniverse = vi.fn();
const listUniverses = vi.fn();
const searchSymbols = vi.fn();
const startDataSync = vi.fn();
const startMembershipSync = vi.fn();
const updateUniverse = vi.fn();
const validateSymbol = vi.fn();

vi.mock("../api/client", () => ({
  defineUniverse: (payload: unknown) => defineUniverse(payload),
  deleteUniverse: (name: string) => deleteUniverse(name),
  getDataSync: (jobId: string) => getDataSync(jobId),
  getMembershipSync: (jobId: string) => getMembershipSync(jobId),
  getUniverse: (name: string) => getUniverse(name),
  listUniverses: () => listUniverses(),
  searchSymbols: (...args: unknown[]) => searchSymbols(...args),
  startDataSync: (payload: unknown) => startDataSync(payload),
  startMembershipSync: (payload: unknown) => startMembershipSync(payload),
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
  listUniverses.mockResolvedValue([
    {
      name: "sp500-lite",
      symbols: ["AAPL"],
      source: "sample",
      memberships: [{ symbol: "AAPL", entry: "2020-01-01", exit: null }],
    },
  ]);
  getUniverse.mockResolvedValue({
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

  it("syncs entry/exit dates and surfaces a removable delisted-symbol list", async () => {
    setupMocks();
    getMembershipSync.mockResolvedValue({
      job_id: "m-1",
      status: "done",
      result: {
        expected_start: "2020-01-01",
        results: [
          { symbol: "AAPL", status: "resolved", entry: "2020-01-01", exit: null, delisted: false },
          {
            symbol: "LEH",
            status: "resolved",
            entry: "2000-01-03",
            exit: "2008-09-15",
            delisted: true,
          },
        ],
      },
      error: null,
      progress: { done: 2, total: 2, current_symbol: "LEH" },
    });
    render(
      <UniverseEditorPage
        draft={{
          name: "u",
          rows: [
            { symbol: "AAPL", entry: "2019-01-01", exit: "" },
            { symbol: "LEH", entry: "2000-01-01", exit: "" },
          ],
        }}
      />,
    );

    fireEvent.click(screen.getByTestId("sync-membership-dates"));

    await waitFor(() => expect(screen.getByLabelText("entry-1")).toHaveValue("2000-01-03"));
    expect(screen.getByLabelText("exit-1")).toHaveValue("2008-09-15");

    const delisted = await screen.findByTestId("delisted-list");
    expect(within(delisted).getByText("LEH")).toBeInTheDocument();
    fireEvent.click(within(delisted).getByText("Remove"));
    expect(screen.queryByTestId("delisted-list")).not.toBeInTheDocument();
  });
});
