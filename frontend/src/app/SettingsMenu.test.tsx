import { fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import type { Settings } from "../api/types";

const getSettings = vi.fn();
const getDataUsage = vi.fn();
const putSettings = vi.fn();
const clearData = vi.fn();

vi.mock("../api/client", () => ({
  getSettings: () => getSettings(),
  getDataUsage: () => getDataUsage(),
  putSettings: (u: unknown) => putSettings(u),
  clearData: (c: string) => clearData(c),
}));

import { SettingsMenu } from "./SettingsMenu";

const SETTINGS: Settings = {
  factors_dir: "/data/factors",
  tiingo_api_key_set: false,
  tiingo_api_key_source: "none",
  tiingo_stored_key_set: false,
  evaluator: "auto",
  cpp_available: false,
};

function renderMenu(settings = SETTINGS) {
  getSettings.mockResolvedValue(settings);
  getDataUsage.mockResolvedValue([
    { key: "sessions", label: "Training sessions", bytes: 2048, count: 3 },
  ]);
  putSettings.mockResolvedValue(settings);
  clearData.mockResolvedValue({ key: "sessions", label: "Training sessions", bytes: 0, count: 0 });
  render(
    <SettingsMenu
      mode="app"
      onRefreshRun={vi.fn()}
      onSaveLocal={vi.fn()}
      onLoadLocal={vi.fn()}
      onSaveBackend={vi.fn()}
      onLoadBackend={vi.fn()}
    />,
  );
}

afterEach(() => vi.clearAllMocks());

describe("SettingsMenu (L6)", () => {
  it("opens an accessible dialog with reorganized settings and no nested Quit", async () => {
    renderMenu();
    fireEvent.click(screen.getByRole("button", { name: "Open settings" }));
    const popover = await screen.findByTestId("settings-popover");
    expect(popover).toHaveAttribute("role", "dialog");
    expect(popover).toHaveAttribute("aria-modal", "false");
    expect(within(popover).getByRole("heading", { name: "Settings" })).toBeInTheDocument();
    expect(within(popover).getByRole("button", { name: /Workspace/ })).toBeInTheDocument();
    expect(within(popover).getByRole("button", { name: /Data & evaluator/ })).toBeInTheDocument();
    expect(within(popover).getByRole("button", { name: /Storage/ })).toBeInTheDocument();
    expect(within(popover).queryByRole("button", { name: /Quit/ })).not.toBeInTheDocument();

    expect(popover).toHaveTextContent("Save local");
    expect(popover).toHaveTextContent("Load backend");
  });

  it("saves the evaluator backend choice", async () => {
    renderMenu();
    fireEvent.click(screen.getByRole("button", { name: "Open settings" }));
    const popover = await screen.findByTestId("settings-popover");
    await waitFor(() => expect(getSettings).toHaveBeenCalled());
    fireEvent.click(within(popover).getByRole("button", { name: /Data & evaluator/ }));
    fireEvent.change(await screen.findByLabelText("Evaluator backend"), {
      target: { value: "python" },
    });
    await waitFor(() => expect(putSettings).toHaveBeenCalledWith({ evaluator: "python" }));
  });

  it("refetches whenever it opens and returns focus to the gear on close", async () => {
    renderMenu();
    const gear = screen.getByRole("button", { name: "Open settings" });
    fireEvent.click(gear);
    await waitFor(() => expect(getSettings).toHaveBeenCalledTimes(1));

    fireEvent.click(await screen.findByRole("button", { name: "Close settings" }));
    await waitFor(() => expect(gear).toHaveFocus());

    fireEvent.click(gear);
    await waitFor(() => expect(getSettings).toHaveBeenCalledTimes(2));
  });

  it("shows environment configuration without a misleading key input", async () => {
    renderMenu({
      ...SETTINGS,
      tiingo_api_key_set: true,
      tiingo_api_key_source: "environment",
    });
    fireEvent.click(screen.getByRole("button", { name: "Open settings" }));
    const popover = await screen.findByTestId("settings-popover");
    await waitFor(() => expect(getSettings).toHaveBeenCalled());
    fireEvent.click(within(popover).getByRole("button", { name: /Data & evaluator/ }));

    expect(await screen.findByText(/Configured through the process environment/)).toBeInTheDocument();
    expect(screen.queryByLabelText("Tiingo API key")).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Save key" })).not.toBeInTheDocument();
  });

  it("can remove an unused stored fallback while an environment key remains effective", async () => {
    vi.spyOn(window, "confirm").mockReturnValue(true);
    renderMenu({
      ...SETTINGS,
      tiingo_api_key_set: true,
      tiingo_api_key_source: "environment",
      tiingo_stored_key_set: true,
    });
    fireEvent.click(screen.getByRole("button", { name: "Open settings" }));
    const popover = await screen.findByTestId("settings-popover");
    await waitFor(() => expect(getSettings).toHaveBeenCalled());
    fireEvent.click(within(popover).getByRole("button", { name: /Data & evaluator/ }));
    fireEvent.click(await screen.findByRole("button", { name: "Remove stored key" }));

    await waitFor(() =>
      expect(putSettings).toHaveBeenCalledWith({ tiingo_api_key: "" }),
    );
    expect(screen.queryByLabelText("Tiingo API key")).not.toBeInTheDocument();
  });

  it("removes only a stored key after explicit confirmation", async () => {
    vi.spyOn(window, "confirm").mockReturnValue(true);
    renderMenu({
      ...SETTINGS,
      tiingo_api_key_set: true,
      tiingo_api_key_source: "stored",
      tiingo_stored_key_set: true,
    });
    fireEvent.click(screen.getByRole("button", { name: "Open settings" }));
    const popover = await screen.findByTestId("settings-popover");
    await waitFor(() => expect(getSettings).toHaveBeenCalled());
    fireEvent.click(within(popover).getByRole("button", { name: /Data & evaluator/ }));
    fireEvent.click(await screen.findByRole("button", { name: "Remove stored key" }));

    await waitFor(() =>
      expect(putSettings).toHaveBeenCalledWith({ tiingo_api_key: "" }),
    );
  });

  it("clears a data category after confirmation", async () => {
    vi.spyOn(window, "confirm").mockReturnValue(true);
    renderMenu();
    fireEvent.click(screen.getByRole("button", { name: "Open settings" }));
    const popover = await screen.findByTestId("settings-popover");
    fireEvent.click(within(popover).getByRole("button", { name: /Local data/ }));
    await screen.findByTestId("data-rows");
    fireEvent.click(screen.getByText("Clear"));
    await waitFor(() => expect(clearData).toHaveBeenCalledWith("sessions"));
  });

  it("surfaces load errors with a retry action", async () => {
    getSettings.mockRejectedValue(new Error("offline"));
    getDataUsage.mockRejectedValue(new Error("offline"));
    render(
      <SettingsMenu
        mode="app"
        onRefreshRun={vi.fn()}
        onSaveLocal={vi.fn()}
        onLoadLocal={vi.fn()}
        onSaveBackend={vi.fn()}
        onLoadBackend={vi.fn()}
      />,
    );

    fireEvent.click(screen.getByRole("button", { name: "Open settings" }));
    expect(await screen.findByRole("alert")).toHaveTextContent(/could not be loaded/i);
    expect(screen.getByRole("button", { name: "Retry" })).toBeInTheDocument();
  });
});
