import { fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

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

const SETTINGS = {
  factors_dir: "/data/factors",
  tiingo_api_key_set: false,
  evaluator: "auto" as const,
  cpp_available: false,
};

function renderMenu(onQuit = vi.fn()) {
  getSettings.mockResolvedValue(SETTINGS);
  getDataUsage.mockResolvedValue([
    { key: "sessions", label: "Training sessions", bytes: 2048, count: 3 },
  ]);
  putSettings.mockResolvedValue(SETTINGS);
  clearData.mockResolvedValue({ key: "sessions", label: "Training sessions", bytes: 0, count: 0 });
  render(
    <SettingsMenu
      mode="app"
      onRefreshRun={vi.fn()}
      onSaveLocal={vi.fn()}
      onLoadLocal={vi.fn()}
      onSaveBackend={vi.fn()}
      onLoadBackend={vi.fn()}
      onQuit={onQuit}
    />,
  );
  return onQuit;
}

afterEach(() => vi.clearAllMocks());

describe("SettingsMenu (L6)", () => {
  it("opens the gear and shows the relocated workspace + quit actions", async () => {
    renderMenu();
    fireEvent.click(screen.getByLabelText("Settings menu"));
    const popover = await screen.findByTestId("settings-popover");
    expect(within(popover).getByRole("button", { name: /Workspace/ })).toBeInTheDocument();
    expect(within(popover).getByRole("button", { name: /Quit/ })).toBeInTheDocument();

    fireEvent.click(within(popover).getByRole("button", { name: /Workspace/ }));
    expect(popover).toHaveTextContent("Save local");
    expect(popover).toHaveTextContent("Load backend");

    fireEvent.click(within(popover).getByRole("button", { name: /Quit/ }));
    expect(screen.getByTestId("quit")).toBeInTheDocument();
  });

  it("saves the evaluator backend choice", async () => {
    renderMenu();
    fireEvent.click(screen.getByLabelText("Settings menu"));
    const popover = await screen.findByTestId("settings-popover");
    fireEvent.click(within(popover).getByRole("button", { name: /Settings/ }));
    fireEvent.change(await screen.findByLabelText("Evaluator backend"), {
      target: { value: "python" },
    });
    await waitFor(() => expect(putSettings).toHaveBeenCalledWith({ evaluator: "python" }));
  });

  it("clears a data category after confirmation", async () => {
    vi.spyOn(window, "confirm").mockReturnValue(true);
    renderMenu();
    fireEvent.click(screen.getByLabelText("Settings menu"));
    const popover = await screen.findByTestId("settings-popover");
    fireEvent.click(within(popover).getByRole("button", { name: /Local data/ }));
    await screen.findByTestId("data-rows");
    fireEvent.click(screen.getByText("Clear"));
    await waitFor(() => expect(clearData).toHaveBeenCalledWith("sessions"));
  });

  it("invokes onQuit from the Quit action", async () => {
    const onQuit = renderMenu();
    fireEvent.click(screen.getByLabelText("Settings menu"));
    const popover = await screen.findByTestId("settings-popover");
    fireEvent.click(within(popover).getByRole("button", { name: /Quit/ }));
    fireEvent.click(await screen.findByTestId("quit"));
    expect(onQuit).toHaveBeenCalled();
  });
});
