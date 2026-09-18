import { fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

const createUniverseFolder = vi.fn();
const deleteUniverseFolder = vi.fn();
const moveUniverses = vi.fn();
const updateUniverseFolder = vi.fn();

vi.mock("../api/client", () => ({
  createUniverseFolder: (...args: unknown[]) => createUniverseFolder(...args),
  deleteUniverseFolder: (...args: unknown[]) => deleteUniverseFolder(...args),
  moveUniverses: (...args: unknown[]) => moveUniverses(...args),
  updateUniverseFolder: (...args: unknown[]) => updateUniverseFolder(...args),
}));

import type { UniverseFolder, UniverseInfo } from "../api/types";
import { UniversePicker, UniverseTree, buildUniverseTree } from "./UniverseTree";

const folders: UniverseFolder[] = [
  { id: "sectors", name: "S&P 500 sectors (GICS)", parent: null, builtin: true },
  { id: "themes", name: "S&P 500 themes", parent: null, builtin: true },
  { id: "tech", name: "Technology", parent: "themes", builtin: true },
];

function universe(name: string, extra: Partial<UniverseInfo> = {}): UniverseInfo {
  return { name, symbols: ["AAA"], memberships: [], source: "bundled", ...extra };
}

const universes: UniverseInfo[] = [
  universe("sp500-lite", { display_name: "Popular US stocks sample", source: "sample", folder_id: null }),
  universe("builtin-sp500-sector-energy", {
    display_name: "S&P 500 Energy sector",
    symbol_count: 21,
    folder_id: "sectors",
    folder_path: ["S&P 500 sectors (GICS)"],
  }),
  universe("builtin-sp500-theme-semiconductors", {
    display_name: "S&P 500 Semiconductors & equipment",
    symbol_count: 20,
    folder_id: "tech",
    folder_path: ["S&P 500 themes", "Technology"],
  }),
];

describe("UniverseTree", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    createUniverseFolder.mockResolvedValue({ id: "new", name: "Mine", parent: null, builtin: false });
    updateUniverseFolder.mockResolvedValue({});
    deleteUniverseFolder.mockResolvedValue({ removed: "tech", moved_to: "themes" });
    moveUniverses.mockResolvedValue({});
  });

  it("keeps folders collapsed until the user opens them", () => {
    const onSelect = vi.fn();
    render(<UniverseTree universes={universes} folders={folders} selected="" onSelect={onSelect} />);

    expect(screen.getByRole("button", { name: "Load Popular US stocks sample" })).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Load S&P 500 Energy sector" })).not.toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Expand folder S&P 500 themes" })).toHaveTextContent("1");

    fireEvent.click(screen.getByRole("button", { name: "Expand folder S&P 500 sectors (GICS)" }));
    fireEvent.click(screen.getByRole("button", { name: "Load S&P 500 Energy sector" }));
    expect(onSelect).toHaveBeenCalledWith("builtin-sp500-sector-energy");

    // Nested folders open one level at a time.
    fireEvent.click(screen.getByRole("button", { name: "Expand folder S&P 500 themes" }));
    expect(screen.queryByRole("button", { name: "Load S&P 500 Semiconductors & equipment" })).not.toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Expand folder Technology" }));
    expect(screen.getByRole("button", { name: "Load S&P 500 Semiconductors & equipment" })).toBeInTheDocument();
  });

  it("filters across folders and reveals only the path of the selected universe", () => {
    render(
      <UniverseTree
        universes={universes}
        folders={folders}
        selected="builtin-sp500-theme-semiconductors"
        onSelect={vi.fn()}
      />,
    );
    expect(screen.getByRole("button", { name: "Load S&P 500 Semiconductors & equipment" })).toHaveAttribute("aria-pressed", "true");
    expect(screen.queryByRole("button", { name: "Load S&P 500 Energy sector" })).not.toBeInTheDocument();

    fireEvent.change(screen.getByLabelText("Filter universes"), { target: { value: "energy" } });
    expect(screen.getByRole("button", { name: "Load S&P 500 Energy sector" })).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Load Popular US stocks sample" })).not.toBeInTheDocument();
  });

  it("creates, renames, moves and deletes folders, then refreshes", async () => {
    const onChanged = vi.fn();
    render(
      <UniverseTree universes={universes} folders={folders} selected="" onSelect={vi.fn()} editable onChanged={onChanged} />,
    );

    fireEvent.click(screen.getByRole("button", { name: "New folder" }));
    fireEvent.change(screen.getByLabelText("New folder name"), { target: { value: "  Mine  " } });
    fireEvent.click(screen.getByRole("button", { name: "Create" }));
    await waitFor(() => expect(createUniverseFolder).toHaveBeenCalledWith("Mine", null));
    await waitFor(() => expect(onChanged).toHaveBeenCalledTimes(1));

    const themes = screen.getByRole("button", { name: "Expand folder S&P 500 themes" }).closest("li")!;
    fireEvent.click(within(themes).getByText("Rename"));
    fireEvent.change(screen.getByLabelText("Rename folder S&P 500 themes"), { target: { value: "Industry themes" } });
    fireEvent.click(screen.getByRole("button", { name: "Save" }));
    await waitFor(() => expect(updateUniverseFolder).toHaveBeenCalledWith("themes", { name: "Industry themes" }));

    // A folder cannot be offered a destination inside itself.
    const moveThemes = screen.getByLabelText("Move folder S&P 500 themes to") as HTMLSelectElement;
    expect([...moveThemes.options].map((option) => option.textContent)).toEqual([
      "Top level",
      "S&P 500 sectors (GICS)",
    ]);
    fireEvent.change(moveThemes, { target: { value: "sectors" } });
    await waitFor(() => expect(updateUniverseFolder).toHaveBeenCalledWith("themes", { parent: "sectors" }));

    fireEvent.change(screen.getByLabelText("Move Popular US stocks sample to folder"), { target: { value: "tech" } });
    await waitFor(() => expect(moveUniverses).toHaveBeenCalledWith(["sp500-lite"], "tech"));

    fireEvent.click(screen.getByRole("button", { name: "Expand folder S&P 500 themes" }));
    const tech = screen.getByRole("button", { name: "Expand folder Technology" }).closest("li")!;
    fireEvent.click(within(tech).getByRole("button", { name: "Delete folder" }));
    expect(deleteUniverseFolder).not.toHaveBeenCalled();
    fireEvent.click(within(tech).getByRole("button", { name: /Confirm: remove folder, keep its 1 universe$/ }));
    await waitFor(() => expect(deleteUniverseFolder).toHaveBeenCalledWith("tech"));
  });

  it("shows API errors inline instead of losing the action", async () => {
    createUniverseFolder.mockRejectedValue(new Error("a folder named 'Mine' already exists here"));
    render(<UniverseTree universes={universes} folders={folders} selected="" onSelect={vi.fn()} editable />);
    fireEvent.click(screen.getByRole("button", { name: "New folder" }));
    fireEvent.change(screen.getByLabelText("New folder name"), { target: { value: "Mine" } });
    fireEvent.click(screen.getByRole("button", { name: "Create" }));
    expect(await screen.findByText("a folder named 'Mine' already exists here")).toBeInTheDocument();
  });

  it("derives read-only folders from folder_path when the folder API is unavailable", () => {
    const tree = buildUniverseTree(universes, null);
    expect(tree.topUniverses.map((item) => item.name)).toEqual(["sp500-lite"]);
    expect(tree.topFolders.map((id) => tree.nodes.get(id)!.folder.name)).toEqual([
      "S&P 500 sectors (GICS)",
      "S&P 500 themes",
    ]);
    render(<UniverseTree universes={universes} selected="" onSelect={vi.fn()} editable />);
    expect(screen.queryByRole("button", { name: "New folder" })).not.toBeInTheDocument();
  });
});

describe("UniversePicker", () => {
  it("opens the folder tree on demand and closes after a choice", () => {
    const onChange = vi.fn();
    render(<UniversePicker universes={universes} value="sp500-lite" onChange={onChange} />);
    const trigger = screen.getByRole("button", { name: "Universe" });
    expect(trigger).toHaveTextContent("Popular US stocks sample (1)");
    expect(screen.queryByTestId("universe-tree")).not.toBeInTheDocument();

    fireEvent.click(trigger);
    fireEvent.click(screen.getByRole("button", { name: "Expand folder S&P 500 sectors (GICS)" }));
    fireEvent.click(screen.getByRole("button", { name: "Use S&P 500 Energy sector" }));
    expect(onChange).toHaveBeenCalledWith("builtin-sp500-sector-energy");
    expect(screen.queryByTestId("universe-tree")).not.toBeInTheDocument();
  });
});
