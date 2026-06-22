import { fireEvent, render, screen, within } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { AppShell } from "./AppShell";

function renderShell(mode: "demo" | "app" = "demo") {
  const onTabChange = vi.fn();
  const onSelectExtendPage = vi.fn();
  render(
    <AppShell
      mode={mode}
      tab="dashboard"
      onTabChange={onTabChange}
      onRefreshRun={vi.fn()}
      onSaveLocal={vi.fn()}
      onLoadLocal={vi.fn()}
      onSaveBackend={vi.fn()}
      onLoadBackend={vi.fn()}
      onQuit={vi.fn()}
      onSelectExtendPage={onSelectExtendPage}
    >
      <div>content</div>
    </AppShell>,
  );
  return { onTabChange, onSelectExtendPage };
}

describe("AppShell", () => {
  it("renders the same primary navigation in demo and app modes", () => {
    renderShell("demo");
    const demoLabels = within(screen.getByTestId("main-nav"))
      .getAllByRole("button")
      .map((button) => button.textContent);

    renderShell("app");
    const allNavs = screen.getAllByTestId("main-nav");
    const appLabels = within(allNavs[1])
      .getAllByRole("button")
      .map((button) => button.textContent);

    expect(appLabels).toEqual(demoLabels);
    expect(demoLabels).toEqual([
      "Train",
      "Metrics",
      "Best factor",
      "Genealogy",
      "Library",
      "Extend",
    ]);
  });

  it("opens the Extend dropdown and selects a sub-page instead of a plain tab switch", () => {
    const { onSelectExtendPage } = renderShell("app");
    const extend = screen.getByRole("button", { name: "Extend" });

    fireEvent.click(extend); // opens the dropdown rather than navigating
    expect(screen.getByTestId("extend-menu")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("menuitem", { name: "Formula Editor" }));
    expect(onSelectExtendPage).toHaveBeenCalledWith("formula");
  });

  it("marks the Extend dropdown read-only in static demo mode", () => {
    renderShell("demo");
    expect(screen.getByRole("button", { name: "Extend" })).toHaveAttribute("aria-disabled", "true");
  });
});
