import { fireEvent, render, screen, within } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { AppShell, type Tab } from "./AppShell";

function renderShell(mode: "demo" | "app" = "demo") {
  const onTabChange = vi.fn();
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
    >
      <div>content</div>
    </AppShell>,
  );
  return onTabChange;
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
    expect(demoLabels).toEqual(["Metrics", "Best factor", "Genealogy", "Extend"]);
  });

  it("marks backend tools as read-only in static demo mode", () => {
    const onTabChange = renderShell("demo");
    const extend = screen.getByRole("button", { name: "Extend" });

    expect(extend).toHaveAttribute("aria-disabled", "true");
    fireEvent.click(extend);
    expect(onTabChange).toHaveBeenCalledWith("extend" satisfies Tab);
  });
});
