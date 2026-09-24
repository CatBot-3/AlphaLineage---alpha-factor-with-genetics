import { fireEvent, render, screen, within } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { AppShell } from "./AppShell";

function renderShell(mode: "demo" | "app" = "demo") {
  const onTabChange = vi.fn();
  const onSelectExtendPage = vi.fn();
  const onQuit = vi.fn();
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
      onQuit={onQuit}
      onSelectExtendPage={onSelectExtendPage}
    >
      <div>content</div>
    </AppShell>,
  );
  return { onTabChange, onSelectExtendPage, onQuit };
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
      "Results",
      "Library",
      "Signals",
      "AgentBeta", // the Beta mark is a child span, so it joins the tab's text content
      "Build & Data",
    ]);
  });

  it("marks Agent as beta in the navigation", () => {
    renderShell("app");
    const agent = screen.getByRole("button", { name: /Agent/ });
    expect(within(agent).getByText("Beta")).toBeInTheDocument();
    expect(agent).toHaveClass("nav__link--beta");
  });

  it("opens the Extend dropdown and selects a sub-page instead of a plain tab switch", () => {
    const { onSelectExtendPage } = renderShell("app");
    const extend = screen.getByRole("button", { name: "Build & Data" });

    fireEvent.click(extend); // opens the dropdown rather than navigating
    expect(screen.getByTestId("extend-menu")).toBeInTheDocument();
    expect(screen.queryByRole("menuitem", { name: "Sync Data" })).not.toBeInTheDocument();
    fireEvent.click(screen.getByRole("menuitem", { name: "Formula Builder" }));
    expect(onSelectExtendPage).toHaveBeenCalledWith("formula");
  });

  it("keeps the local Extend workspace accessible in static demo mode", () => {
    const { onSelectExtendPage } = renderShell("demo");
    const extend = screen.getByRole("button", { name: "Build & Data" });

    expect(extend).not.toHaveAttribute("aria-disabled");
    fireEvent.click(extend);
    fireEvent.click(screen.getByRole("menuitem", { name: "Formula Builder" }));
    expect(onSelectExtendPage).toHaveBeenCalledWith("formula");
  });

  it("uses an accessible gear and keeps Quit as a first-level app action", () => {
    const { onQuit } = renderShell("app");

    expect(screen.getByRole("button", { name: "Open settings" })).toHaveAttribute(
      "aria-haspopup",
      "dialog",
    );
    fireEvent.click(screen.getByRole("button", { name: "Quit" }));
    expect(onQuit).toHaveBeenCalledOnce();
  });

  it("does not show a process Quit action in static demo mode", () => {
    renderShell("demo");
    expect(screen.queryByRole("button", { name: "Quit" })).not.toBeInTheDocument();
  });
});
