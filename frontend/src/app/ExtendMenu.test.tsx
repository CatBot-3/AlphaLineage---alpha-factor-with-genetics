import { fireEvent, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { ExtendMenu } from "./ExtendMenu";

afterEach(() => vi.clearAllMocks());

describe("ExtendMenu", () => {
  it("opens on click and fires onSelect for a chosen page", () => {
    const onSelect = vi.fn();
    render(<ExtendMenu current={false} onSelect={onSelect} />);

    expect(screen.queryByTestId("extend-menu")).not.toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Extend" }));
    expect(screen.getByTestId("extend-menu")).toBeInTheDocument();

    fireEvent.click(screen.getByRole("menuitem", { name: "Universe Editor" }));
    expect(onSelect).toHaveBeenCalledWith("universe");
    // selecting closes the menu
    expect(screen.queryByTestId("extend-menu")).not.toBeInTheDocument();
  });

  it("closes on Escape", () => {
    render(<ExtendMenu current={false} onSelect={vi.fn()} />);
    fireEvent.click(screen.getByRole("button", { name: "Extend" }));
    expect(screen.getByTestId("extend-menu")).toBeInTheDocument();
    fireEvent.keyDown(document, { key: "Escape" });
    expect(screen.queryByTestId("extend-menu")).not.toBeInTheDocument();
  });
});
