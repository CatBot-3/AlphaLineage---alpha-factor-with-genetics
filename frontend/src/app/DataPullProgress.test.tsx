import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { DataPullProgress } from "./DataPullProgress";

describe("DataPullProgress", () => {
  it("renders nothing when there is no progress", () => {
    render(<DataPullProgress progress={null} />);
    expect(screen.queryByTestId("data-pull-progress")).not.toBeInTheDocument();
  });

  it("renders nothing once the pull has finished", () => {
    render(<DataPullProgress progress={{ done: 3, total: 3, current_symbol: "AAPL" }} />);
    expect(screen.queryByTestId("data-pull-progress")).not.toBeInTheDocument();
  });

  it("shows the current symbol and a proportional fill while in progress", () => {
    render(<DataPullProgress progress={{ done: 1, total: 4, current_symbol: "AAPL" }} />);
    const bar = screen.getByTestId("data-pull-progress");
    expect(bar).toHaveTextContent("AAPL");
    expect(bar).toHaveTextContent("(1/4)");
    expect(screen.getByRole("progressbar")).toHaveAttribute("aria-valuenow", "25");
  });
});
