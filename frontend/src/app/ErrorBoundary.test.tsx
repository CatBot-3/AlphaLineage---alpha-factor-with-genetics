import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { ErrorBoundary } from "./ErrorBoundary";

function Boom(): never {
  throw new Error("kaboom");
}

describe("ErrorBoundary", () => {
  it("shows a contained fallback instead of blanking when a child throws", () => {
    vi.spyOn(console, "error").mockImplementation(() => undefined); // silence React's error log
    render(
      <ErrorBoundary>
        <Boom />
      </ErrorBoundary>,
    );
    expect(screen.getByTestId("error-boundary")).toHaveTextContent(/went wrong/i);
    expect(screen.getByTestId("error-boundary")).toHaveTextContent("kaboom");
  });

  it("renders children when nothing throws", () => {
    render(
      <ErrorBoundary>
        <p>all good</p>
      </ErrorBoundary>,
    );
    expect(screen.getByText("all good")).toBeInTheDocument();
    expect(screen.queryByTestId("error-boundary")).not.toBeInTheDocument();
  });
});
