import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import type { FormulaSpec, PrimitiveInfo } from "../api/types";
import type { RunRequestForm } from "./RunConfigForm";

const { universeCoverage, formulaList, primitiveList } = vi.hoisted(() => ({
  universeCoverage: vi.fn((_name?: string, _asOf?: string) =>
    Promise.resolve({
      as_of: "2026-07-15",
      eligible_symbols: ["AAPL"] as string[],
      cached_symbols: ["AAPL"] as string[],
      missing_symbols: [] as string[],
      incomplete_symbols: [] as string[],
      complete: true,
    }),
  ),
  formulaList: vi.fn<() => Promise<FormulaSpec[]>>(() => Promise.resolve([])),
  primitiveList: vi.fn<() => Promise<PrimitiveInfo[]>>(() =>
    Promise.resolve([
      { name: "ts_mean", kind: "operator", arg_types: ["series", "window"], out_type: "series", user: false, category: "time_series" },
      { name: "gt", kind: "operator", arg_types: ["series", "series"], out_type: "bool", user: false, category: "condition" },
      { name: "close", kind: "operand", arg_types: [], out_type: "series", user: false, category: "data" },
    ])),
}));

vi.mock("../api/client", () => ({
  listUniverses: () => Promise.resolve([]),
  getUniverseCoverage: (name: string, asOf: string) => universeCoverage(name, asOf),
  listFormulaResults: () =>
    Promise.resolve([
      {
        id: "result-1",
        name: "Momentum result",
        saved_at: "2026-07-14T00:00:00Z",
        tree: { name: "close" },
      },
    ]),
  getPrimitives: () => primitiveList(),
  listFormulas: () => formulaList(),
  getTrainingCapabilities: () =>
    Promise.resolve({
      default_profile: "auto",
      detected_cpus: 20,
      worker_capacity: 19,
      available_memory_bytes: 16_000_000_000,
      memory_budget_bytes: 4_294_967_296,
      cpu_budget_percent_min: 10,
      cpu_budget_percent_max: 100,
      evaluator: "auto",
      cpp_available: true,
      fallback_reason: null,
      profiles: {
        light: {
          profile: "light",
          percent: 25,
          requested_workers: 5,
          effective_workers: 5,
          workers: 5,
          worker_capacity: 19,
          run_memory_budget_bytes: 1_130_254_625,
        },
        auto: {
          profile: "auto",
          percent: 50,
          requested_workers: 10,
          effective_workers: 10,
          workers: 10,
          worker_capacity: 19,
          run_memory_budget_bytes: 2_260_509_250,
        },
        maximum: {
          profile: "maximum",
          percent: 100,
          requested_workers: 19,
          effective_workers: 19,
          workers: 19,
          worker_capacity: 19,
          run_memory_budget_bytes: 4_294_967_296,
        },
      },
    }),
}));

import { RunConfigForm } from "./RunConfigForm";

afterEach(() => vi.clearAllMocks());

describe("RunConfigForm function space", () => {
  it("labels compatible training seeds as Formula Results", async () => {
    render(<RunConfigForm onStart={vi.fn()} />);

    expect(
      await screen.findByText("Seed training from Formula Results (optional)"),
    ).toBeInTheDocument();
    expect(screen.queryByText("Seed from saved factors (optional)")).not.toBeInTheDocument();
  });

  it("lists operator categories and fires the Edit-functions callback", async () => {
    const onOpenFormulaEditor = vi.fn();
    render(<RunConfigForm onStart={vi.fn()} onOpenFormulaEditor={onOpenFormulaEditor} />);

    expect(await screen.findByTestId("function-cat-time_series")).toBeInTheDocument();
    expect(screen.getByTestId("function-cat-condition")).toBeInTheDocument();
    // operands (data) are not operators, so they don't appear as a toggle
    expect(screen.queryByTestId("function-cat-data")).not.toBeInTheDocument();

    fireEvent.click(screen.getByTestId("edit-functions"));
    expect(onOpenFormulaEditor).toHaveBeenCalled();
  });

  it("excludes the condition category from enabled_categories by default", async () => {
    const onStart = vi.fn();
    render(<RunConfigForm onStart={onStart} />);
    await screen.findByTestId("function-cat-condition");

    fireEvent.submit(screen.getByTestId("run-config-form"));
    const req = onStart.mock.calls[0][0] as RunRequestForm;
    expect(req.config.enabled_categories).toContain("time_series");
    expect(req.config.enabled_categories).not.toContain("condition");
  });

  it("includes condition once the user enables it", async () => {
    const onStart = vi.fn();
    render(<RunConfigForm onStart={onStart} />);
    await screen.findByTestId("function-cat-condition");

    fireEvent.click(screen.getByLabelText("enable condition"));
    fireEvent.submit(screen.getByTestId("run-config-form"));
    const req = onStart.mock.calls[0][0] as RunRequestForm;
    expect(req.config.enabled_categories).toContain("condition");
  });

  it("selects managed formulas individually while preserving their tuning contract", async () => {
    formulaList.mockResolvedValueOnce([
      {
        name: "ta_macd_histogram",
        display_name: "MACD",
        description: "Canonical MACD.",
        arg_types: ["window", "window"],
        inputs: [
          { name: "fast", type: "window", description: "Fast EMA.", default: 12, role: "parameter", tuning: { enabled: true, min: 8, max: 20, step: 1, radius: 1 } },
          { name: "slow", type: "window", description: "Slow EMA.", default: 26, role: "parameter", tuning: { enabled: true, min: 20, max: 40, step: 1, radius: 1 } },
        ],
        constraints: [{ left: "fast", operator: "lt", right: "slow" }],
        out_type: "series",
        body: { name: "close" },
        category: "technical_indicators",
        origin: "catalog_formula",
        registered: true,
      },
    ]);
    primitiveList.mockResolvedValueOnce([
      { name: "ta_macd_histogram", logical_name: "ta_macd_histogram", display_name: "MACD", kind: "operator", arg_types: ["window", "window"], out_type: "series", user: true, origin: "catalog_formula", category: "technical_indicators" },
      { name: "ts_mean", kind: "operator", arg_types: ["series", "window"], out_type: "series", user: false, category: "time_series" },
    ]);

    const onStart = vi.fn();
    render(<RunConfigForm onStart={onStart} />);
    await screen.findByTestId("function-cat-technical_indicators");
    expect(screen.getByText(/fast 12 \(8–20, step 1\)/)).toBeInTheDocument();
    expect(screen.getByText(/fast < slow/)).toBeInTheDocument();

    fireEvent.click(screen.getByLabelText("enable technical_indicators"));
    fireEvent.click(screen.getByLabelText("enable formula ta_macd_histogram"));
    fireEvent.submit(screen.getByTestId("run-config-form"));

    const request = onStart.mock.calls[0][0] as RunRequestForm;
    expect(request.config.enabled_categories).toContain("technical_indicators");
    expect(request.config.enabled_formula_names).toEqual([]);
  });

  it("submits the explicit research as-of date", async () => {
    const onStart = vi.fn();
    render(<RunConfigForm onStart={onStart} />);
    await screen.findByTestId("function-cat-time_series");
    fireEvent.change(screen.getByLabelText("As of date"), { target: { value: "2024-12-31" } });
    await waitFor(() => expect(screen.getByRole("button", { name: "Start training" })).toBeEnabled());
    fireEvent.submit(screen.getByTestId("run-config-form"));
    const req = onStart.mock.calls[0][0] as RunRequestForm;
    expect(req.as_of).toBe("2024-12-31");
  });

  it("blocks incomplete universes and offers Data Sync", async () => {
    universeCoverage.mockResolvedValueOnce({
      as_of: "2026-07-15",
      eligible_symbols: ["AAPL", "MSFT"],
      cached_symbols: ["AAPL"],
      missing_symbols: ["MSFT"],
      incomplete_symbols: ["MSFT"],
      complete: false,
    });
    const onStart = vi.fn();
    const onOpenDataSync = vi.fn();
    render(<RunConfigForm onStart={onStart} onOpenDataSync={onOpenDataSync} />);

    expect(await screen.findByText("Price history needs attention")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Start training" })).toBeDisabled();
    fireEvent.click(screen.getByRole("button", { name: "Data Sync" }));
    expect(onOpenDataSync).toHaveBeenCalledWith("sp500-lite");
    fireEvent.submit(screen.getByTestId("run-config-form"));
    expect(onStart).not.toHaveBeenCalled();
  });

  it("defaults to visible device-relative Auto resources and supports Custom", async () => {
    const onStart = vi.fn();
    render(<RunConfigForm onStart={onStart} />);

    expect(await screen.findByTestId("resource-summary")).toHaveTextContent(
      /50%: 10 of 20 CPUs \(one reserved\)/i,
    );
    fireEvent.change(screen.getByLabelText("Resource profile"), {
      target: { value: "custom" },
    });
    fireEvent.change(screen.getByLabelText("Custom CPU percentage"), {
      target: { value: "70" },
    });
    fireEvent.submit(screen.getByTestId("run-config-form"));

    const req = onStart.mock.calls[0][0] as RunRequestForm;
    expect(req.resources).toEqual({ profile: "custom", cpu_budget_percent: 70 });
  });

  // The dice is a convenience, not an optimisation step: re-rolling the seed until the search
  // looks good is seed-hacking, which is why the agent is refused this key and why this stays a
  // deliberate, human click rather than anything automatic.
  it("re-rolls the seed on demand and submits the new value", async () => {
    const onStart = vi.fn();
    render(<RunConfigForm onStart={onStart} />);

    const seed = (await screen.findByLabelText("Seed")) as HTMLInputElement;
    const before = seed.value;

    const random = vi.spyOn(Math, "random").mockReturnValue(0.123456);
    fireEvent.click(screen.getByTestId("randomize-seed"));
    random.mockRestore();

    expect(seed.value).toBe("123456");
    expect(seed.value).not.toBe(before);

    fireEvent.submit(screen.getByTestId("run-config-form"));
    const req = onStart.mock.calls[0][0] as RunRequestForm;
    expect(req.config.seed).toBe(123456);
  });
});

