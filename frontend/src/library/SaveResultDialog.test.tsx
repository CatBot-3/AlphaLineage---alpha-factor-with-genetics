import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { SaveResultDialog } from "./SaveResultDialog";

describe("SaveResultDialog", () => {
  it("prefills the result name and waits for the authoritative save", async () => {
    let finish!: () => void;
    const onSave = vi.fn(() => new Promise<void>(resolve => { finish = resolve; }));
    const onClose = vi.fn();
    render(<SaveResultDialog suggestedName="Momentum · Round 2" onSave={onSave} onClose={onClose}/>);
    const name = screen.getByLabelText("Result name");
    expect(name).toHaveValue("Momentum · Round 2");
    fireEvent.change(name, { target: { value: "  Daily momentum  " } });
    fireEvent.click(screen.getByRole("button", { name: "Save result" }));
    expect(onSave).toHaveBeenCalledWith("Daily momentum");
    expect(screen.getByRole("button", { name: "Saving…" })).toBeDisabled();
    expect(onClose).not.toHaveBeenCalled();
    finish();
    await waitFor(() => expect(onClose).toHaveBeenCalledOnce());
  });

  it("keeps the user's name and offers retry after a failed save", async () => {
    const onSave = vi.fn().mockRejectedValueOnce(new Error("Storage is unavailable"))
      .mockResolvedValueOnce(undefined);
    const onClose = vi.fn();
    render(<SaveResultDialog suggestedName="Round 1" onSave={onSave} onClose={onClose}/>);
    fireEvent.change(screen.getByLabelText("Result name"), { target: { value: "My formula" } });
    fireEvent.click(screen.getByRole("button", { name: "Save result" }));
    expect(await screen.findByRole("alert")).toHaveTextContent("Storage is unavailable");
    expect(screen.getByLabelText("Result name")).toHaveValue("My formula");
    expect(onClose).not.toHaveBeenCalled();
    fireEvent.click(screen.getByRole("button", { name: "Save result" }));
    await waitFor(() => expect(onClose).toHaveBeenCalledOnce());
  });
});
