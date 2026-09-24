import { useEffect, useRef, useState } from "react";

export function SaveResultDialog({ suggestedName, onSave, onClose }: {
  suggestedName: string; onSave: (name: string) => Promise<void>; onClose: () => void;
}) {
  const dialog = useRef<HTMLDialogElement>(null);
  const [name, setName] = useState(suggestedName);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  useEffect(() => {
    const element = dialog.current;
    if (element?.showModal) element.showModal();
    else element?.setAttribute("open", "");
    return () => element?.close?.();
  }, []);
  return <dialog ref={dialog} className="formula-dialog save-result-dialog" aria-labelledby="save-result-title" onCancel={onClose}>
    <form onSubmit={async event => {
      event.preventDefault(); setBusy(true); setError(null);
      try { await onSave(name.trim()); onClose(); }
      catch (reason) { setError(reason instanceof Error ? reason.message : String(reason)); }
      finally { setBusy(false); }
    }}>
      <h3 id="save-result-title">Save to Library</h3>
      <label className="field"><span>Result name</span><input autoFocus value={name} onChange={event => setName(event.target.value)} required disabled={busy}/></label>
      <p className="hint">The formula and its research context are preserved. Saving leaves the holdout locked.</p>
      {error && <p role="alert">{error}</p>}
      <div className="signals-toolbar"><button type="submit" disabled={busy || !name.trim()}>{busy ? "Saving…" : "Save result"}</button><button type="button" className="ghost" onClick={onClose} disabled={busy}>Cancel</button></div>
    </form>
  </dialog>;
}
