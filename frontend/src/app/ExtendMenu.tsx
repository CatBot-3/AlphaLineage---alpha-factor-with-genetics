// The "Extend" nav item as a dropdown: clicking it drops a menu of the three Extend pages.
// Mirrors the SettingsMenu popover pattern (open state, outside-click / Escape close, aria).

import { useEffect, useRef, useState } from "react";
import type { ExtendPage } from "../extend/ExtendPanel";

const PAGES: Array<{ id: ExtendPage; label: string }> = [
  { id: "universe", label: "Universe Editor" },
  { id: "sync", label: "Sync Data" },
  { id: "formula", label: "Formula Editor" },
];

export function ExtendMenu({
  current,
  disabled,
  onSelect,
}: {
  current: boolean;
  disabled?: boolean;
  onSelect: (page: ExtendPage) => void;
}) {
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!open) return;
    function onDown(e: MouseEvent) {
      if (ref.current && !ref.current.contains(e.target as globalThis.Node)) setOpen(false);
    }
    function onKey(e: KeyboardEvent) {
      if (e.key === "Escape") setOpen(false);
    }
    document.addEventListener("mousedown", onDown);
    document.addEventListener("keydown", onKey);
    return () => {
      document.removeEventListener("mousedown", onDown);
      document.removeEventListener("keydown", onKey);
    };
  }, [open]);

  return (
    <div className="nav-dropdown" ref={ref}>
      <button
        type="button"
        className="nav__link"
        aria-haspopup="menu"
        aria-expanded={open}
        aria-current={current ? "page" : undefined}
        aria-disabled={disabled ? "true" : undefined}
        title={
          disabled ? "Available as a locally saved draft in static demo mode" : undefined
        }
        onClick={() => setOpen((v) => !v)}
      >
        Extend
      </button>
      {open && (
        <div className="nav-dropdown__menu" role="menu" data-testid="extend-menu">
          {PAGES.map((p) => (
            <button
              key={p.id}
              type="button"
              role="menuitem"
              onClick={() => {
                setOpen(false);
                onSelect(p.id);
              }}
            >
              {p.label}
            </button>
          ))}
        </div>
      )}
    </div>
  );
}
