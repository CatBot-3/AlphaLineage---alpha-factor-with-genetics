import {
  useEffect,
  useRef,
  useState,
  type CSSProperties,
  type KeyboardEvent,
  type PointerEvent,
  type ReactNode,
} from "react";

export interface GraphWorkspacePane {
  id: string;
  label: string;
  content: ReactNode;
}

interface DragState {
  side: "left" | "right";
  startX: number;
  startWidth: number;
  startCollapsed: boolean;
}

const MIN_CENTER_WIDTH = 320;
const KEYBOARD_STEP = 24;
const COLLAPSE_DISTANCE = 42;

function storedWidth(key: string | undefined, fallback: number): number {
  if (!key || typeof window === "undefined") return fallback;
  try {
    const value = Number(window.localStorage.getItem(key));
    return Number.isFinite(value) && value > 0 ? value : fallback;
  } catch {
    return fallback;
  }
}

function storedCollapsed(key: string | undefined): boolean {
  if (!key || typeof window === "undefined") return false;
  try {
    return window.localStorage.getItem(key) === "true";
  } catch {
    return false;
  }
}

function clamp(value: number, minimum: number, maximum: number): number {
  return Math.min(Math.max(value, minimum), Math.max(minimum, maximum));
}

/**
 * A keyboard-accessible, resizable shell shared by graph-heavy read-only views.
 * Pane widths are presentation state only and never enter session/workspace data.
 */
export function ResizableGraphWorkspace({
  left,
  center,
  right,
  storageKey,
  leftDefault = 280,
  rightDefault = 340,
  leftMin = 220,
  rightMin = 260,
  paneMax = 520,
  className = "",
}: {
  left?: GraphWorkspacePane;
  center: GraphWorkspacePane;
  right?: GraphWorkspacePane;
  storageKey?: string;
  leftDefault?: number;
  rightDefault?: number;
  leftMin?: number;
  rightMin?: number;
  paneMax?: number;
  className?: string;
}) {
  const rootRef = useRef<HTMLDivElement | null>(null);
  const [leftWidth, setLeftWidth] = useState(() =>
    storedWidth(storageKey ? `${storageKey}:left` : undefined, leftDefault),
  );
  const [rightWidth, setRightWidth] = useState(() =>
    storedWidth(storageKey ? `${storageKey}:right` : undefined, rightDefault),
  );
  const [leftCollapsed, setLeftCollapsed] = useState(() =>
    storedCollapsed(storageKey ? `${storageKey}:left-collapsed` : undefined),
  );
  const [rightCollapsed, setRightCollapsed] = useState(() =>
    storedCollapsed(storageKey ? `${storageKey}:right-collapsed` : undefined),
  );
  const [drag, setDrag] = useState<DragState | null>(null);

  useEffect(() => {
    if (typeof window.matchMedia !== "function") return;
    const narrow = window.matchMedia("(max-width: 900px)");
    const expandStackedPanes = () => {
      if (!narrow.matches) return;
      if (left) setLeftCollapsed(false);
      if (right) setRightCollapsed(false);
    };
    expandStackedPanes();
    narrow.addEventListener?.("change", expandStackedPanes);
    return () => narrow.removeEventListener?.("change", expandStackedPanes);
  }, [Boolean(left), Boolean(right)]);

  function maximumFor(side: "left" | "right"): number {
    const measured = rootRef.current?.getBoundingClientRect().width ?? 0;
    const total = measured > 0 ? measured : 1_280;
    const other = side === "left"
      ? (right && !rightCollapsed ? rightWidth : 0)
      : (left && !leftCollapsed ? leftWidth : 0);
    const separators = (left ? 8 : 0) + (right ? 8 : 0);
    const minimum = side === "left" ? leftMin : rightMin;
    return Math.max(minimum, Math.min(paneMax, total - other - separators - MIN_CENTER_WIDTH));
  }

  function setWidth(side: "left" | "right", requested: number) {
    if (side === "left") {
      setLeftWidth(clamp(requested, leftMin, maximumFor("left")));
    } else {
      setRightWidth(clamp(requested, rightMin, maximumFor("right")));
    }
  }

  function setCollapsed(side: "left" | "right", collapsed: boolean) {
    if (side === "left") setLeftCollapsed(collapsed);
    else setRightCollapsed(collapsed);
  }

  useEffect(() => {
    if (!drag) return;
    const previousCursor = document.body.style.cursor;
    const previousSelection = document.body.style.userSelect;
    document.body.style.cursor = "col-resize";
    document.body.style.userSelect = "none";
    const move = (event: globalThis.PointerEvent) => {
      const pointerDelta = event.clientX - drag.startX;
      const openingDistance = drag.side === "left" ? pointerDelta : -pointerDelta;
      const minimum = drag.side === "left" ? leftMin : rightMin;
      if (drag.startCollapsed) {
        if (openingDistance > COLLAPSE_DISTANCE) {
          setCollapsed(drag.side, false);
          setWidth(drag.side, minimum + openingDistance - COLLAPSE_DISTANCE);
        }
        return;
      }
      const width = drag.side === "left"
        ? drag.startWidth + pointerDelta
        : drag.startWidth - pointerDelta;
      if (width < minimum - COLLAPSE_DISTANCE) {
        setCollapsed(drag.side, true);
        return;
      }
      setCollapsed(drag.side, false);
      setWidth(drag.side, width);
    };
    const stop = () => setDrag(null);
    window.addEventListener("pointermove", move);
    window.addEventListener("pointerup", stop, { once: true });
    window.addEventListener("pointercancel", stop, { once: true });
    return () => {
      window.removeEventListener("pointermove", move);
      window.removeEventListener("pointerup", stop);
      window.removeEventListener("pointercancel", stop);
      document.body.style.cursor = previousCursor;
      document.body.style.userSelect = previousSelection;
    };
  }, [
    drag,
    leftCollapsed,
    leftMin,
    paneMax,
    rightCollapsed,
    rightMin,
    leftWidth,
    rightWidth,
  ]);

  useEffect(() => {
    if (!storageKey) return;
    try {
      window.localStorage.setItem(`${storageKey}:left`, String(Math.round(leftWidth)));
      window.localStorage.setItem(`${storageKey}:right`, String(Math.round(rightWidth)));
      window.localStorage.setItem(`${storageKey}:left-collapsed`, String(leftCollapsed));
      window.localStorage.setItem(`${storageKey}:right-collapsed`, String(rightCollapsed));
    } catch {
      // Private browsing or a full storage quota must not disable resizing.
    }
  }, [leftCollapsed, leftWidth, rightCollapsed, rightWidth, storageKey]);

  function startResize(side: "left" | "right", event: PointerEvent<HTMLDivElement>) {
    if (event.button !== 0) return;
    event.preventDefault();
    setDrag({
      side,
      startX: event.clientX,
      startWidth: side === "left"
        ? (leftCollapsed ? 0 : leftWidth)
        : (rightCollapsed ? 0 : rightWidth),
      startCollapsed: side === "left" ? leftCollapsed : rightCollapsed,
    });
  }

  function resizeWithKeyboard(side: "left" | "right", event: KeyboardEvent<HTMLDivElement>) {
    const collapsed = side === "left" ? leftCollapsed : rightCollapsed;
    const current = side === "left" ? leftWidth : rightWidth;
    const physicalDirection = side === "left" ? 1 : -1;
    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      setCollapsed(side, !collapsed);
    } else if (event.key === "ArrowLeft") {
      event.preventDefault();
      if (collapsed) setCollapsed(side, false);
      setWidth(side, current - KEYBOARD_STEP * physicalDirection);
    } else if (event.key === "ArrowRight") {
      event.preventDefault();
      if (collapsed) setCollapsed(side, false);
      setWidth(side, current + KEYBOARD_STEP * physicalDirection);
    } else if (event.key === "Home") {
      event.preventDefault();
      setCollapsed(side, true);
    } else if (event.key === "End") {
      event.preventDefault();
      setCollapsed(side, false);
      setWidth(side, maximumFor(side));
    }
  }

  const classes = [
    "graph-workspace",
    left ? "graph-workspace--has-left" : "",
    right ? "graph-workspace--has-right" : "",
    leftCollapsed ? "is-left-collapsed" : "",
    rightCollapsed ? "is-right-collapsed" : "",
    drag ? `is-resizing-${drag.side}` : "",
    className,
  ].filter(Boolean).join(" ");

  return (
    <div
      ref={rootRef}
      className={classes}
      style={{
        "--graph-left-width": `${leftCollapsed ? 0 : Math.round(leftWidth)}px`,
        "--graph-right-width": `${rightCollapsed ? 0 : Math.round(rightWidth)}px`,
      } as CSSProperties}
    >
      {left && (
        <aside
          id={left.id}
          className="graph-workspace__pane graph-workspace__pane--left"
          aria-label={left.label}
          aria-hidden={leftCollapsed}
        >
          {left.content}
        </aside>
      )}
      {left && (
        <div
          className="graph-workspace__resizer graph-workspace__resizer--left"
          role="separator"
          tabIndex={0}
          aria-label={`Resize ${left.label}`}
          aria-controls={left.id}
          aria-orientation="vertical"
          aria-valuemin={0}
          aria-valuemax={Math.round(maximumFor("left"))}
          aria-valuenow={leftCollapsed ? 0 : Math.round(leftWidth)}
          aria-valuetext={leftCollapsed ? "Collapsed" : `${Math.round(leftWidth)} pixels`}
          aria-keyshortcuts="Enter Space ArrowLeft ArrowRight Home End"
          title="Drag to resize. Drag past the minimum to collapse; press Enter to toggle."
          onPointerDown={(event) => startResize("left", event)}
          onKeyDown={(event) => resizeWithKeyboard("left", event)}
          onDoubleClick={() => setCollapsed("left", !leftCollapsed)}
        />
      )}
      <main id={center.id} className="graph-workspace__center" aria-label={center.label}>
        {center.content}
      </main>
      {right && (
        <div
          className="graph-workspace__resizer graph-workspace__resizer--right"
          role="separator"
          tabIndex={0}
          aria-label={`Resize ${right.label}`}
          aria-controls={right.id}
          aria-orientation="vertical"
          aria-valuemin={0}
          aria-valuemax={Math.round(maximumFor("right"))}
          aria-valuenow={rightCollapsed ? 0 : Math.round(rightWidth)}
          aria-valuetext={rightCollapsed ? "Collapsed" : `${Math.round(rightWidth)} pixels`}
          aria-keyshortcuts="Enter Space ArrowLeft ArrowRight Home End"
          title="Drag to resize. Drag past the minimum to collapse; press Enter to toggle."
          onPointerDown={(event) => startResize("right", event)}
          onKeyDown={(event) => resizeWithKeyboard("right", event)}
          onDoubleClick={() => setCollapsed("right", !rightCollapsed)}
        />
      )}
      {right && (
        <aside
          id={right.id}
          className="graph-workspace__pane graph-workspace__pane--right"
          aria-label={right.label}
          aria-hidden={rightCollapsed}
        >
          {right.content}
        </aside>
      )}
    </div>
  );
}
