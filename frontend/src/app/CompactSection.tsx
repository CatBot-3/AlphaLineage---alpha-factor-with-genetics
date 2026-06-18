import type { ReactNode } from "react";
import { useId, useState } from "react";

export function CompactSection({
  title,
  summary,
  children,
  defaultOpen = false,
  className = "",
}: {
  title: string;
  summary?: ReactNode;
  children: ReactNode;
  defaultOpen?: boolean;
  className?: string;
}) {
  const [open, setOpen] = useState(defaultOpen);
  const bodyId = useId();

  return (
    <section className={`compact-section ${className}`.trim()} data-open={open}>
      <button
        type="button"
        className="compact-section__trigger"
        aria-expanded={open}
        aria-controls={bodyId}
        onClick={() => setOpen((value) => !value)}
      >
        <span className="compact-section__chevron" aria-hidden="true">
          {open ? "v" : ">"}
        </span>
        <span className="compact-section__title">{title}</span>
        {summary && <span className="compact-section__summary">{summary}</span>}
      </button>
      {open && (
        <div id={bodyId} className="compact-section__body">
          {children}
        </div>
      )}
    </section>
  );
}
