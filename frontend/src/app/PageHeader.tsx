import type { ReactNode } from "react";

export function PageHeader({
  title,
  description,
  eyebrow = "AlphaLineage",
  actions,
}: {
  title: string;
  description: string;
  eyebrow?: string;
  actions?: ReactNode;
}) {
  return (
    <header className="page-header">
      <div className="page-header__copy">
        <div className="view-tag">
          <span className="view-tag__mark" aria-hidden="true" />
          <span>{eyebrow}</span>
        </div>
        <h1>{title}</h1>
        <p>{description}</p>
      </div>
      {actions && <div className="page-header__actions">{actions}</div>}
      <div className="view-rule" aria-hidden="true" />
    </header>
  );
}
