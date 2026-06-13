"""P0-T5 - survivorship-bias audit report.

Free/prototype data sources often silently drop delisted names, which inflates
backtest results. This report makes the universe's delisted coverage explicit: it
enumerates active vs delisted members so a reader can judge whether the dataset is
survivorship-biased before trusting any downstream metric.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from alphalineage.data import paths
from alphalineage.data.universe import Universe

_DISCLAIMER = (
    "Prototype data: this sample universe is not a survivorship-bias-free index "
    "history. Real point-in-time constituents require a licensed data source."
)


def survivorship_report(universe: Universe) -> str:
    """Render a Markdown survivorship audit for ``universe``."""
    active = universe.active()
    delisted = universe.delisted()
    total = len(universe.memberships)
    generated = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%SZ")

    lines: list[str] = [
        f"# Survivorship audit - {universe.name}",
        "",
        f"_Generated {generated}_",
        "",
        f"- Total members: **{total}**",
        f"- Active: **{len(active)}**",
        f"- Delisted: **{len(delisted)}**",
        "",
        "## Delisted names",
        "",
    ]
    if delisted:
        lines.append("| Symbol | Entry | Exit |")
        lines.append("| --- | --- | --- |")
        for m in sorted(delisted, key=lambda x: x.symbol):
            lines.append(f"| {m.symbol} | {m.entry.date()} | {m.exit.date() if m.exit else ''} |")
    else:
        lines.append("_No delisted names recorded - this dataset is likely survivorship-biased._")
    lines += [
        "",
        "## Active names",
        "",
        ", ".join(sorted(m.symbol for m in active)) or "_none_",
        "",
        "---",
        "",
        f"> {_DISCLAIMER}",
        "",
    ]
    return "\n".join(lines)


def write_report(universe: Universe, path: Path | None = None) -> Path:
    """Write the survivorship report to ``data_cache/reports/`` (or ``path``)."""
    target = path or (paths.reports_dir() / f"survivorship_{universe.name}.md")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(survivorship_report(universe), encoding="utf-8")
    return target
