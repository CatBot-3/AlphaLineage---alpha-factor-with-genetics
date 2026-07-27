"""P0-T5 - survivorship-bias audit report.

Free/prototype constituent sources often omit historical removals, which inflates
backtest results. A declared universe exit and a security delisting are different
events, so this report describes only the membership evidence actually stored.
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
    exited = universe.exited()
    total = len(universe.memberships)
    generated = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%SZ")

    lines: list[str] = [
        f"# Survivorship audit - {universe.name}",
        "",
        f"_Generated {generated}_",
        "",
        f"- Total members: **{total}**",
        f"- Open membership intervals: **{len(active)}**",
        f"- Declared membership exits: **{len(exited)}**",
        "",
        (
            "_A membership exit means the symbol left this universe; it does not establish "
            "that the security was delisted._"
        ),
        "",
        "## Declared membership exits",
        "",
    ]
    if exited:
        lines.append("| Symbol | Entry | Exit |")
        lines.append("| --- | --- | --- |")
        for m in sorted(exited, key=lambda x: x.symbol):
            lines.append(f"| {m.symbol} | {m.entry.date()} | {m.exit.date() if m.exit else ''} |")
    else:
        lines.append(
            "_No historical membership exits are recorded - this dataset is likely "
            "survivorship-biased._"
        )
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
