"""Small markdown report builder shared by the Phase 0 and labels reports."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _one_per(n: int, k: int) -> str:
    return f"1 per {n / k:,.0f}" if k else "none"


def _fmt(value) -> str:
    if isinstance(value, float | np.floating):
        if np.isnan(value):
            return "-"
        return f"{value:,.4g}" if abs(value) < 1000 else f"{value:,.0f}"
    if isinstance(value, int | np.integer):
        return f"{value:,}"
    return str(value)


class Report:
    def __init__(self, title: str = "Report") -> None:
        self.lines = [f"# {title}"]

    def section(self, title: str) -> None:
        self.lines += ["", f"## {title}", ""]

    def item(self, key: str, value) -> None:
        self.lines.append(f"- {key}: {_fmt(value)}")

    def note(self, text: str) -> None:
        self.lines += ["", text, ""]

    def table(self, df: pd.DataFrame) -> None:
        cols = list(df.columns)
        self.lines.append("| " + " | ".join(map(str, cols)) + " |")
        self.lines.append("|" + "---|" * len(cols))
        for row in df.itertuples(index=False):
            self.lines.append("| " + " | ".join(_fmt(v) for v in row) + " |")
        self.lines.append("")

    def text(self) -> str:
        out: list[str] = []
        for line in self.lines:  # collapse runs of blank lines
            if line or (out and out[-1]):
                out.append(line)
        return "\n".join(out).strip() + "\n"
