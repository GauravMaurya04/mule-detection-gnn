"""Import IBM Multi-GNN's own training log as a row of our transaction-task results.

IBM logs "Validation F1: x" and "Test F1: y" every epoch and reports the test F1 at the epoch
with the best validation F1 (first such epoch, since it updates only on a strict improvement).
Their F1 uses argmax over two classes, so it sits in our "F1 @ 0.5 (IBM protocol)" column.
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path

from mule.config import Config

VAL = re.compile(r"Validation F1:\s*([0-9.]+)")
TEST = re.compile(r"Test F1:\s*([0-9.]+)")


def parse_log(text: str) -> dict:
    val = [float(v) for v in VAL.findall(text)]
    test = [float(v) for v in TEST.findall(text)]
    n = min(len(val), len(test))
    if n == 0:
        raise ValueError("no 'Validation F1' / 'Test F1' lines found in the log")
    best = max(range(n), key=lambda i: (val[i], -i))  # highest val F1, earliest epoch on ties
    return {"epochs": n, "best_epoch": best, "val_f1": val[best], "test_f1": test[best]}


def import_multignn(
    cfg: Config, log: Path, name: str = "multignn", seconds: float | None = None
) -> dict:
    parsed = parse_log(log.read_text(errors="replace"))
    seconds_file = log.parent / "seconds.txt"
    if seconds is None and seconds_file.exists():
        seconds = float(seconds_file.read_text().strip())
    per_epoch = seconds / parsed["epochs"] if seconds else None

    def cell(v):
        return {"mean": v, "std": 0.0}

    result = {
        "model": name,
        "task": "edge",
        "final": True,
        "imported_from": str(log),
        "evaluated_at": datetime.now().isoformat(timespec="seconds"),
        "seeds": 1,
        "epochs": parsed["epochs"],
        "best_epoch": parsed["best_epoch"],
        "seconds_per_epoch": per_epoch,
        "windows": {
            "val": {"minority_f1@0.5": cell(parsed["val_f1"])},
            "test": {"minority_f1@0.5": cell(parsed["test_f1"])},
        },
    }
    out = cfg.paths.runs_dir / "edge" / name
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics_final.json").write_text(json.dumps(result, indent=2))
    return {**parsed, "seconds_per_epoch": per_epoch}


# Egressy et al., "Provably Powerful Graph Neural Networks for Directed Multigraphs" (AAAI 2024),
# AML results table, AML Small HI column: test minority-class F1 in %, mean +- std over 5 seeds.
PAPER_ROWS = {
    "paper_multigin_eu": (64.79, 1.22),
    "paper_multipna_eu": (68.16, 2.65),
}


def add_paper_rows(cfg: Config) -> dict:
    """Published Multi-GNN numbers as reference rows (test window, IBM's argmax protocol)."""
    written = {}
    for name, (mean, std) in PAPER_ROWS.items():
        result = {
            "model": name,
            "task": "edge",
            "final": True,
            "seeds": 5,
            "source": "Egressy et al. 2024 (AAAI), AML Small HI, test minority F1",
            "evaluated_at": datetime.now().isoformat(timespec="seconds"),
            "windows": {"test": {"minority_f1@0.5": {"mean": mean / 100, "std": std / 100}}},
        }
        out = cfg.paths.runs_dir / "edge" / name
        out.mkdir(parents=True, exist_ok=True)
        (out / "metrics_final.json").write_text(json.dumps(result, indent=2))
        written[name] = mean / 100
    return written
