"""Temporal split, reproducing IBM Multi-GNN's data_loading.py.

IBM cuts the timeline into whole days counted from the first timestamp, then picks the two
cut days (i, j) whose train/val/test transaction shares have the smallest worst-case relative
error against the target shares. On HI-Small that gives train Sep 1-6, val Sep 7-8, test Sep 9-18.

On top of IBM's three windows:
- the train period is cut into snapshots of `train_snapshot_days` (2 days -> train_s1..s3), so
  training rows have the same shape as the 2-day validation and test windows;
- test_no_tail stops where normal activity ends (the post-Sep 10 tail is a generator artifact).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

DAY = pd.Timedelta(days=1)
EVAL_WINDOWS = ("val", "test", "test_no_tail")


def daily_counts(ts: pd.Series) -> tuple[pd.Timestamp, np.ndarray]:
    origin = ts.min()
    day_index = ((ts - origin) // DAY).to_numpy()
    return origin, np.bincount(day_index)


def choose_cut_days(counts: np.ndarray, shares=(0.6, 0.2, 0.2)) -> tuple[int, int]:
    """IBM's search: minimise the largest relative error of the three split shares."""
    best, best_ij = None, None
    for i in range(1, len(counts)):
        for j in range(i + 1, len(counts)):
            totals = np.array([counts[:i].sum(), counts[i:j].sum(), counts[j:].sum()], float)
            props = totals / totals.sum()
            score = max(abs(p - t) / t for p, t in zip(props, shares, strict=True))
            if best is None or score < best:  # strict <, so ties keep the first, as min() does
                best, best_ij = score, (i, j)
    if best_ij is None:
        raise ValueError("need at least three days of data to split")
    return best_ij


def build_manifest(
    ts: pd.Series, shares=(0.6, 0.2, 0.2), no_tail_end=None, train_snapshot_days: int = 2
) -> dict:
    origin, counts = daily_counts(ts)
    i, j = choose_cut_days(counts, shares)
    val_start, test_start = origin + i * DAY, origin + j * DAY
    end = ts.max() + pd.Timedelta(minutes=1)  # windows are [start, end)
    no_tail = min(pd.Timestamp(no_tail_end), end) if no_tail_end else end

    windows: dict[str, tuple[pd.Timestamp, pd.Timestamp]] = {}
    snapshots = []
    for k, day in enumerate(range(0, i, train_snapshot_days), start=1):
        start = origin + day * DAY
        stop = min(origin + (day + train_snapshot_days) * DAY, val_start)
        windows[f"train_s{k}"] = (start, stop)
        snapshots.append(f"train_s{k}")
    windows["val"] = (val_start, test_start)
    windows["test"] = (test_start, end)
    windows["test_no_tail"] = (test_start, no_tail)
    period = counts[:i].sum(), counts[i:j].sum(), counts[j:].sum()
    return {
        "method": "ibm_days",
        "target_shares": list(shares),
        "actual_shares": [round(float(s) / float(counts.sum()), 4) for s in period],
        "cut_days": [int(i), int(j)],
        "train_period": (origin, val_start),
        "train_snapshots": snapshots,
        "windows": windows,
    }


def label_windows(manifest: dict) -> list[str]:
    """Every window that gets labels and features, in time order."""
    return [*manifest["train_snapshots"], *EVAL_WINDOWS]


def window_bounds(manifest: dict, name: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    start, end = manifest["windows"][name]
    return pd.Timestamp(start), pd.Timestamp(end)


def assign_window(ts: pd.Series, manifest: dict) -> pd.Series:
    """Snapshot / val / test name for each transaction (test_no_tail is a subset of test)."""
    names = [*manifest["train_snapshots"], "val", "test"]
    out = pd.Series(pd.NA, index=ts.index, dtype="string")
    for name in names:
        start, end = window_bounds(manifest, name)
        out[(ts >= start) & (ts < end)] = name
    return out.astype(pd.CategoricalDtype(names))


def save_manifest(manifest: dict, path: Path) -> None:
    out = dict(manifest)
    out["windows"] = {k: [str(a), str(b)] for k, (a, b) in manifest["windows"].items()}
    out["train_period"] = [str(t) for t in manifest["train_period"]]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2))


def load_manifest(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"{path} not found: run `mule labels` first")
    raw = json.loads(path.read_text())
    raw["windows"] = {k: (pd.Timestamp(a), pd.Timestamp(b)) for k, (a, b) in raw["windows"].items()}
    raw["train_period"] = tuple(pd.Timestamp(t) for t in raw["train_period"])
    return raw
