"""Reason-code catalog and the firing rule.

Attribution ranks the evidence; the catalog phrases it. A code fires when its condition holds
(against training-population percentiles) AND its feature group is among the account's top
positive SHAP groups, so a code is never "true but irrelevant" nor "important but unstateable".
RC08 (risky neighbourhood) comes from the GNN member and is placed by its share of the risk.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

GNN_CODE = "RC08"


@dataclass
class Code:
    id: str
    name: str
    group: str
    feature: str
    direction: str  # "high" or "near_zero"
    text: str
    transform: str | None = None


def load_catalog(path: Path) -> list[Code]:
    raw = yaml.safe_load(Path(path).read_text())["codes"]
    codes = [Code(id=k, **v) for k, v in raw.items()]
    bad = [c.id for c in codes if c.direction not in ("high", "near_zero")]
    if bad:
        raise ValueError(f"unknown direction in {bad}")
    return codes


def _apply(values, how: str | None):
    values = np.asarray(values, dtype=float)
    return np.expm1(values) if how == "expm1" else values


def thresholds(train: pd.DataFrame, catalog: list[Code], q_high: float, q_low: float) -> dict:
    """Cut-offs and typical values from the training population (never from test)."""
    out = {}
    for c in catalog:
        v = _apply(train[c.feature].dropna(), c.transform)
        if c.direction == "high":
            cut = float(np.quantile(v, q_high)) if len(v) else np.inf
        else:
            cut = float(np.quantile(np.abs(v), q_low)) if len(v) else -np.inf
        out[c.id] = {"cut": cut, "typical": float(np.median(v)) if len(v) else np.nan}
    return out


def fire(
    row: pd.Series,
    group_contrib: pd.Series,
    catalog: list[Code],
    cuts: dict,
    top_groups: int = 3,
    max_codes: int = 3,
) -> list[dict]:
    """Codes for one account. `group_contrib`: SHAP contribution per feature group."""
    positive = group_contrib[group_contrib > 0].sort_values(ascending=False).head(top_groups)
    fired = []
    for c in catalog:
        if c.group not in positive.index or c.feature not in row.index:
            continue
        value = float(_apply([row[c.feature]], c.transform)[0])
        if np.isnan(value):
            continue
        cut = cuts[c.id]["cut"]
        hit = value >= cut and value > 0 if c.direction == "high" else abs(value) <= cut
        if hit:
            fired.append(
                {
                    "code": c.id,
                    "name": c.name,
                    "group": c.group,
                    "text": c.text.format(value=value, typical=cuts[c.id]["typical"]),
                    "strength": float(positive[c.group]),
                }
            )
    fired.sort(key=lambda d: -d["strength"])
    return fired[:max_codes]


def gnn_code(gnn_share: float, minimum: float) -> dict | None:
    """RC08 when the GNN supplies more than `minimum` of the account's added risk."""
    if gnn_share <= minimum:
        return None
    return {
        "code": GNN_CODE,
        "name": "Risky neighbourhood",
        "group": "gnn",
        "text": f"The network model supplies {gnn_share:.0%} of this account's added risk: "
        "its trading partners look like mules",
        "strength": gnn_share,
    }


def combine(fired: list[dict], gnn: dict | None, max_codes: int = 3) -> list[dict]:
    """GNN-led accounts (share >= 50%) lead with RC08; otherwise it closes the list."""
    if gnn is None:
        return fired[:max_codes]
    if gnn["strength"] >= 0.5:
        return [gnn, *fired[: max_codes - 1]]
    return [*fired[: max_codes - 1], gnn]


def risk_shares(
    p_a: np.ndarray, p_b: np.ndarray, weight_a: float, base_a: float, base_b: float
) -> np.ndarray:
    """Share of each account's added risk (above the population mean) supplied by member B."""
    add_a = np.clip(weight_a * (p_a - base_a), 0, None)
    add_b = np.clip((1 - weight_a) * (p_b - base_b), 0, None)
    total = add_a + add_b
    return np.where(total > 0, add_b / np.where(total > 0, total, 1), 0.0)
