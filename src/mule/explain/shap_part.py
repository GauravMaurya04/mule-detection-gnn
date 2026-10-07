"""Exact TreeSHAP for an XGBoost run, averaged over its seeds, summed by feature group."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from mule.config import Config
from mule.models.registry import make_model


def seed_shap(cfg: Config, task: str, model: str, X: pd.DataFrame) -> pd.DataFrame:
    seeds = sorted((cfg.paths.runs_dir / task / model).glob("seed*"))
    if not seeds:
        raise FileNotFoundError(f"no trained seeds for {task}/{model}")
    parts = []
    for seed_dir in seeds:
        info = json.loads((seed_dir / "run.json").read_text())
        m = make_model(model, task, cfg)
        m.load(seed_dir, info)
        parts.append(m.explain(X[info["feature_columns"]]).to_numpy())
    cols = json.loads((seeds[0] / "run.json").read_text())["feature_columns"]
    return pd.DataFrame(np.mean(parts, axis=0), index=X.index, columns=cols)


def group_sums(contrib: pd.DataFrame) -> pd.DataFrame:
    groups = [c.split("_")[0] for c in contrib.columns]
    return contrib.T.groupby(groups).sum().T
