"""Paired bootstrap: is model A really better than model B on the same rows?

Both models' scores are averaged over their seeds, then rows are resampled with replacement
and the PR-AUC difference is recomputed on each resample. Because both models see the same
resampled rows, this is far more sensitive than comparing two separate confidence intervals.
Validation only, unless --final (test windows are guarded exactly as in the evaluator).

Decision rule (set before any result was seen): if the 95% interval of A - B excludes 0, the
better model wins; otherwise the model with fewer features wins.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from mule.config import Config
from mule.eval.evaluate import check_window_allowed

ID = {"node": ("account_id", "is_mule"), "edge": ("tx_id", "is_laundering")}


def _seed_mean(cfg: Config, task: str, model: str, window: str) -> pd.DataFrame:
    seeds = sorted((cfg.paths.runs_dir / task / model).glob("seed*"))
    if not seeds:
        raise FileNotFoundError(f"no trained seeds for {task}/{model}")
    key, label = ID[task]
    frames = [
        pd.read_parquet(s / f"scores_{window}.parquet", columns=[key, label, "score"])
        for s in seeds
    ]
    out = frames[0][[key, label]].copy()
    out["score"] = np.mean(
        [f.set_index(key)["score"].reindex(out[key]).to_numpy() for f in frames], axis=0
    )
    return out


def _n_features(cfg: Config, task: str, model: str) -> int:
    info = json.loads((cfg.paths.runs_dir / task / model / "seed0" / "run.json").read_text())
    if "feature_columns" in info:
        return len(info["feature_columns"])
    stats = cfg.paths.runs_dir / task / model / "seed0" / "graph_stats.json"
    return len(json.loads(stats.read_text())["feature_columns"]) if stats.exists() else 0


def paired_bootstrap(
    y: np.ndarray, a: np.ndarray, b: np.ndarray, n: int, seed: int = 0
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n):
        idx = rng.integers(0, len(y), len(y))
        if y[idx].any():
            diffs.append(
                average_precision_score(y[idx], a[idx]) - average_precision_score(y[idx], b[idx])
            )
    return np.asarray(diffs)


def compare(
    cfg: Config,
    task: str,
    model_a: str,
    model_b: str,
    window: str = "val",
    n: int = 1000,
    final: bool = False,
) -> dict:
    check_window_allowed(window, final)
    key, label = ID[task]
    a = _seed_mean(cfg, task, model_a, window)
    b = _seed_mean(cfg, task, model_b, window).set_index(key).reindex(a[key])
    y = a[label].to_numpy(bool)
    sa, sb = a["score"].to_numpy(float), b["score"].to_numpy(float)
    diffs = paired_bootstrap(y, sa, sb, n)
    low, high = np.percentile(diffs, [2.5, 97.5])
    fa, fb = _n_features(cfg, task, model_a), _n_features(cfg, task, model_b)
    if low > 0:
        winner, reason = model_a, "interval above 0"
    elif high < 0:
        winner, reason = model_b, "interval below 0"
    else:
        winner = model_a if fa < fb else model_b
        reason = f"interval includes 0, so fewer features wins ({min(fa, fb)} vs {max(fa, fb)})"
    return {
        "task": task,
        "window": window,
        "a": model_a,
        "b": model_b,
        "pr_auc_a": float(average_precision_score(y, sa)),
        "pr_auc_b": float(average_precision_score(y, sb)),
        "difference": float(average_precision_score(y, sa) - average_precision_score(y, sb)),
        "ci_95": [float(low), float(high)],
        "share_a_better": float((diffs > 0).mean()),
        "resamples": len(diffs),
        "winner": winner,
        "reason": reason,
    }
