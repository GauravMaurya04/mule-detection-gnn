"""Calibration and ensembles, built from saved scores only (no retraining).

Both average a model over its seeds first, so `<model>_cal` and `ens_<a>__<b>` are single
"seed0" runs that the evaluator, `mule compare` and the results tables treat like any model.

Calibration: Platt scaling (logistic regression on the score's logit) or isotonic regression,
whichever has the lower 5-fold cross-validated Brier score on validation; then refit on all
of validation. Platt keeps the ranking (PR-AUC, P@K unchanged); isotonic can create ties.

Ensemble: both models are Platt-calibrated on validation so their scores share a scale, then
averaged with the weight w (on A) that maximises validation PR-AUC over a 0.05 grid.
"""

from __future__ import annotations

import json
import shutil

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss
from sklearn.model_selection import StratifiedKFold

from mule.config import Config
from mule.eval.metrics import best_f1_threshold
from mule.labels.split import EVAL_WINDOWS

ID = {"node": ("account_id", "is_mule"), "edge": ("tx_id", "is_laundering")}
EPS = 1e-6


def seed_mean(cfg: Config, task: str, model: str, window: str) -> pd.DataFrame:
    """The model's scores for a window, averaged over seeds; keeps every metadata column."""
    seeds = sorted((cfg.paths.runs_dir / task / model).glob("seed*"))
    if not seeds:
        raise FileNotFoundError(f"no trained seeds for {task}/{model}")
    key = ID[task][0]
    first = pd.read_parquet(seeds[0] / f"scores_{window}.parquet")
    scores = [first["score"].to_numpy(float)]
    for s in seeds[1:]:
        other = pd.read_parquet(s / f"scores_{window}.parquet", columns=[key, "score"])
        scores.append(other.set_index(key)["score"].reindex(first[key]).to_numpy(float))
    return first.assign(score=np.mean(scores, axis=0))


def _logit(s: np.ndarray) -> np.ndarray:
    s = np.clip(np.asarray(s, float), EPS, 1 - EPS)
    return np.log(s / (1 - s))


class Platt:
    name = "platt"

    def fit(self, s, y):
        self.m = LogisticRegression(C=1e6, max_iter=1000).fit(_logit(s)[:, None], y)
        return self

    def predict(self, s):
        return self.m.predict_proba(_logit(s)[:, None])[:, 1]


class Isotonic:
    name = "isotonic"

    def fit(self, s, y):
        self.m = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(s, y)
        return self

    def predict(self, s):
        return self.m.predict(np.asarray(s, float))


def choose_calibrator(s: np.ndarray, y: np.ndarray, folds: int = 5, seed: int = 0) -> dict:
    cv = StratifiedKFold(folds, shuffle=True, random_state=seed)
    scores = {}
    for cls in (Platt, Isotonic):
        briers = []
        for tr, te in cv.split(s, y):
            briers.append(brier_score_loss(y[te], cls().fit(s[tr], y[tr]).predict(s[te])))
        scores[cls.name] = float(np.mean(briers))
    best = min(scores, key=scores.get)
    return {"choice": best, "cv_brier": scores}


def _source_info(cfg: Config, task: str, model: str) -> dict:
    info = json.loads((cfg.paths.runs_dir / task / model / "seed0" / "run.json").read_text())
    n = len(info.get("feature_columns", [])) or info.get("n_features", 0)
    if not n:
        stats = cfg.paths.runs_dir / task / model / "seed0" / "graph_stats.json"
        n = len(json.loads(stats.read_text())["feature_columns"]) if stats.exists() else 0
    return {
        "train_seconds": info.get("train_seconds", 0.0),
        "score_seconds_per_1k": info.get("score_seconds_per_1k", 0.0),
        "n_features": n,
    }


def _write(cfg: Config, task: str, name: str, frames: dict, info: dict, force: bool) -> None:
    out = cfg.paths.runs_dir / task / name
    if (out / "metrics_final.json").exists() and not force:
        raise FileExistsError(f"{out} has recorded --final results; pass --force to rebuild it")
    if out.exists():
        shutil.rmtree(out)
    seed_dir = out / "seed0"
    seed_dir.mkdir(parents=True)
    for window, df in frames.items():
        df.to_parquet(seed_dir / f"scores_{window}.parquet", index=False)
    label = ID[task][1]
    val = frames["val"]
    info = {
        **info,
        "seed": 0,
        "probabilistic": True,
        "threshold": best_f1_threshold(val[label].to_numpy(), val["score"].to_numpy()),
    }
    (seed_dir / "run.json").write_text(json.dumps(info, indent=2))


def run_calibrate(cfg: Config, task: str, model: str, force: bool = False) -> dict:
    label = ID[task][1]
    frames = {w: seed_mean(cfg, task, model, w) for w in EVAL_WINDOWS}
    val = frames["val"]
    y, s = val[label].to_numpy(int), val["score"].to_numpy(float)
    choice = choose_calibrator(s, y)
    cal = (Platt if choice["choice"] == "platt" else Isotonic)().fit(s, y)
    for df in frames.values():
        df["score"] = cal.predict(df["score"].to_numpy(float))
    name = f"{model}_cal"
    _write(
        cfg,
        task,
        name,
        frames,
        {**_source_info(cfg, task, model), "calibrated_from": model, **choice},
        force,
    )
    return {"model": name, **choice}


def run_ensemble(
    cfg: Config, task: str, a: str, b: str, step: float = 0.05, force: bool = False
) -> dict:
    key, label = ID[task]
    fa = {w: seed_mean(cfg, task, a, w) for w in EVAL_WINDOWS}
    fb = {w: seed_mean(cfg, task, b, w).set_index(key)["score"] for w in EVAL_WINDOWS}
    y = fa["val"][label].to_numpy(int)
    sa_val = fa["val"]["score"].to_numpy(float)
    sb_val = fb["val"].reindex(fa["val"][key]).to_numpy(float)
    ca, cb = Platt().fit(sa_val, y), Platt().fit(sb_val, y)
    pa, pb = ca.predict(sa_val), cb.predict(sb_val)
    grid = np.round(np.arange(0, 1 + 1e-9, step), 4)
    aucs = [average_precision_score(y, w * pa + (1 - w) * pb) for w in grid]
    weight = float(grid[int(np.argmax(aucs))])
    frames = {}
    for w_name, df in fa.items():
        sa = ca.predict(df["score"].to_numpy(float))
        sb = cb.predict(fb[w_name].reindex(df[key]).to_numpy(float))
        frames[w_name] = df.assign(score=weight * sa + (1 - weight) * sb)
    ia, ib = _source_info(cfg, task, a), _source_info(cfg, task, b)
    info = {
        "members": [a, b],
        "weight_a": weight,
        "val_pr_auc_by_weight": dict(zip(map(float, grid), map(float, aucs), strict=True)),
        "n_features": ia["n_features"] + ib["n_features"],
        "train_seconds": ia["train_seconds"] + ib["train_seconds"],
        "score_seconds_per_1k": ia["score_seconds_per_1k"] + ib["score_seconds_per_1k"],
    }
    name = f"ens_{a}__{b}"
    _write(cfg, task, name, frames, info, force)
    return {
        "model": name,
        "weight_a": weight,
        "val_pr_auc": float(max(aucs)),
        "val_pr_auc_a_alone": float(aucs[-1]),
        "val_pr_auc_b_alone": float(aucs[0]),
    }


def ensemble_components(cfg: Config, task: str, name: str) -> dict:
    """Rebuild an ensemble's members, weight and Platt calibrators (refit on validation exactly
    as run_ensemble did), so its score can be split into its members' contributions."""
    info = json.loads((cfg.paths.runs_dir / task / name / "seed0" / "run.json").read_text())
    if "members" not in info:
        raise ValueError(f"{name} is not an ensemble")
    a, b = info["members"]
    key, label = ID[task]
    fa = seed_mean(cfg, task, a, "val")
    sb = seed_mean(cfg, task, b, "val").set_index(key)["score"].reindex(fa[key]).to_numpy(float)
    y = fa[label].to_numpy(int)
    return {
        "a": a,
        "b": b,
        "weight_a": float(info["weight_a"]),
        "cal_a": Platt().fit(fa["score"].to_numpy(float), y),
        "cal_b": Platt().fit(sb, y),
    }
