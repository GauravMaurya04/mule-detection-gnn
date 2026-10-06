"""XGBoost random search on validation (Day 7), for the champion's feature groups.

Each trial trains seed 0 with early stopping on validation and is scored by validation PR-AUC;
the best settings are saved to runs/<task>/_tuning/xgb_tuned.json and retrained with N seeds
as the model `xgb_tuned`. Twenty trials on a few hundred validation positives make the best
trial's score somewhat optimistic; test is scored once, at the final evaluation.
"""

from __future__ import annotations

import json
import time

import numpy as np

from mule.config import Config
from mule.eval.metrics import pr_auc
from mule.models.data import load_edge, load_node
from mule.models.registry import XGB_GROUPS
from mule.models.train import _view, run_train
from mule.models.xgb import XGBModel


def sample_params(rng: np.random.Generator) -> dict:
    return {
        "max_depth": int(rng.integers(3, 11)),
        "learning_rate": float(np.exp(rng.uniform(np.log(0.01), np.log(0.2)))),
        "min_child_weight": float(np.exp(rng.uniform(np.log(0.5), np.log(20)))),
        "subsample": float(rng.uniform(0.5, 1.0)),
        "colsample_bytree": float(rng.uniform(0.4, 1.0)),
        "reg_lambda": float(np.exp(rng.uniform(np.log(0.1), np.log(20)))),
        "gamma": float(rng.uniform(0.0, 2.0)),
    }


def run_xgb_tune(
    cfg: Config,
    task: str,
    base: str,
    trials: int = 20,
    n_seeds: int = 3,
    force: bool = False,
    data=None,
) -> dict:
    if base not in XGB_GROUPS:
        raise ValueError(f"base must be one of {sorted(XGB_GROUPS)}")
    groups = XGB_GROUPS[base]
    raw = data or (load_node(cfg) if task == "node" else load_edge(cfg))
    columns = XGBModel(cfg.models.xgb, "xgb_tuned", groups).select_columns(raw["train"].X.columns)
    view = {k: _view(v, columns) for k, v in raw.items()}
    rng = np.random.default_rng(cfg.seed)
    results = [{"params": {}, "label": "defaults"}] + [
        {"params": sample_params(rng), "label": f"trial {i + 1}"} for i in range(trials)
    ]
    for r in results:  # the defaults run too, so tuning can never pick something worse on val
        t0 = time.perf_counter()
        model = XGBModel(cfg.models.xgb.model_copy(update=r["params"]), "xgb_tuned", groups)
        info = model.fit(view["train"], view["val"], seed=0)
        r["val_pr_auc"] = pr_auc(view["val"].y, model.predict_scores(view["val"].X))
        r["best_iteration"] = info["best_iteration"]
        r["seconds"] = round(time.perf_counter() - t0, 1)
        print(
            f"  {r['label']:>9}: val PR-AUC {r['val_pr_auc']:.4f} ({r['seconds']}s) {r['params']}",
            flush=True,
        )
    best = max(results, key=lambda r: r["val_pr_auc"])
    out = cfg.paths.runs_dir / task / "_tuning"
    out.mkdir(parents=True, exist_ok=True)
    spec = {"base": base, "groups": list(groups), "params": best["params"], "trials": results}
    (out / "xgb_tuned.json").write_text(json.dumps(spec, indent=2))
    # Retrain on the column subset already in memory. Passing the full matrix as well would
    # hold two copies at once (~6 GB on HI-Small transactions) and can exhaust WSL's memory.
    del raw
    summary = run_train(cfg, task, "xgb_tuned", n_seeds, data=view, force=force)
    return {"best": best, "trials": results, "final": summary}


def run_stack(
    cfg: Config,
    base: str,
    n_seeds: int = 3,
    kind: str = "sage",
    refresh_oof: bool = False,
    force: bool = False,
) -> dict:
    """Out-of-fold GNN scores (unless already built), then xgb_stacked on the account task.
    Uses xgb_tuned's settings when they were tuned for the same base, so xgb_stacked vs
    xgb_tuned isolates what the GNN score adds."""
    oof_meta = None
    if refresh_oof or not (cfg.paths.processed_dir / "gnn_oof_node.parquet").exists():
        from mule.models.gnn.stack import run_gnn_oof  # needs PyTorch; only when (re)building

        oof_meta = run_gnn_oof(cfg, kind)
    tuned = cfg.paths.runs_dir / "node" / "_tuning" / "xgb_tuned.json"
    params, source = {}, "config defaults"
    if tuned.exists() and json.loads(tuned.read_text())["base"] == base:
        params, source = json.loads(tuned.read_text())["params"], "xgb_tuned"
    spec = {
        "base": base,
        "groups": [*XGB_GROUPS[base], "st"],
        "params": params,
        "params_from": source,
    }
    out = cfg.paths.runs_dir / "node" / "_tuning"
    out.mkdir(parents=True, exist_ok=True)
    (out / "xgb_stacked.json").write_text(json.dumps(spec, indent=2))
    summary = run_train(cfg, "node", "xgb_stacked", n_seeds, force=force)
    return {"oof": oof_meta, "params_from": source, "final": summary}
