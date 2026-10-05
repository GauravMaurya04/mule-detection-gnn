"""Stage 4: train one model on one task for N seeds; save model, val threshold and scores."""

from __future__ import annotations

import json
import shutil
import time

import pandas as pd

from mule.config import Config
from mule.eval.metrics import best_f1_threshold, pr_auc
from mule.labels.split import EVAL_WINDOWS
from mule.models.data import load_edge, load_node
from mule.models.registry import make_model
from mule.tracking import log_run


def run_train(cfg: Config, task: str, model_name: str, n_seeds: int = 3, data=None) -> dict:
    data = data or (load_node(cfg) if task == "node" else load_edge(cfg))
    train, val = data["train"], data["val"]
    out_dir = cfg.paths.runs_dir / task / model_name
    if out_dir.exists():
        shutil.rmtree(out_dir)  # a retrain replaces the old run completely
    probe = make_model(model_name, task, cfg)
    seeds = 1 if probe.deterministic else n_seeds
    summary = {
        "task": task,
        "model": model_name,
        "train_rows": len(train.y),
        "train_positives": int(train.y.sum()),
        "seeds": [],
    }

    for seed in range(seeds):
        model = make_model(model_name, task, cfg)
        t0 = time.perf_counter()
        info = model.fit(train, val, seed)
        train_seconds = time.perf_counter() - t0

        seed_dir = out_dir / f"seed{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        scored = {}
        for window in EVAL_WINDOWS:
            part = data[window]
            t1 = time.perf_counter()
            scores = model.predict_scores(part.X)
            elapsed = time.perf_counter() - t1
            if window == "val":
                per_1k = 1000 * elapsed / max(len(scores), 1)
            frame = part.meta.drop(columns=["window"]).assign(score=scores)
            frame.to_parquet(seed_dir / f"scores_{window}.parquet", index=False)
            scored[window] = scores
        threshold = best_f1_threshold(val.y, scored["val"])
        model.save(seed_dir)
        run_info = {
            "seed": seed,
            "threshold": threshold,
            "train_seconds": round(train_seconds, 2),
            "score_seconds_per_1k": round(per_1k, 5),
            "feature_columns": list(train.X.columns),
            "config_hash": cfg.section_hash("dataset", "split", "labels", "features", "models"),
            **info,
        }
        (seed_dir / "run.json").write_text(json.dumps(run_info, indent=2))
        val_auc = pr_auc(val.y, scored["val"])
        summary["seeds"].append(
            {"seed": seed, "val_pr_auc": val_auc, "train_seconds": round(train_seconds, 1)}
        )
        log_run(
            cfg,
            f"train-{task}-{model_name}-seed{seed}",
            params={"task": task, "model": model_name, "seed": seed, **model.params()},
            metrics={"val.pr_auc": val_auc, "train_seconds": train_seconds},
            tags={"stage": "train"},
        )
    (out_dir / "train_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def run_baselines(cfg: Config, n_seeds: int = 3) -> pd.DataFrame:
    """v1: every baseline on both tasks, data built once per task."""
    rows = []
    for task in ("node", "edge"):
        data = load_node(cfg) if task == "node" else load_edge(cfg)
        for model_name in ("rule_ach", "rules", "xgb"):
            s = run_train(cfg, task, model_name, n_seeds, data=data)
            for seed in s["seeds"]:
                rows.append({"task": task, "model": model_name, **seed})
    return pd.DataFrame(rows)
