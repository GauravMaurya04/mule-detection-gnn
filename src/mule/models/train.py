"""Stage 4: train one model on one task for N seeds; save model, val threshold and scores."""

from __future__ import annotations

import json
import shutil
import time

import pandas as pd

from mule.config import Config
from mule.eval.metrics import best_f1_threshold, pr_auc
from mule.labels.split import EVAL_WINDOWS
from mule.models.data import TaskData, load_edge, load_node
from mule.models.registry import make_model
from mule.tracking import log_run


def _view(part: TaskData, columns: list[str]) -> TaskData:
    if list(part.X.columns) == columns:  # no copy when the model uses every column
        return part
    return TaskData(part.X[columns], part.y, part.meta)


def run_train(
    cfg: Config, task: str, model_name: str, n_seeds: int = 3, data=None, force: bool = False
) -> dict:
    out_dir = cfg.paths.runs_dir / task / model_name
    if (out_dir / "metrics_final.json").exists() and not force:
        raise FileExistsError(
            f"{out_dir} has recorded --final results, so the model is frozen. "
            "Retraining would delete them; pass --force if you really mean to."
        )
    probe = make_model(model_name, task, cfg)
    data = data or (load_node(cfg) if task == "node" else load_edge(cfg))
    columns = probe.select_columns(data["train"].X.columns)
    data = {k: _view(v, columns) for k, v in data.items()}
    train, val = data["train"], data["val"]
    if out_dir.exists():
        shutil.rmtree(out_dir)  # a retrain replaces the old run completely
    seeds = 1 if probe.deterministic else n_seeds
    summary = {
        "task": task,
        "model": model_name,
        "train_rows": len(train.y),
        "train_positives": int(train.y.sum()),
        "features": len(columns),
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
            "feature_columns": columns,
            "probabilistic": model.probabilistic,
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


def run_baselines(cfg: Config, n_seeds: int = 3, models=None) -> pd.DataFrame:
    """Train several models on both tasks, building each task's data once. Frozen models
    (those with recorded --final results) are skipped, never retrained."""
    from mule.models.registry import MODEL_NAMES

    rows = []
    for task in ("node", "edge"):
        todo = [
            m
            for m in (models or MODEL_NAMES)
            if not (cfg.paths.runs_dir / task / m / "metrics_final.json").exists()
        ]
        if not todo:
            continue
        data = load_node(cfg) if task == "node" else load_edge(cfg)
        for model_name in todo:
            s = run_train(cfg, task, model_name, n_seeds, data=data)
            for seed in s["seeds"]:
                rows.append({"task": task, "model": model_name, "features": s["features"], **seed})
    return pd.DataFrame(rows)
