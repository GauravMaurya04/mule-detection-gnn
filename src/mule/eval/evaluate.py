"""Stage 5: metrics from saved scores. Test windows are scored only with --final.

Every model writes runs/<task>/<model>/seed<k>/scores_<window>.parquet during training; this
stage turns them into metrics, averages over seeds, logs to MLflow and rebuilds
reports/results_<task>.md.

A normal run writes metrics.json (validation only). A --final run also writes
metrics_final.json (validation + test windows), which later normal runs never touch, so the
recorded test results survive until the model is retrained (retraining deletes the folder).
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from mule.config import Config
from mule.eval import metrics as M
from mule.labels.stage import k_values
from mule.report import Report
from mule.tracking import log_run

TEST_WINDOWS = ("test", "test_no_tail")


def check_window_allowed(window: str, final: bool) -> None:
    if window in TEST_WINDOWS and not final:
        raise PermissionError(f"window {window!r} is test data: pass --final to evaluate it")


def node_metrics(df: pd.DataFrame, threshold: float, cfg: Config, ci: int = 0) -> dict:
    y, s = df["is_mule"].to_numpy(bool), df["score"].to_numpy(float)
    ks = k_values(len(df), int(y.sum()), cfg.evaluation.k_fixed, cfg.evaluation.k_share)
    out = {"rows": len(df), "positives": int(y.sum()), "pr_auc": M.pr_auc(y, s)}
    for name, k in ks.items():
        out[f"precision@{name}"], out[f"recall@{name}"] = M.precision_recall_at_k(y, s, k)
    out.update({f"{k}@val_threshold": v for k, v in M.threshold_metrics(y, s, threshold).items()})
    k_share = list(ks.values())[1]
    out["adjacent_in_top_0.1%"] = M.expected_in_top_k(df["adjacent"].to_numpy(), s, k_share)
    k_r = ks["R-precision"]
    for group, mask in (
        ("repeat", df["prior_mule"].to_numpy(bool)),
        ("first_time", ~df["prior_mule"].to_numpy(bool)),
    ):
        flag = y & mask
        out[f"{group}_mules"] = int(flag.sum())
        out[f"recall_{group}@R"] = (
            M.expected_in_top_k(flag, s, k_r) / flag.sum() if flag.sum() else np.nan
        )
    if ci:
        out["pr_auc_ci_low"], out["pr_auc_ci_high"] = M.bootstrap_pr_auc_ci(y, s, ci)
    return out


def edge_metrics(df: pd.DataFrame, threshold: float) -> dict:
    y, s = df["is_laundering"].to_numpy(bool), df["score"].to_numpy(float)
    t = M.threshold_metrics(y, s, threshold)
    return {
        "rows": len(df),
        "positives": int(y.sum()),
        "pr_auc": M.pr_auc(y, s),
        "minority_f1": t["f1"],
        "precision@val_threshold": t["precision"],
        "recall@val_threshold": t["recall"],
    }


def _aggregate(per_seed: list[dict]) -> dict:
    """Mean and std per metric over the seeds that report it (CIs exist for seed 0 only)."""
    keys = dict.fromkeys(k for m in per_seed for k in m)
    out = {}
    for k in keys:
        values = np.array([m[k] for m in per_seed if k in m], dtype=float)
        all_nan = np.isnan(values).all()
        out[k] = {
            "mean": float("nan") if all_nan else float(np.nanmean(values)),
            "std": 0.0 if all_nan or len(values) < 2 else float(np.nanstd(values)),
        }
    return out


def evaluate_model(cfg: Config, task: str, model_dir: Path, final: bool) -> dict:
    windows = ["val", *(TEST_WINDOWS if final else ())]
    seeds = sorted(model_dir.glob("seed*"))
    if not seeds:
        raise FileNotFoundError(f"no trained seeds in {model_dir}: run `mule train` first")
    result = {
        "model": model_dir.name,
        "task": task,
        "final": final,
        "evaluated_at": datetime.now().isoformat(timespec="seconds"),
        "seeds": len(seeds),
        "windows": {},
    }
    for window in windows:
        check_window_allowed(window, final)
        per_seed = []
        for i, seed_dir in enumerate(seeds):
            info = json.loads((seed_dir / "run.json").read_text())
            df = pd.read_parquet(seed_dir / f"scores_{window}.parquet")
            if task == "node":
                ci = cfg.evaluation.bootstrap if (i == 0 and window in TEST_WINDOWS) else 0
                m = node_metrics(df, info["threshold"], cfg, ci)
            else:
                m = edge_metrics(df, info["threshold"])
            m["train_seconds"] = info["train_seconds"]
            m["score_seconds_per_1k"] = info["score_seconds_per_1k"]
            per_seed.append(m)
        result["windows"][window] = _aggregate(per_seed)
    val_only = {**result, "final": False, "windows": {"val": result["windows"]["val"]}}
    (model_dir / "metrics.json").write_text(json.dumps(val_only, indent=2))
    if final:
        (model_dir / "metrics_final.json").write_text(json.dumps(result, indent=2))
    flat = {f"{w}.{k}": v["mean"] for w, ms in result["windows"].items() for k, v in ms.items()}
    log_run(
        cfg,
        f"evaluate-{task}-{model_dir.name}",
        params={"task": task, "model": model_dir.name},
        metrics=flat,
        tags={"stage": "evaluate", "final": str(final)},
    )
    return result


NODE_COLUMNS = [
    ("pr_auc", "PR-AUC"),
    ("precision@K=100", "P@100"),
    ("precision@K=0.1% of active", "P@0.1%"),
    ("precision@R-precision", "R-prec"),
    ("f1@val_threshold", "F1"),
    ("fpr@val_threshold", "FPR"),
    ("adjacent_in_top_0.1%", "adjacent in top 0.1%"),
    ("recall_first_time@R", "first-time recall@R"),
    ("recall_repeat@R", "repeat recall@R"),
]
EDGE_COLUMNS = [
    ("pr_auc", "PR-AUC"),
    ("minority_f1", "minority F1"),
    ("precision@val_threshold", "precision"),
    ("recall@val_threshold", "recall"),
]


def _num(x: float) -> str:
    return f"{x:.1e}" if 0 < abs(x) < 0.001 else f"{x:.3f}"  # tiny FPRs stay visible


def _cell(v: dict | None) -> str:
    if v is None or np.isnan(v["mean"]):
        return "-"
    return _num(v["mean"]) + (f" ± {_num(v['std'])}" if v["std"] else "")


def write_results(cfg: Config, task: str) -> Path:
    task_dir = cfg.paths.runs_dir / task
    results = []
    for model_dir in sorted(p for p in task_dir.iterdir() if p.is_dir()):
        for name in ("metrics_final.json", "metrics.json"):  # prefer the recorded final run
            if (model_dir / name).exists():
                results.append(json.loads((model_dir / name).read_text()))
                break
    cols = NODE_COLUMNS if task == "node" else EDGE_COLUMNS
    rep = Report(f"Results: {'account' if task == 'node' else 'transaction'} task")
    rep.note(
        "Mean ± std over seeds. Thresholds are chosen on validation and frozen for test. "
        "Test windows appear only for models evaluated with --final."
    )
    for window in ("val", *TEST_WINDOWS):
        rows = []
        for r in results:
            ms = r["windows"].get(window)
            if not ms:
                continue
            row = {"model": r["model"], "seeds": r["seeds"]}
            for key, label in cols:
                v = ms.get(key)
                row[label] = (
                    "-"
                    if v is None or np.isnan(v["mean"])
                    else (f"{v['mean']:.3f}" + (f" ± {v['std']:.3f}" if v["std"] else ""))
                )
            if "pr_auc_ci_low" in ms:
                row["PR-AUC 95% CI (seed 0)"] = (
                    f"{ms['pr_auc_ci_low']['mean']:.3f}-{ms['pr_auc_ci_high']['mean']:.3f}"
                )
            row["positives"] = int(ms["positives"]["mean"])
            rows.append(row)
        if rows:
            rep.section(f"Window: {window}")
            rep.table(pd.DataFrame(rows))
    path = cfg.paths.reports_dir / f"results_{task}.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(rep.text())
    return path


def run_evaluate(cfg: Config, task: str, models: list[str] | None, final: bool) -> Path:
    task_dir = cfg.paths.runs_dir / task
    names = models or sorted(p.name for p in task_dir.iterdir() if p.is_dir())
    for name in names:
        evaluate_model(cfg, task, task_dir / name, final)
    return write_results(cfg, task)
