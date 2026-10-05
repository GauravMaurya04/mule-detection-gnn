"""MLflow logging that never breaks a run: metrics.json files are the record, MLflow is a view."""

from __future__ import annotations

import re
import subprocess
import warnings

from mule.config import Config


def git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True
        )
        return out.stdout.strip() or "unknown"
    except OSError:
        return "unknown"


def _clean(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_\-. /]", "_", name)


def log_run(
    cfg: Config, run_name: str, params: dict, metrics: dict, tags: dict | None = None
) -> None:
    if not cfg.tracking.mlflow_uri:
        return
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            import mlflow

            mlflow.set_tracking_uri(cfg.tracking.mlflow_uri)
            mlflow.set_experiment("mule-detection")
            with mlflow.start_run(run_name=run_name):
                mlflow.set_tags({"git_commit": git_commit(), **(tags or {})})
                mlflow.log_params({_clean(k): v for k, v in params.items()})
                clean = {_clean(k): float(v) for k, v in metrics.items() if v == v}  # drop NaN
                mlflow.log_metrics(clean)
    except Exception as err:  # tracking is a convenience; never fail the pipeline over it
        print(f"WARNING: MLflow logging skipped ({type(err).__name__}: {err})")
