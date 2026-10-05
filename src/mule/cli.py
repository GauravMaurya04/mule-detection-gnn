"""Single entry point: `mule <command>`. Each command is one pipeline stage."""

from __future__ import annotations

import time
from pathlib import Path

import typer

from mule.config import load_config

app = typer.Typer(add_completion=False, no_args_is_help=True, help="Mule detection pipeline.")
CONFIG = typer.Option(Path("configs/base.yaml"), "--config", "-c", help="Path to the YAML config.")


def _timed(label: str, fn, *args):
    start = time.perf_counter()
    result = fn(*args)
    typer.echo(f"{label} done in {time.perf_counter() - start:.0f}s")
    return result


@app.command()
def ingest(config: Path = CONFIG) -> None:
    """Raw CSVs -> data/interim Parquet (typed, account ids, USD amounts)."""
    from mule.data.ingest import run_ingest

    typer.echo(_timed("ingest", run_ingest, load_config(config)))


@app.command()
def phase0(config: Path = CONFIG) -> None:
    """Dataset facts, split, mule counts and delays -> reports/phase0.md."""
    from mule.phase0 import run_phase0

    cfg = load_config(config)
    _timed("phase0", run_phase0, cfg)
    typer.echo((cfg.paths.reports_dir / "phase0.md").read_text())


@app.command()
def labels(config: Path = CONFIG) -> None:
    """Split manifest + account labels per window -> data/processed, reports/labels.md."""
    from mule.labels.stage import run_labels

    cfg = load_config(config)
    _timed("labels", run_labels, cfg)
    typer.echo((cfg.paths.reports_dir / "labels.md").read_text())


@app.command()
def features(config: Path = CONFIG) -> None:
    """Tier 1 account features per window -> data/processed/account_features.parquet."""
    from mule.features.stage import run_features

    typer.echo(_timed("features", run_features, load_config(config)))


TASK = typer.Option("node", "--task", "-t", help="node (accounts) or edge (transactions).")


@app.command()
def train(
    config: Path = CONFIG,
    task: str = TASK,
    model: str = typer.Option("xgb", "--model", "-m", help="rule_ach, rules or xgb."),
    seeds: int = typer.Option(3, "--seeds", help="Seeds to train (rules always use one)."),
) -> None:
    """Train one model on one task; saves models, val thresholds and scores under runs/."""
    from mule.models.train import run_train

    summary = _timed("train", run_train, load_config(config), task, model, seeds)
    for s in summary["seeds"]:
        typer.echo(f"  seed {s['seed']}: val PR-AUC {s['val_pr_auc']:.4f} ({s['train_seconds']}s)")


@app.command()
def baselines(config: Path = CONFIG, seeds: int = typer.Option(3, "--seeds")) -> None:
    """v1: rule_ach, rules and xgb on both tasks, then validation metrics."""
    from mule.eval.evaluate import run_evaluate
    from mule.models.train import run_baselines

    cfg = load_config(config)
    typer.echo(_timed("baselines", run_baselines, cfg, seeds).to_string(index=False))
    for task in ("node", "edge"):
        typer.echo((run_evaluate(cfg, task, None, final=False)).read_text())


@app.command()
def evaluate(
    config: Path = CONFIG,
    task: str = TASK,
    model: list[str] = typer.Option(None, "--model", "-m", help="Repeatable; default: all."),
    final: bool = typer.Option(False, "--final", help="Also score the test windows (logged)."),
) -> None:
    """Metrics from saved scores -> runs/<task>/<model>/metrics.json, reports/results_<task>.md."""
    from mule.eval.evaluate import run_evaluate

    path = run_evaluate(load_config(config), task, model or None, final)
    typer.echo(path.read_text())


@app.command()
def score(
    config: Path = CONFIG,
    account: str = typer.Option(None, "--account", "-a", help="BANK:ACCOUNT, e.g. 10:8000EBD30"),
    top: int = typer.Option(0, "--top", help="List the N highest-scoring accounts instead."),
    window: str = typer.Option("test", "--window", "-w", help="val, test or test_no_tail."),
    model: str = typer.Option("xgb", "--model", "-m"),
) -> None:
    """Score one account with its strongest evidence, or list the top accounts."""
    from mule.score import score_account, top_accounts

    cfg = load_config(config)
    if account:
        typer.echo(score_account(cfg, account, window, model))
    else:
        typer.echo(top_accounts(cfg, top or 10, window, model))


if __name__ == "__main__":
    app()
