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
    force: bool = typer.Option(False, "--force", help="Retrain even a frozen (--final) model."),
) -> None:
    """Train one model on one task; saves models, val thresholds and scores under runs/."""
    from mule.models.train import run_train

    summary = _timed("train", run_train, load_config(config), task, model, seeds, None, force)
    for s in summary["seeds"]:
        typer.echo(f"  seed {s['seed']}: val PR-AUC {s['val_pr_auc']:.4f} ({s['train_seconds']}s)")


@app.command()
def baselines(
    config: Path = CONFIG,
    seeds: int = typer.Option(3, "--seeds"),
    model: list[str] = typer.Option(None, "--model", "-m", help="Repeatable; default: all."),
) -> None:
    """Train models on both tasks (skipping frozen ones), then validation metrics."""
    from mule.eval.evaluate import run_evaluate
    from mule.models.train import run_baselines

    cfg = load_config(config)
    typer.echo(_timed("baselines", run_baselines, cfg, seeds, model or None).to_string(index=False))
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


@app.command()
def compare(
    config: Path = CONFIG,
    task: str = TASK,
    a: str = typer.Option(..., "--a", help="Model A, e.g. xgb_pattern"),
    b: str = typer.Option(..., "--b", help="Model B, e.g. xgb_graph"),
    window: str = typer.Option("val", "--window", "-w"),
    n: int = typer.Option(1000, "--n", help="Bootstrap resamples (use ~200 for transactions)."),
    final: bool = typer.Option(False, "--final"),
) -> None:
    """Paired bootstrap of the PR-AUC difference A - B on the same rows (validation by default)."""
    from mule.eval.compare import compare as run_compare

    result = run_compare(load_config(config), task, a, b, window, n, final)
    for k, v in result.items():
        typer.echo(f"{k}: {v}")


GNN_KIND = typer.Option("gine_rmp", "--model", "-m", help="sage, gine or gine_rmp.")


@app.command("gnn-time")
def gnn_time(config: Path = CONFIG, model: str = GNN_KIND) -> None:
    """Time one epoch + one validation pass before committing to a grid."""
    from mule.models.gnn.train import run_gnn_time

    for k, v in run_gnn_time(load_config(config), model).items():
        typer.echo(f"{k}: {v}")


@app.command("gnn-train")
def gnn_train(
    config: Path = CONFIG,
    model: str = GNN_KIND,
    seeds: int = typer.Option(3, "--seeds"),
    force: bool = typer.Option(False, "--force"),
) -> None:
    """Train a GNN (tuned parameters if `gnn-tune` ran, else the config defaults)."""
    from mule.models.gnn.train import run_gnn_train

    summary = _timed(
        "gnn-train", run_gnn_train, load_config(config), model, seeds, None, None, None, force
    )
    for s in summary["seeds"]:
        typer.echo(
            f"  seed {s['seed']}: val PR-AUC {s['val_pr_auc']:.4f} "
            f"(best epoch {s['best_epoch']}, {s['train_seconds']}s)"
        )


@app.command("gnn-tune")
def gnn_tune(
    config: Path = CONFIG,
    model: str = GNN_KIND,
    seeds: int = typer.Option(3, "--seeds"),
    force: bool = typer.Option(False, "--force"),
) -> None:
    """Run the shared grid (seed 0), then train the winner with N seeds."""
    from mule.models.gnn.train import run_gnn_tune

    result = _timed("gnn-tune", run_gnn_tune, load_config(config), model, seeds, force)
    for t in result["trials"]:
        typer.echo(f"  {t['params']}: val PR-AUC {t['val_pr_auc']:.4f} ({t['seconds']}s)")
    typer.echo(f"best: {result['best']['params']}")


@app.command("import-multignn")
def import_multignn_cmd(
    config: Path = CONFIG,
    log: Path = typer.Option(Path("runs/edge/multignn/train.log"), "--log"),
    name: str = typer.Option("multignn", "--name"),
) -> None:
    """Add IBM Multi-GNN's own run (from its log) to the transaction results."""
    from mule.eval.evaluate import write_results
    from mule.eval.multignn import import_multignn

    cfg = load_config(config)
    typer.echo(import_multignn(cfg, log, name))
    typer.echo(write_results(cfg, "edge").read_text())


if __name__ == "__main__":
    app()
