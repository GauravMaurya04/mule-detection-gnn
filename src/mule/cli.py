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


@app.command("tune-xgb")
def tune_xgb(
    config: Path = CONFIG,
    task: str = TASK,
    base: str = typer.Option(..., "--base", help="Champion feature set, e.g. xgb_graph."),
    trials: int = typer.Option(20, "--trials"),
    seeds: int = typer.Option(3, "--seeds"),
    force: bool = typer.Option(False, "--force"),
) -> None:
    """Random search on validation for the champion's features, then N seeds of the best."""
    from mule.models.tune import run_xgb_tune

    result = _timed("tune-xgb", run_xgb_tune, load_config(config), task, base, trials, seeds, force)
    best = result["best"]
    typer.echo(f"best: {best['label']} val PR-AUC {best['val_pr_auc']:.4f} {best['params']}")
    for s in result["final"]["seeds"]:
        typer.echo(f"  xgb_tuned seed {s['seed']}: val PR-AUC {s['val_pr_auc']:.4f}")


@app.command()
def stack(
    config: Path = CONFIG,
    base: str = typer.Option("xgb_graph", "--base", help="Account champion feature set."),
    seeds: int = typer.Option(3, "--seeds"),
    refresh_oof: bool = typer.Option(False, "--refresh-oof", help="Rebuild the GNN scores."),
    force: bool = typer.Option(False, "--force"),
) -> None:
    """Out-of-fold GraphSAGE scores, then xgb_stacked (account task)."""
    from mule.models.tune import run_stack

    result = _timed(
        "stack", run_stack, load_config(config), base, seeds, "sage", refresh_oof, force
    )
    if result["oof"]:
        for f in result["oof"]["folds"]:
            typer.echo(f"  fold holding out {f['held_out']}: val PR-AUC {f['val_pr_auc']:.4f}")
    typer.echo(f"xgb_stacked settings from: {result['params_from']}")
    for s in result["final"]["seeds"]:
        typer.echo(f"  xgb_stacked seed {s['seed']}: val PR-AUC {s['val_pr_auc']:.4f}")


@app.command()
def calibrate(
    config: Path = CONFIG,
    task: str = TASK,
    model: str = typer.Option(..., "--model", "-m"),
    force: bool = typer.Option(False, "--force"),
) -> None:
    """Seed-averaged, calibrated copy of a model: <model>_cal (Platt or isotonic, by CV Brier)."""
    from mule.eval.combine import run_calibrate

    typer.echo(run_calibrate(load_config(config), task, model, force))


@app.command()
def ensemble(
    config: Path = CONFIG,
    task: str = TASK,
    a: str = typer.Option(..., "--a"),
    b: str = typer.Option(..., "--b"),
    force: bool = typer.Option(False, "--force"),
) -> None:
    """Weighted average of two calibrated models; the weight is chosen on validation PR-AUC."""
    from mule.eval.combine import run_ensemble

    typer.echo(run_ensemble(load_config(config), task, a, b, force=force))


@app.command()
def explain(
    config: Path = CONFIG,
    window: str = typer.Option("test", "--window", "-w"),
    top: int = typer.Option(0, "--top", help="Accounts to explain (default: config explain.top)."),
    no_gnn: bool = typer.Option(False, "--no-gnn", help="Skip evidence transfers (no PyTorch)."),
) -> None:
    """Reason codes + evidence transfers for the champion's top accounts, with checks."""
    from mule.explain.stage import run_explain

    cfg = load_config(config)
    _timed("explain", run_explain, cfg, window, top or None, not no_gnn)
    typer.echo((cfg.paths.reports_dir / f"explanations_{window}.md").read_text())


@app.command()
def why(
    config: Path = CONFIG,
    account: str = typer.Option(..., "--account", "-a", help="BANK:ACCOUNT"),
    window: str = typer.Option("test", "--window", "-w"),
    no_gnn: bool = typer.Option(False, "--no-gnn"),
) -> None:
    """Why does the champion score this account the way it does?"""
    import json as _json

    import pandas as pd

    from mule.explain.stage import explain_accounts
    from mule.io import read_parquet

    cfg = load_config(config)
    bank, _, acct = account.partition(":")
    index = read_parquet(cfg.paths.interim_dir / "account_index.parquet")
    hit = index[(index["bank"] == int(bank)) & (index["account"] == acct)]
    if hit.empty:
        raise typer.BadParameter(f"account {account} not found")
    aid = int(hit["account_id"].iloc[0])
    scores = pd.read_parquet(
        cfg.paths.runs_dir / "node" / cfg.explain.champion / "seed0" / f"scores_{window}.parquet"
    )
    if aid not in set(scores["account_id"]):
        typer.echo(f"{account} had no transactions in window {window}.")
        raise typer.Exit()
    score = float(scores.loc[scores["account_id"] == aid, "score"].iloc[0])
    rank = int((scores["score"] > score).sum()) + 1
    row = explain_accounts(cfg, window, [aid], not no_gnn).iloc[0]
    typer.echo(f"{account}: score {score:.4f}, rank {rank:,} of {len(scores):,} in {window}")
    typer.echo(f"GNN share of added risk: {row['gnn_share']:.0%}")
    codes = _json.loads(row["codes"])
    typer.echo("Reason codes:" if codes else "No reason code fires (pattern not covered).")
    for c in codes:
        typer.echo(f"  {c['code']} {c['name']}: {c['text']}")
    transfers = _json.loads(row["evidence"])
    if transfers:
        typer.echo("Evidence transfers (from the network model):")
        for t in transfers:
            typer.echo(
                f"  {t['ts']}  {t['direction']:>3}  {t['amount_usd']:>12,.2f} USD  "
                f"{t['format']:<12} {t['counterparty']}"
            )


@app.command("pattern-recall")
def pattern_recall(
    config: Path = CONFIG,
    node_model: str = typer.Option("ens_xgb_graph__gnn_sage", "--node-model"),
    edge_model: str = typer.Option("ens_xgb_tuned__gnn_sage", "--edge-model"),
    final: bool = typer.Option(False, "--final", help="Required: reads test labels."),
) -> None:
    """Recall by laundering pattern for the champions (test and no-tail windows)."""
    from mule.eval.day9 import run_pattern_recall

    typer.echo(run_pattern_recall(load_config(config), node_model, edge_model, final=final))


@app.command()
def stress(
    config: Path = CONFIG,
    delay_hours: float = typer.Option(48, "--delay-hours"),
    final: bool = typer.Option(False, "--final", help="Required: reads test labels."),
    no_gnn: bool = typer.Option(False, "--no-gnn"),
) -> None:
    """Delay mules' outgoing laundering and re-score the frozen account champion."""
    from mule.eval.day9 import run_stress

    cfg = load_config(config)
    typer.echo(_timed("stress", run_stress, cfg, delay_hours, "test", final, not no_gnn))


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
    node_features: str = typer.Option(None, "--node-features", help="engineered or minimal."),
    params_from: str = typer.Option(None, "--params-from", help="Reuse tuning, e.g. gnn_sage."),
) -> None:
    """Train an account-task GNN (tuned parameters if available, else config defaults)."""
    from mule.models.gnn.train import run_gnn_train

    cfg = load_config(config)
    if node_features:
        cfg.gnn = cfg.gnn.model_copy(update={"node_features": node_features})
    summary = _timed(
        "gnn-train", run_gnn_train, cfg, model, seeds, None, None, None, force, params_from
    )
    for s in summary["seeds"]:
        typer.echo(
            f"  seed {s['seed']}: val PR-AUC {s['val_pr_auc']:.4f} "
            f"(best epoch {s['best_epoch']}, {s['train_seconds']}s)"
        )


@app.command("gnn-edge-train")
def gnn_edge_train(
    config: Path = CONFIG,
    model: str = GNN_KIND,
    seeds: int = typer.Option(3, "--seeds"),
    force: bool = typer.Option(False, "--force"),
) -> None:
    """Train the transaction-task GNN (encoder parameters transfer from account-task tuning)."""
    from mule.models.gnn.edge import run_gnn_edge_train

    summary = _timed("gnn-edge-train", run_gnn_edge_train, load_config(config), model, seeds, force)
    for s in summary["seeds"]:
        typer.echo(
            f"  seed {s['seed']}: val PR-AUC {s['val_pr_auc']:.4f} "
            f"(best epoch {s['best_epoch']}, {s['train_seconds']}s)"
        )


@app.command("add-paper-rows")
def add_paper_rows_cmd(config: Path = CONFIG) -> None:
    """Add the published Multi-GNN test F1 (AML Small HI) as reference rows."""
    from mule.eval.evaluate import write_results
    from mule.eval.multignn import add_paper_rows

    cfg = load_config(config)
    typer.echo(add_paper_rows(cfg))
    typer.echo(write_results(cfg, "edge").read_text())


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
