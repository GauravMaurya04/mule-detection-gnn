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


if __name__ == "__main__":
    app()
