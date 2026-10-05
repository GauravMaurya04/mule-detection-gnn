"""Artifact read/write with schema checks, so a stage fails loudly on a stale or broken file."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

TX_SCHEMA = [
    "tx_id",
    "raw_row",
    "ts",
    "src_id",
    "dst_id",
    "src_bank",
    "dst_bank",
    "amount_paid",
    "currency_paid",
    "amount_received",
    "currency_received",
    "amount_usd",
    "payment_format",
    "is_laundering",
    "is_self",
    "cross_bank",
]
ACCOUNT_INDEX_SCHEMA = ["account_id", "bank", "account"]


def _check(df: pd.DataFrame, required: list[str] | None, name: str) -> None:
    if required:
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"{name}: missing columns {missing}")


def write_parquet(df: pd.DataFrame, path: Path, required: list[str] | None = None) -> Path:
    _check(df, required, path.name)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return path


def read_parquet(
    path: Path, required: list[str] | None = None, columns: list[str] | None = None
) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{path} not found: run the earlier pipeline stage first")
    df = pd.read_parquet(path, columns=columns)
    _check(df, required, path.name)
    return df
