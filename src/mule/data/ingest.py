"""Stage 1: raw AMLworld CSVs -> typed Parquet with integer account ids and USD amounts."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pandas as pd

from mule.config import Config
from mule.data.currency import estimate_usd_rates, save_rates, to_usd
from mule.io import ACCOUNT_INDEX_SCHEMA, TX_SCHEMA, write_parquet

EXPECTED_HEADER = [
    "Timestamp",
    "From Bank",
    "Account",
    "To Bank",
    "Account",
    "Amount Received",
    "Receiving Currency",
    "Amount Paid",
    "Payment Currency",
    "Payment Format",
    "Is Laundering",
]
COLUMNS = [
    "ts",
    "src_bank",
    "src_account",
    "dst_bank",
    "dst_account",
    "amount_received",
    "currency_received",
    "amount_paid",
    "currency_paid",
    "payment_format",
    "is_laundering",
]
DTYPES = {
    "ts": "string",
    "src_bank": "int32",
    "src_account": "string[pyarrow]",
    "dst_bank": "int32",
    "dst_account": "string[pyarrow]",
    "amount_received": "float64",
    "currency_received": "category",
    "amount_paid": "float64",
    "currency_paid": "category",
    "payment_format": "category",
    "is_laundering": "int8",
}
TS_FORMAT = "%Y/%m/%d %H:%M"
ACCOUNTS_COLUMNS = {
    "Bank Name": "bank_name",
    "Bank ID": "bank_id",
    "Account Number": "account",
    "Entity ID": "entity_id",
    "Entity Name": "entity_name",
}


def check_header(path: Path) -> None:
    with open(path, newline="") as f:
        header = [h.strip() for h in next(csv.reader(f))]
    if header != EXPECTED_HEADER:
        raise ValueError(
            f"Unexpected header in {path.name}\n  got:      {header}\n  expected: {EXPECTED_HEADER}"
        )


def read_transactions(path: Path) -> pd.DataFrame:
    """Read the transactions CSV. The raw header repeats 'Account', so we supply our own names."""
    check_header(path)
    df = pd.read_csv(path, header=0, names=COLUMNS, dtype=DTYPES)
    df["ts"] = pd.to_datetime(df["ts"], format=TS_FORMAT)
    df["raw_row"] = np.arange(len(df), dtype=np.int32)
    return df


def assign_account_ids(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """An account is the (bank, account) pair. Ids are dense ints, sorted by that pair."""
    n = len(df)
    both = pd.DataFrame(
        {
            "bank": np.concatenate([df["src_bank"].to_numpy(), df["dst_bank"].to_numpy()]),
            "account": pd.concat([df["src_account"], df["dst_account"]], ignore_index=True),
        }
    )
    ids = both.groupby(["bank", "account"], sort=True).ngroup().to_numpy(dtype=np.int32)
    df = df.assign(src_id=ids[:n], dst_id=ids[n:]).drop(columns=["src_account", "dst_account"])
    index = (
        both.assign(account_id=ids)
        .drop_duplicates("account_id")
        .sort_values("account_id")
        .reset_index(drop=True)[ACCOUNT_INDEX_SCHEMA]
    )
    index["account"] = index["account"].astype(str)
    return df, index


def read_accounts_meta(path: Path) -> pd.DataFrame:
    """The accounts file maps each (bank, account) to its owning entity."""
    meta = pd.read_csv(path, dtype=str)
    meta.columns = [c.strip() for c in meta.columns]
    missing = set(ACCOUNTS_COLUMNS) - set(meta.columns)
    if missing:
        raise ValueError(
            f"{path.name}: expected columns {sorted(missing)}, got {list(meta.columns)}"
        )
    meta = meta.rename(columns=ACCOUNTS_COLUMNS)
    meta["bank"] = pd.to_numeric(meta["bank_id"], errors="coerce").astype("Int32")
    return meta


def run_ingest(cfg: Config) -> dict:
    raw, interim = cfg.paths.raw_dir, cfg.paths.interim_dir
    tx = read_transactions(raw / f"{cfg.dataset}_Trans.csv")
    tx, index = assign_account_ids(tx)
    tx["is_self"] = tx["src_id"] == tx["dst_id"]
    tx["cross_bank"] = tx["src_bank"] != tx["dst_bank"]

    rates = estimate_usd_rates(tx)
    tx["amount_usd"] = to_usd(tx["amount_paid"], tx["currency_paid"], rates)

    tx = tx.sort_values(["ts", "raw_row"], kind="stable").reset_index(drop=True)
    tx["tx_id"] = np.arange(len(tx), dtype=np.int32)
    tx = tx[TX_SCHEMA]

    write_parquet(tx, interim / "transactions.parquet", TX_SCHEMA)
    write_parquet(index, interim / "account_index.parquet", ACCOUNT_INDEX_SCHEMA)
    save_rates(rates, interim / "usd_rates.json")

    summary = {"transactions": len(tx), "accounts": len(index)}
    accounts_path = raw / f"{cfg.dataset}_accounts.csv"
    try:
        meta = read_accounts_meta(accounts_path)
        write_parquet(meta, interim / "accounts_meta.parquet")
        summary["accounts_file_rows"] = len(meta)
    except (FileNotFoundError, ValueError) as err:
        print(f"WARNING: accounts file skipped: {err}")
    missing_rates = [c for c, r in rates.items() if pd.isna(r)]
    if missing_rates:
        print(f"WARNING: no USD rate for {missing_rates}; their amount_usd is NaN")
    return summary
