"""Parse HI-Small_Patterns.txt (laundering attempts grouped by pattern type) and match it
to the transactions.

The file holds blocks like:
    BEGIN LAUNDERING ATTEMPT - CYCLE:  Max 4 hops
    2022/09/01 00:06,021174,800737690,012,80011F990,2848.96,Euro,2848.96,Euro,ACH,1
    END LAUNDERING ATTEMPT - CYCLE
It covers only part of all laundering, so pattern-level results describe that subset.
"""

from __future__ import annotations

import csv
import re
from pathlib import Path

import numpy as np
import pandas as pd

from mule.data.ingest import TS_FORMAT

BEGIN = re.compile(r"^BEGIN LAUNDERING ATTEMPT\s*-\s*([A-Z][A-Z\-]*)")
END_PREFIX = "END LAUNDERING ATTEMPT"
FIELDS = [
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
]


def read_patterns(path: Path) -> pd.DataFrame:
    rows: list[list] = []
    attempt, pattern = -1, None
    with open(path, newline="") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            begin = BEGIN.match(line)
            if begin:
                attempt += 1
                pattern = begin.group(1)
                continue
            if line.startswith(END_PREFIX):
                pattern = None
                continue
            if pattern is None:
                continue
            fields = next(csv.reader([line]))
            if len(fields) >= len(FIELDS):
                rows.append([attempt, pattern, *fields[: len(FIELDS)]])

    df = pd.DataFrame(rows, columns=["attempt_id", "pattern", *FIELDS])
    df["ts"] = pd.to_datetime(df["ts"], format=TS_FORMAT)
    for col in ("src_bank", "dst_bank"):
        df[col] = df[col].astype("int64").astype("int32")  # handles zero-padded ids like "021174"
    for col in ("amount_received", "amount_paid"):
        df[col] = df[col].astype(float)
    df["pattern_row"] = np.arange(len(df))
    return df


def match_patterns(patterns: pd.DataFrame, tx: pd.DataFrame, index: pd.DataFrame) -> pd.DataFrame:
    """Attach tx_id and the transaction's label to each pattern row (NaN tx_id = not found)."""
    p = patterns.copy()
    lookup = index.assign(account=index["account"].astype(str))
    for side in ("src", "dst"):
        ids = lookup.rename(
            columns={
                "bank": f"{side}_bank",
                "account": f"{side}_account",
                "account_id": f"{side}_id",
            }
        )
        p = p.merge(ids, on=[f"{side}_bank", f"{side}_account"], how="left")

    candidates = tx.loc[
        tx["src_id"].isin(p["src_id"].dropna().unique()),
        ["tx_id", "ts", "src_id", "dst_id", "amount_paid", "currency_paid", "is_laundering"],
    ].copy()
    candidates["currency_paid"] = candidates["currency_paid"].astype(str)
    candidates["amount_key"] = candidates["amount_paid"].round(2)
    p["amount_key"] = p["amount_paid"].round(2)
    for col in ("src_id", "dst_id"):
        p[col] = p[col].astype("Int64")
        candidates[col] = candidates[col].astype("Int64")

    keys = ["ts", "src_id", "dst_id", "currency_paid", "amount_key"]
    matched = p.merge(candidates[[*keys, "tx_id", "is_laundering"]], on=keys, how="left")
    matched["n_matches"] = matched.groupby("pattern_row")["tx_id"].transform("count")
    return matched.drop_duplicates("pattern_row").reset_index(drop=True)
