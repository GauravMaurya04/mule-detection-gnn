"""A small synthetic dataset already in the interim format, with laundering chains every day,
so every window has mules and the model stages can be tested end to end on CPU in seconds."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from mule.io import ACCOUNT_INDEX_SCHEMA, TX_SCHEMA

FORMATS = ["ACH", "Bitcoin", "Cash", "Cheque", "Credit Card", "Reinvestment", "Wire"]


def make_interim(root: Path, days: int = 10, accounts: int = 400, seed: int = 0) -> Path:
    rng = np.random.default_rng(seed)
    t0 = pd.Timestamp("2022-09-01")
    rows = []
    for _ in range(days * 600):  # background traffic
        s, d = rng.integers(0, accounts, 2)
        rows.append(
            (
                t0 + pd.Timedelta(minutes=int(rng.integers(0, days * 1440))),
                s,
                d,
                float(rng.lognormal(5, 1.5)),
                rng.choice(FORMATS[1:]),
                0,
            )
        )
    for day in range(days):  # 6 chains a day: source -> m1 -> m2 -> sink, ACH, hours apart
        for _ in range(6):
            src, m1, m2, sink = rng.choice(accounts, 4, replace=False)
            start = t0 + pd.Timedelta(days=day, hours=int(rng.integers(0, 12)))
            amount = float(rng.uniform(2000, 9000))
            for hop, (a, b) in enumerate([(src, m1), (m1, m2), (m2, sink)]):
                ts = start + pd.Timedelta(hours=5 * hop)
                rows.append((ts, a, b, amount * (0.95**hop), "ACH", 1))
    tx = pd.DataFrame(
        rows, columns=["ts", "src_id", "dst_id", "amount_usd", "payment_format", "is_laundering"]
    )
    tx = tx.sort_values("ts", kind="stable").reset_index(drop=True)
    n = len(tx)
    tx["tx_id"] = np.arange(n, dtype=np.int32)
    tx["raw_row"] = tx["tx_id"]
    tx["src_bank"] = (tx["src_id"] % 9).astype("int32")
    tx["dst_bank"] = (tx["dst_id"] % 9).astype("int32")
    tx["amount_paid"] = tx["amount_received"] = tx["amount_usd"]
    tx["currency_paid"] = pd.Categorical(["US Dollar"] * n)
    tx["currency_received"] = pd.Categorical(np.where(rng.random(n) < 0.9, "US Dollar", "Euro"))
    tx["payment_format"] = pd.Categorical(tx["payment_format"], categories=FORMATS)
    tx["is_laundering"] = tx["is_laundering"].astype("int8")
    tx["is_self"] = tx["src_id"] == tx["dst_id"]
    tx["cross_bank"] = tx["src_bank"] != tx["dst_bank"]
    for col in ("src_id", "dst_id"):
        tx[col] = tx[col].astype("int32")

    interim = root / "data" / "interim"
    interim.mkdir(parents=True)
    tx[TX_SCHEMA].to_parquet(interim / "transactions.parquet", index=False)
    index = pd.DataFrame(
        {
            "account_id": np.arange(accounts, dtype=np.int32),
            "bank": np.arange(accounts) % 9,
            "account": [f"A{i:05d}" for i in range(accounts)],
        }
    )
    index[ACCOUNT_INDEX_SCHEMA].to_parquet(interim / "account_index.parquet", index=False)
    (root / "base.yaml").write_text(
        f"""dataset: SYN
paths:
  raw_dir: {root / "data" / "raw"}
  interim_dir: {interim}
  processed_dir: {root / "data" / "processed"}
  reports_dir: {root / "reports"}
  runs_dir: {root / "runs"}
split:
  shares: [0.6, 0.2, 0.2]
  no_tail_end: "2022-09-10"
  train_snapshot_days: 2
models:
  xgb:
    n_estimators: 30
    early_stopping_rounds: 10
evaluation:
  bootstrap: 50
tracking:
  mlflow_uri: "sqlite:///{root / "mlflow.db"}"
"""
    )
    return root / "base.yaml"
