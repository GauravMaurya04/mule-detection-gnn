"""Design matrices for both tasks, built only from saved artifacts (labels, features, transactions).

Account task: one row per (window, account). Training rows are the train snapshots stacked.
Transaction task: one row per transaction. Columns are the transaction's own features plus the
Tier 1 features of its sender and receiver *for the window it falls in*. Absolute timestamps are
never columns: only hour of day, day of week and the gap since the sender's previous transfer.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
import pandas as pd

from mule.config import Config
from mule.features.account import PAYMENT_FORMATS, _slug
from mule.io import TX_SCHEMA, read_parquet
from mule.labels.split import EVAL_WINDOWS, load_manifest, window_bounds

NODE_META = ["window", "account_id", "is_mule", "adjacent", "prior_mule"]


@dataclass
class TaskData:
    X: pd.DataFrame  # float32 feature matrix
    y: np.ndarray  # 0/1 labels
    meta: pd.DataFrame  # ids and evaluation flags, row-aligned with X


def _check_features(cfg: Config) -> list[str]:
    path = cfg.paths.processed_dir / "account_features.meta.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found: run `mule features` first")
    meta = json.loads(path.read_text())
    if meta["config_hash"] != cfg.section_hash("dataset", "split", "labels", "features"):
        raise RuntimeError("account features are stale for this config: rerun `mule features`")
    return meta["feature_columns"]


def load_node(cfg: Config) -> dict[str, TaskData]:
    manifest = load_manifest(cfg.paths.processed_dir / "split_manifest.json")
    columns = _check_features(cfg)
    labels = read_parquet(cfg.paths.processed_dir / "account_labels.parquet")
    feats = read_parquet(cfg.paths.processed_dir / "account_features.parquet")
    df = labels.merge(feats, on=["window", "account_id"], how="inner", validate="one_to_one")
    if len(df) != len(labels):
        raise RuntimeError("labels and features disagree on accounts: rerun labels + features")
    df["window"] = df["window"].astype(str)
    parts = {"train": df[df["window"].isin(manifest["train_snapshots"])]}
    parts.update({w: df[df["window"] == w] for w in EVAL_WINDOWS})
    return {
        name: TaskData(
            part[columns].astype("float32").reset_index(drop=True),
            part["is_mule"].to_numpy(dtype=np.int8),
            part[NODE_META].reset_index(drop=True),
        )
        for name, part in parts.items()
    }


def edge_own_features(tx: pd.DataFrame) -> pd.DataFrame:
    currencies = sorted(
        set(tx["currency_paid"].cat.categories) | set(tx["currency_received"].cat.categories)
    )
    paid = pd.Categorical(tx["currency_paid"].astype(str), categories=currencies).codes
    received = pd.Categorical(tx["currency_received"].astype(str), categories=currencies).codes
    gap = tx.groupby("src_id")["ts"].diff().dt.total_seconds() / 3600  # past only
    own = pd.DataFrame(
        {
            "e_log_usd": np.log1p(tx["amount_usd"].to_numpy()),
            "e_hour": tx["ts"].dt.hour.to_numpy(),
            "e_dow": tx["ts"].dt.dayofweek.to_numpy(),
            "e_src_gap_h": gap.to_numpy(),
            "e_cross_bank": tx["cross_bank"].to_numpy(),
            "e_cross_currency": paid != received,
            "e_is_self": tx["is_self"].to_numpy(),
            "e_cur_paid": paid,
            "e_cur_received": received,
        }
    )
    fmt = pd.Categorical(tx["payment_format"].astype(str), categories=PAYMENT_FORMATS).codes
    for code, name in enumerate(PAYMENT_FORMATS):
        own[f"e_fmt_{_slug(name)}"] = fmt == code
    return own.astype("float32")


def load_edge(cfg: Config, names=("train", *EVAL_WINDOWS)) -> dict[str, TaskData]:
    """Transaction rows with own + sender + receiver features. Each matrix is preallocated and
    filled in place (no hstack/concat copies): the train matrix alone is ~1.5 GB on HI-Small."""
    manifest = load_manifest(cfg.paths.processed_dir / "split_manifest.json")
    columns = _check_features(cfg)
    tx = read_parquet(cfg.paths.interim_dir / "transactions.parquet", TX_SCHEMA)
    own = edge_own_features(tx).to_numpy()
    feats = read_parquet(cfg.paths.processed_dir / "account_features.parquet")
    feats["window"] = feats["window"].astype(str)
    by_window = {w: g.set_index("account_id")[columns] for w, g in feats.groupby("window")}
    del feats
    n_own, n_acct = own.shape[1], len(columns)
    own_names = list(edge_own_features(tx.head(1)).columns)
    col_names = [*own_names, *[f"src_{c}" for c in columns], *[f"dst_{c}" for c in columns]]
    src_all, dst_all, ts = tx["src_id"].to_numpy(), tx["dst_id"].to_numpy(), tx["ts"]

    def mask_for(window: str) -> np.ndarray:
        start, end = window_bounds(manifest, window)
        return ((ts >= start) & (ts < end)).to_numpy()

    def build(windows: list[str]) -> TaskData:
        masks = [mask_for(w) for w in windows]
        n = int(sum(m.sum() for m in masks))
        X = np.empty((n, n_own + 2 * n_acct), dtype=np.float32)
        metas, row = [], 0
        for window, mask in zip(windows, masks, strict=True):
            k = int(mask.sum())
            acct = by_window[window]
            X[row : row + k, :n_own] = own[mask]
            X[row : row + k, n_own : n_own + n_acct] = acct.reindex(src_all[mask]).to_numpy()
            X[row : row + k, n_own + n_acct :] = acct.reindex(dst_all[mask]).to_numpy()
            metas.append(tx.loc[mask, ["tx_id", "is_laundering"]].assign(window=window))
            row += k
        meta = pd.concat(metas, ignore_index=True)
        return TaskData(
            pd.DataFrame(X, columns=col_names, copy=False),
            meta["is_laundering"].to_numpy(np.int8),
            meta,
        )

    out = {}
    for name in names:
        out[name] = build(list(manifest["train_snapshots"]) if name == "train" else [name])
    return out
