"""Window graphs for the GNNs, built from the same saved artifacts as the tabular models.

One graph per window (train snapshots, val, test, test_no_tail). Nodes are the accounts that
appear in the window's feature span (window plus lookback) or are labelled in the window;
edges are the span's non-self transfers, sender -> receiver. Node features are the 48 account
features (or a constant, for the ablation) standardised with train-snapshot statistics, plus
an `active` flag; accounts seen only in the lookback have no feature row and get zeros.
Edge features never include absolute time: amount, hour and weekday (as sin/cos), the gap
since the sender's previous transfer, format, cross-bank and cross-currency flags.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch

from mule.config import Config
from mule.features.account import PAYMENT_FORMATS, _slug
from mule.io import TX_SCHEMA, read_parquet
from mule.labels.split import EVAL_WINDOWS, load_manifest, window_bounds
from mule.models.data import _check_features, edge_own_features

EDGE_COLUMNS = [
    "log_usd",
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "log_gap_h",
    "gap_missing",
    "cross_bank",
    "cross_currency",
    *[f"fmt_{_slug(f)}" for f in PAYMENT_FORMATS],
]
CONTINUOUS_EDGE = [EDGE_COLUMNS.index("log_usd"), EDGE_COLUMNS.index("log_gap_h")]


@dataclass
class WindowGraph:
    name: str
    nodes: np.ndarray  # account_id of every node, sorted
    x: torch.Tensor  # [N, F] node features
    edge_index: torch.Tensor  # [2, E] sender -> receiver (node positions)
    edge_attr: torch.Tensor  # [E, D]
    label_nodes: torch.Tensor  # node positions of the window's labelled accounts
    y: torch.Tensor  # float labels, aligned with label_nodes
    meta: pd.DataFrame  # account_id, is_mule, adjacent, prior_mule; row i <-> label_nodes[i]


def edge_table(tx: pd.DataFrame) -> np.ndarray:
    """GNN edge features for every transaction row (float32, columns = EDGE_COLUMNS)."""
    own = edge_own_features(tx)
    hour = own["e_hour"].to_numpy() * (2 * np.pi / 24)
    dow = own["e_dow"].to_numpy() * (2 * np.pi / 7)
    gap = own["e_src_gap_h"].to_numpy()
    cols = [
        own["e_log_usd"].to_numpy(),
        np.sin(hour),
        np.cos(hour),
        np.sin(dow),
        np.cos(dow),
        np.log1p(np.nan_to_num(gap, nan=0.0)),
        np.isnan(gap).astype(np.float32),
        own["e_cross_bank"].to_numpy(),
        own["e_cross_currency"].to_numpy(),
        *[own[f"e_fmt_{_slug(f)}"].to_numpy() for f in PAYMENT_FORMATS],
    ]
    return np.column_stack(cols).astype(np.float32)


def _safe_std(values: np.ndarray, axis=0) -> tuple[np.ndarray, np.ndarray]:
    with warnings.catch_warnings():  # an all-missing column (e.g. no accounts file) is fine
        warnings.simplefilter("ignore", RuntimeWarning)
        mean = np.nanmean(values, axis=axis)
        std = np.nanstd(values, axis=axis)
    mean = np.nan_to_num(mean, nan=0.0)
    std = np.where(~np.isfinite(std) | (std < 1e-8), 1.0, std)
    return mean.astype(np.float32), std.astype(np.float32)


def build_graphs(cfg: Config, node_features: str = "engineered") -> tuple[dict, dict]:
    """All window graphs plus the train-only standardisation statistics (saved with each run)."""
    if node_features not in ("engineered", "minimal"):
        raise ValueError("node_features must be 'engineered' or 'minimal'")
    manifest = load_manifest(cfg.paths.processed_dir / "split_manifest.json")
    columns = _check_features(cfg)
    tx = read_parquet(cfg.paths.interim_dir / "transactions.parquet", TX_SCHEMA)
    edges_all = edge_table(tx)
    labels = read_parquet(cfg.paths.processed_dir / "account_labels.parquet")
    labels["window"] = labels["window"].astype(str)
    feats = read_parquet(cfg.paths.processed_dir / "account_features.parquet")
    feats["window"] = feats["window"].astype(str)
    lookback = pd.Timedelta(hours=cfg.features.lookback_hours)
    ts, not_self = tx["ts"], ~tx["is_self"].to_numpy()
    src_all, dst_all = tx["src_id"].to_numpy(), tx["dst_id"].to_numpy()

    raw = {}
    for w in [*manifest["train_snapshots"], *EVAL_WINDOWS]:
        start, end = window_bounds(manifest, w)
        mask = ((ts >= start - lookback) & (ts < end)).to_numpy() & not_self
        lab = labels[labels["window"] == w].sort_values("account_id").reset_index(drop=True)
        nodes = np.union1d(np.union1d(src_all[mask], dst_all[mask]), lab["account_id"].to_numpy())
        feat_w = feats[feats["window"] == w].set_index("account_id")[columns]
        raw[w] = {
            "nodes": nodes,
            "src": np.searchsorted(nodes, src_all[mask]),
            "dst": np.searchsorted(nodes, dst_all[mask]),
            "edge": edges_all[mask],
            "X": feat_w.reindex(nodes).to_numpy(dtype=np.float32),
            "active": np.isin(nodes, lab["account_id"].to_numpy()),
            "lab": lab,
        }
    del tx, edges_all, feats

    snaps = manifest["train_snapshots"]
    node_mean, node_std = _safe_std(np.vstack([raw[s]["X"][raw[s]["active"]] for s in snaps]))
    train_edges = np.vstack([raw[s]["edge"][:, CONTINUOUS_EDGE] for s in snaps])
    edge_mean, edge_std = _safe_std(train_edges)
    del train_edges

    graphs = {}
    for w, r in raw.items():
        n = len(r["nodes"])
        active = r["active"].astype(np.float32)[:, None]
        if node_features == "engineered":
            X = np.nan_to_num((r["X"] - node_mean) / node_std, nan=0.0)  # mean-impute missing
            x = np.hstack([X, active])
        else:
            x = np.hstack([np.ones((n, 1), np.float32), active])
        edge = r["edge"].copy()
        edge[:, CONTINUOUS_EDGE] = (edge[:, CONTINUOUS_EDGE] - edge_mean) / edge_std
        lab = r["lab"]
        graphs[w] = WindowGraph(
            name=w,
            nodes=r["nodes"],
            x=torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32)),
            edge_index=torch.from_numpy(np.vstack([r["src"], r["dst"]]).astype(np.int64)),
            edge_attr=torch.from_numpy(np.ascontiguousarray(edge, dtype=np.float32)),
            label_nodes=torch.from_numpy(np.searchsorted(r["nodes"], lab["account_id"].to_numpy())),
            y=torch.from_numpy(lab["is_mule"].to_numpy(dtype=np.float32)),
            meta=lab[["account_id", "is_mule", "adjacent", "prior_mule"]],
        )
    stats = {
        "node_features": node_features,
        "feature_columns": columns,
        "node_mean": node_mean.tolist(),
        "node_std": node_std.tolist(),
        "edge_columns": EDGE_COLUMNS,
        "edge_mean": edge_mean.tolist(),
        "edge_std": edge_std.tolist(),
    }
    return graphs, stats
