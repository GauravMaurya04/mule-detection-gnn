"""Stage 2: split manifest + account labels per window -> data/processed; report -> labels.md."""

from __future__ import annotations

import numpy as np
import pandas as pd

from mule.config import Config
from mule.io import read_parquet, write_parquet
from mule.labels.mule_rule import label_accounts, window_slice
from mule.labels.split import build_manifest, label_windows, save_manifest, window_bounds
from mule.report import Report

LABEL_SCHEMA = ["window", "account_id", "is_mule", "adjacent", "prior_mule"]
TX_COLUMNS = ["tx_id", "ts", "src_id", "dst_id", "is_laundering", "is_self"]


def k_values(active: int, mules: int, k_fixed: int = 100, k_share: float = 0.001) -> dict:
    """The three agreed K values: fixed 100, 0.1% of active accounts, and R-precision."""
    return {
        f"K={k_fixed}": k_fixed,
        f"K={k_share:.1%} of active": max(1, round(active * k_share)),
        "R-precision": mules,
    }


def labels_by_window(tx: pd.DataFrame, manifest: dict, rule: str) -> pd.DataFrame:
    """Labels for every window. prior_mule: a mule in some earlier window (repeat behaviour)."""
    frames, earlier_mules = [], np.array([], dtype=np.int64)
    for name in label_windows(manifest):
        start, end = window_bounds(manifest, name)
        labels = label_accounts(tx, start, end, rule=rule)
        labels["prior_mule"] = np.isin(labels["account_id"].to_numpy(), earlier_mules)
        labels.insert(0, "window", name)
        frames.append(labels)
        if name not in ("test", "test_no_tail"):  # test windows never feed "earlier"
            mule_ids = labels.loc[labels["is_mule"], "account_id"].to_numpy()
            earlier_mules = np.union1d(earlier_mules, mule_ids)
    out = pd.concat(frames, ignore_index=True)
    out["window"] = pd.Categorical(out["window"], categories=label_windows(manifest))
    return out


def window_summary(tx: pd.DataFrame, manifest: dict, labels: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name in label_windows(manifest):
        start, end = window_bounds(manifest, name)
        w = window_slice(tx, start, end)
        lab = labels.loc[labels["window"] == name]
        mule = lab["is_mule"].to_numpy()
        repeat = int((mule & lab["prior_mule"].to_numpy()).sum())
        rows.append(
            {
                "window": name,
                "from": str(start.date()),
                "to (excl.)": str(end.date()) if end == end.normalize() else str(end),
                "transactions": len(w),
                "laundering_tx": int(w["is_laundering"].sum()),
                "active_accounts": len(lab),
                "mules": int(mule.sum()),
                "mules_rule_A": int(label_accounts(tx, start, end, rule="within")["is_mule"].sum()),
                "adjacent": int(lab["adjacent"].sum()),
                "repeat_mules": repeat,
                "first_time_mules": int(mule.sum()) - repeat,
            }
        )
    return pd.DataFrame(rows)


def run_labels(cfg: Config) -> dict:
    tx = read_parquet(cfg.paths.interim_dir / "transactions.parquet", columns=TX_COLUMNS)
    manifest = build_manifest(
        tx["ts"], cfg.split.shares, cfg.split.no_tail_end, cfg.split.train_snapshot_days
    )
    manifest["config_hash"] = cfg.section_hash("dataset", "split", "labels")
    save_manifest(manifest, cfg.paths.processed_dir / "split_manifest.json")

    labels = labels_by_window(tx, manifest, cfg.labels.rule)
    write_parquet(labels, cfg.paths.processed_dir / "account_labels.parquet", LABEL_SCHEMA)

    summary = window_summary(tx, manifest, labels)
    snaps = summary[summary["window"].isin(manifest["train_snapshots"])]
    train_mules = labels[labels["window"].isin(manifest["train_snapshots"]) & labels["is_mule"]]

    rep = Report("Labels report")
    rep.section("Split")
    rep.item("Method", "IBM Multi-GNN day-based search")
    rep.item("Cut days", manifest["cut_days"])
    rep.item("Actual shares of transactions", manifest["actual_shares"])
    rep.item("Training snapshots", ", ".join(manifest["train_snapshots"]))
    rep.item("Label rule", cfg.labels.rule)
    rep.section("Windows")
    rep.table(summary)
    rep.item("Training positive rows (all snapshots)", int(snaps["mules"].sum()))
    rep.item("Distinct mule accounts behind them", int(train_mules["account_id"].nunique()))
    rep.note(
        "mules_rule_A uses the stricter rule (receipt and send both inside the window). "
        "repeat_mules were also mules in an earlier window; test windows never count as earlier."
    )
    rep.section("K values per evaluation window")
    k_rows = [
        {
            "window": r["window"],
            **k_values(
                r["active_accounts"], r["mules"], cfg.evaluation.k_fixed, cfg.evaluation.k_share
            ),
        }
        for r in summary.to_dict(orient="records")
        if r["window"] not in manifest["train_snapshots"]
    ]
    rep.table(pd.DataFrame(k_rows))
    cfg.paths.reports_dir.mkdir(parents=True, exist_ok=True)
    (cfg.paths.reports_dir / "labels.md").write_text(rep.text())
    return {"windows": summary.to_dict(orient="records")}
