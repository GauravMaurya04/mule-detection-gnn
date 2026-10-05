"""Stage 2: split manifest + account labels per window -> data/processed; report -> labels.md."""

from __future__ import annotations

import numpy as np
import pandas as pd

from mule.config import Config
from mule.io import read_parquet, write_parquet
from mule.labels.mule_rule import label_accounts, window_slice
from mule.labels.split import ALL_WINDOWS, build_manifest, save_manifest, window_bounds
from mule.report import Report

LABEL_SCHEMA = ["window", "account_id", "is_mule", "adjacent", "unseen"]
TX_COLUMNS = ["tx_id", "ts", "src_id", "dst_id", "is_laundering", "is_self"]


def k_values(active: int, mules: int) -> dict[str, int]:
    """The three agreed K values: fixed 100, 0.1% of active accounts, and R-precision."""
    return {"K=100": 100, "K=0.1% of active": max(1, round(active * 0.001)), "R-precision": mules}


def labels_by_window(tx: pd.DataFrame, manifest: dict, rule: str) -> pd.DataFrame:
    frames, seen = [], np.array([], dtype=np.int64)
    for name in ALL_WINDOWS:
        start, end = window_bounds(manifest, name)
        labels = label_accounts(tx, start, end, rule=rule)
        # Unseen = first active in this window. test_no_tail compares against train + val too.
        labels["unseen"] = (
            ~np.isin(labels["account_id"].to_numpy(), seen) if name != "train" else False
        )
        labels.insert(0, "window", name)
        frames.append(labels)
        if name in ("train", "val"):
            seen = np.union1d(seen, labels["account_id"].to_numpy())
    out = pd.concat(frames, ignore_index=True)
    out["window"] = pd.Categorical(out["window"], categories=list(ALL_WINDOWS))
    return out


def window_summary(tx: pd.DataFrame, manifest: dict, labels: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name in ALL_WINDOWS:
        start, end = window_bounds(manifest, name)
        w = window_slice(tx, start, end)
        lab = labels.loc[labels["window"] == name]
        rule_a = label_accounts(tx, start, end, rule="within")
        unseen = lab["unseen"].to_numpy()
        rows.append(
            {
                "window": name,
                "from": str(start.date()),
                "to (excl.)": str(end.date()) if end.time() == pd.Timestamp(0).time() else str(end),
                "transactions": len(w),
                "laundering_tx": int(w["is_laundering"].sum()),
                "active_accounts": len(lab),
                "mules": int(lab["is_mule"].sum()),
                "mules_rule_A": int(rule_a["is_mule"].sum()),
                "adjacent": int(lab["adjacent"].sum()),
                "unseen_accounts": np.nan if name == "train" else int(unseen.sum()),
                "unseen_mules": (
                    np.nan if name == "train" else int((unseen & lab["is_mule"].to_numpy()).sum())
                ),
            }
        )
    return pd.DataFrame(rows)


def run_labels(cfg: Config) -> dict:
    tx = read_parquet(cfg.paths.interim_dir / "transactions.parquet", columns=TX_COLUMNS)
    manifest = build_manifest(tx["ts"], cfg.split.shares, cfg.split.no_tail_end)
    manifest["config_hash"] = cfg.section_hash("dataset", "split", "labels")
    save_manifest(manifest, cfg.paths.processed_dir / "split_manifest.json")

    labels = labels_by_window(tx, manifest, cfg.labels.rule)
    write_parquet(labels, cfg.paths.processed_dir / "account_labels.parquet", LABEL_SCHEMA)

    summary = window_summary(tx, manifest, labels)
    rep = Report("Labels report")
    rep.section("Split")
    rep.item("Method", "IBM Multi-GNN day-based search")
    rep.item("Cut days", manifest["cut_days"])
    rep.item("Actual shares of transactions", manifest["actual_shares"])
    rep.item("Label rule", cfg.labels.rule)
    rep.section("Windows")
    rep.table(summary)
    rep.note("mules_rule_A uses the stricter rule (receipt and send both inside the window).")
    rep.section("K values per evaluation window")
    k_rows = [
        {"window": r["window"], **k_values(r["active_accounts"], r["mules"])}
        for r in summary.to_dict(orient="records")
        if r["window"] != "train"
    ]
    rep.table(pd.DataFrame(k_rows))
    cfg.paths.reports_dir.mkdir(parents=True, exist_ok=True)
    (cfg.paths.reports_dir / "labels.md").write_text(rep.text())
    return {"windows": summary.to_dict(orient="records")}
