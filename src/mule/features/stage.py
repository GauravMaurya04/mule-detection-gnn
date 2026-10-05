"""Stage 3: Tier 1 account features for every window -> data/processed/account_features.parquet."""

from __future__ import annotations

import json
import time

import pandas as pd

from mule.config import Config
from mule.features.account import GROUPS, account_features, feature_groups
from mule.io import TX_SCHEMA, read_parquet, write_parquet
from mule.labels.split import ALL_WINDOWS, load_manifest, window_bounds


def run_features(cfg: Config) -> dict:
    processed = cfg.paths.processed_dir
    manifest = load_manifest(processed / "split_manifest.json")
    expected = cfg.section_hash("dataset", "split", "labels")
    if manifest.get("config_hash") != expected:
        raise RuntimeError("split_manifest.json is stale for this config: rerun `mule labels`")
    tx = read_parquet(cfg.paths.interim_dir / "transactions.parquet", TX_SCHEMA)

    frames, timings = [], {}
    for name in ALL_WINDOWS:
        t0 = time.perf_counter()
        start, end = window_bounds(manifest, name)
        feats = account_features(
            tx, start, end, cfg.features.lookback_hours, cfg.features.forward_hours
        )
        frames.append(feats.reset_index().assign(window=name))
        timings[name] = round(time.perf_counter() - t0, 1)
    out = pd.concat(frames, ignore_index=True)
    out["window"] = pd.Categorical(out["window"], categories=list(ALL_WINDOWS))
    columns = [c for c in out.columns if c not in ("window", "account_id")]
    out = out[["window", "account_id", *columns]]
    write_parquet(out, processed / "account_features.parquet")

    meta = {
        "config_hash": cfg.section_hash("dataset", "split", "labels", "features"),
        "feature_columns": columns,
        "groups": {GROUPS[k]: v for k, v in feature_groups(columns).items()},
        "rows_per_window": out["window"].value_counts().reindex(ALL_WINDOWS).to_dict(),
        "seconds_per_window": timings,
    }
    (processed / "account_features.meta.json").write_text(json.dumps(meta, indent=2))
    return {"features": len(columns), **meta["rows_per_window"], "seconds": timings}
