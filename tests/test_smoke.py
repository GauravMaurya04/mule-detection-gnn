"""End-to-end: raw files in the real formats -> ingest -> Phase 0 report. Runs on CPU in seconds."""

import json

import pandas as pd

from mule.config import load_config
from mule.data.ingest import run_ingest
from mule.features.stage import run_features
from mule.labels.mule_rule import label_accounts
from mule.labels.stage import run_labels
from mule.phase0 import run_phase0


def test_ingest_and_phase0_end_to_end(tiny_project):
    cfg = load_config(tiny_project / "base.yaml")
    summary = run_ingest(cfg)
    assert summary["transactions"] == 8 + 5 * 8 * 3

    tx = pd.read_parquet(cfg.paths.interim_dir / "transactions.parquet")
    assert tx["ts"].is_monotonic_increasing
    assert tx["tx_id"].tolist() == list(range(len(tx)))
    index = pd.read_parquet(cfg.paths.interim_dir / "account_index.parquet")
    ids = dict(zip(index["account"], index["account_id"], strict=True))

    labels = label_accounts(tx).set_index("account_id")
    mules = {a for a, i in ids.items() if labels.loc[i, "is_mule"]}
    adjacent = {a for a, i in ids.items() if labels.loc[i, "adjacent"]}
    assert mules == {"B1", "C1", "F1", "G1"}
    assert adjacent == {"A1", "D1", "E1"}

    euro = tx.loc[tx["currency_paid"] == "Euro", "amount_usd"]
    assert (euro.round(6) == 110.0).all()  # 100 EUR paid -> 110 USD

    nums = run_phase0(cfg)
    assert nums["pattern_coverage"] == 6 / 8  # self-transfers are laundering but not in a pattern
    assert nums["pattern_match_rate"] == 1.0
    assert nums["accounts_file_coverage"] > 0
    assert (cfg.paths.reports_dir / "phase0.md").exists()
    saved = json.loads((cfg.paths.reports_dir / "phase0_numbers.json").read_text())
    assert saved["mules_whole_period"] == 4


def test_labels_and_features_stages(tiny_project):
    cfg = load_config(tiny_project / "base.yaml")
    run_ingest(cfg)
    summary = run_labels(cfg)
    names = [w["window"] for w in summary["windows"]]
    assert names[-3:] == ["val", "test", "test_no_tail"]
    assert all(n.startswith("train_s") for n in names[:-3])
    assert (cfg.paths.processed_dir / "split_manifest.json").exists()
    assert (cfg.paths.reports_dir / "labels.md").exists()

    labels = pd.read_parquet(cfg.paths.processed_dir / "account_labels.parquet")
    train = labels[labels["window"].astype(str).str.startswith("train_s")]
    assert int(train["is_mule"].sum()) == 4  # all laundering happens on day one
    assert not train["prior_mule"].any()  # nobody can be a repeat mule in the first snapshot

    out = run_features(cfg)
    feats = pd.read_parquet(cfg.paths.processed_dir / "account_features.parquet")
    meta = json.loads((cfg.paths.processed_dir / "account_features.meta.json").read_text())
    assert out["features"] == len(meta["feature_columns"]) == 32  # fixture has only 4 formats
    merged = labels.merge(feats, on=["window", "account_id"], how="left", validate="one_to_one")
    assert len(merged) == len(labels)  # every labelled account has a feature row
