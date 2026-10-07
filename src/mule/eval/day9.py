"""Day 9 analyses of the final champions: recall by laundering pattern, and one stress test.

Both read test labels, so both sit behind the same --final guard as the evaluator. Neither
changes a model: they score frozen models on re-sliced or perturbed data.

Recall by pattern: each laundering transaction gets the pattern type of its attempt in the
patterns file, or "none" (the 38% of laundering that belongs to no pattern). Transactions:
share caught at the champion's frozen validation threshold. Accounts: each mule is tagged with
the pattern types of the laundering transfers it took part in during the window, and its
recall is the expected share inside the top R (R = number of mules), as in the main tables.

Stress test: mules' outgoing laundering transfers in the test window are delayed by
`delay_hours` (kept inside the window), so pass-through gets slower while everything else stays
put. Features are recomputed from the perturbed transactions, the frozen champion re-scores
them, and PR-AUC against the original labels shows how much the model relied on speed.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from mule.config import Config
from mule.data.patterns import match_patterns, read_patterns
from mule.eval import metrics as M
from mule.eval.combine import ensemble_components, seed_mean
from mule.eval.evaluate import check_window_allowed
from mule.features.account import account_features, entity_table
from mule.io import ACCOUNT_INDEX_SCHEMA, TX_SCHEMA, read_parquet
from mule.labels.split import load_manifest, window_bounds
from mule.models.registry import make_model
from mule.report import Report


def pattern_of_transaction(cfg: Config, tx: pd.DataFrame) -> pd.Series:
    """tx_id -> pattern type for laundering transactions ("none" if in no attempt)."""
    index = read_parquet(cfg.paths.interim_dir / "account_index.parquet", ACCOUNT_INDEX_SCHEMA)
    patterns = match_patterns(
        read_patterns(cfg.paths.raw_dir / f"{cfg.dataset}_Patterns.txt"), tx, index
    )
    found = patterns.dropna(subset=["tx_id"]).drop_duplicates("tx_id")
    by_tx = pd.Series(found["pattern"].to_numpy(), index=found["tx_id"].astype(int).to_numpy())
    laundering = tx.loc[tx["is_laundering"] == 1, "tx_id"].to_numpy()
    return by_tx.reindex(laundering).fillna("none")


def _transaction_recall(cfg, model, window, pattern_by_tx) -> pd.DataFrame:
    info = json.loads((cfg.paths.runs_dir / "edge" / model / "seed0" / "run.json").read_text())
    df = seed_mean(cfg, "edge", model, window)
    df = df[df["is_laundering"] == 1].assign(caught=lambda d: d["score"] >= info["threshold"])
    df["pattern"] = pattern_by_tx.reindex(df["tx_id"].to_numpy()).to_numpy()
    out = df.groupby("pattern").agg(transactions=("caught", "size"), recall=("caught", "mean"))
    return out.reset_index()


def _account_recall(cfg, model, window, tx, pattern_by_tx) -> pd.DataFrame:
    manifest = load_manifest(cfg.paths.processed_dir / "split_manifest.json")
    start, end = window_bounds(manifest, window)
    df = seed_mean(cfg, "node", model, window)
    y, s = df["is_mule"].to_numpy(bool), df["score"].to_numpy(float)
    in_window = tx[(tx["ts"] >= start) & (tx["ts"] < end) & (tx["is_laundering"] == 1)]
    tagged = pd.concat(
        [
            pd.DataFrame(
                {
                    "account_id": in_window["src_id"].to_numpy(),
                    "pattern": pattern_by_tx.reindex(in_window["tx_id"]).to_numpy(),
                }
            ),
            pd.DataFrame(
                {
                    "account_id": in_window["dst_id"].to_numpy(),
                    "pattern": pattern_by_tx.reindex(in_window["tx_id"]).to_numpy(),
                }
            ),
        ]
    ).drop_duplicates()
    mules = set(df.loc[y, "account_id"].astype(int))
    rows, k = [], int(y.sum())
    for pattern, group in tagged[tagged["account_id"].isin(mules)].groupby("pattern"):
        flag = df["account_id"].isin(group["account_id"]).to_numpy() & y
        rows.append(
            {
                "pattern": pattern,
                "mules": int(flag.sum()),
                "recall@R": M.expected_in_top_k(flag, s, k) / max(flag.sum(), 1),
            }
        )
    return pd.DataFrame(rows)


def run_pattern_recall(
    cfg: Config,
    node_model: str,
    edge_model: str,
    windows=("test", "test_no_tail"),
    final: bool = False,
) -> str:
    for w in windows:
        check_window_allowed(w, final)
    tx = read_parquet(cfg.paths.interim_dir / "transactions.parquet", TX_SCHEMA)
    pattern_by_tx = pattern_of_transaction(cfg, tx)
    rep = Report("Recall by laundering pattern")
    rep.note(
        '"none" = laundering that belongs to no pattern attempt (about 38% of all laundering).'
    )
    for w in windows:
        rep.section(f"Transactions, {edge_model}, window {w} (validation threshold)")
        rep.table(_transaction_recall(cfg, edge_model, w, pattern_by_tx))
        rep.section(f"Accounts, {node_model}, window {w} (recall inside the top R)")
        rep.table(_account_recall(cfg, node_model, w, tx, pattern_by_tx))
    path = cfg.paths.reports_dir / "recall_by_pattern.md"
    path.write_text(rep.text())
    return rep.text()


def delay_mule_sends(
    tx: pd.DataFrame, mule_ids, start, end, delay_hours: float
) -> tuple[pd.DataFrame, int]:
    """Delay mules' outgoing laundering transfers inside [start, end), kept inside the window.
    Returns the re-sorted transactions and how many transfers moved."""
    out = tx.copy()
    m = (
        (out["ts"] >= start)
        & (out["ts"] < end)
        & (out["is_laundering"] == 1)
        & out["src_id"].isin(mule_ids)
        & ~out["is_self"]
    ).to_numpy()
    latest = pd.Timestamp(end) - pd.Timedelta(minutes=1)
    shifted = (out.loc[m, "ts"] + pd.Timedelta(hours=delay_hours)).clip(upper=latest)
    out.loc[m, "ts"] = shifted
    return out.sort_values(["ts", "tx_id"], kind="stable").reset_index(drop=True), int(m.sum())


def run_stress(
    cfg: Config,
    delay_hours: float = 48,
    window: str = "test",
    final: bool = False,
    with_gnn: bool = True,
) -> str:
    check_window_allowed(window, final)
    manifest = load_manifest(cfg.paths.processed_dir / "split_manifest.json")
    start, end = window_bounds(manifest, window)
    comps = ensemble_components(cfg, "node", cfg.explain.champion)
    tx = read_parquet(cfg.paths.interim_dir / "transactions.parquet", TX_SCHEMA)
    labels = read_parquet(cfg.paths.processed_dir / "account_labels.parquet")
    lab = labels[labels["window"].astype(str) == window].set_index("account_id")
    mules = lab.index[lab["is_mule"]].to_numpy()
    stressed, moved = delay_mule_sends(tx, mules, start, end, delay_hours)

    entity = None
    meta = cfg.paths.interim_dir / "accounts_meta.parquet"
    if meta.exists():
        index = read_parquet(cfg.paths.interim_dir / "account_index.parquet", ACCOUNT_INDEX_SCHEMA)
        entity = entity_table(pd.read_parquet(meta), index)
    f = cfg.features
    feats = account_features(
        stressed, start, end, f.lookback_hours, f.forward_hours, entity, f.pattern_pair_cap
    )

    # XGBoost member, re-scored on the stressed features (seed average, as in the ensemble).
    parts = []
    for seed_dir in sorted((cfg.paths.runs_dir / "node" / comps["a"]).glob("seed*")):
        info = json.loads((seed_dir / "run.json").read_text())
        model = make_model(comps["a"], "node", cfg)
        model.load(seed_dir, info)
        parts.append(model.predict_scores(feats[info["feature_columns"]].astype("float32")))
    p_a_new = pd.Series(np.mean(parts, axis=0), index=feats.index)

    base_a = seed_mean(cfg, "node", comps["a"], window).set_index("account_id")["score"]
    base_b = seed_mean(cfg, "node", comps["b"], window).set_index("account_id")["score"]
    ids = base_a.index
    y = lab["is_mule"].reindex(ids).to_numpy(bool)
    w = comps["weight_a"]

    def ens(pa, pb):
        return w * comps["cal_a"].predict(pa) + (1 - w) * comps["cal_b"].predict(pb)

    rows = [
        {
            "model": f"{comps['a']} (XGBoost member)",
            "original": M.pr_auc(y, base_a.to_numpy()),
            "stressed": M.pr_auc(y, p_a_new.reindex(ids).to_numpy()),
        }
    ]
    p_b_new, note = None, "GNN member re-scored on the stressed graph."
    if with_gnn:
        try:
            from mule.explain.gnn_part import load_gnn
            from mule.models.gnn.graph import build_graphs
            from mule.models.gnn.train import _device, score_graph

            net, stats = load_gnn(cfg, comps["b"])
            feats_all = pd.concat([feats.reset_index().assign(window=window)])
            graphs, _ = build_graphs(
                cfg,
                stats["node_features"],
                stats=stats,
                windows=[window],
                tx=stressed,
                feats=feats_all,
            )
            g = graphs[window]
            p = score_graph(net, g, _device(cfg.gnn.score_device), cfg.gnn.edge_chunk)
            p_b_new = pd.Series(p, index=g.meta["account_id"].to_numpy())
            rows.append(
                {
                    "model": f"{comps['b']} (GNN member)",
                    "original": M.pr_auc(y, base_b.reindex(ids).to_numpy()),
                    "stressed": M.pr_auc(y, p_b_new.reindex(ids).to_numpy()),
                }
            )
        except ImportError:
            note = "PyTorch not available: the GNN member keeps its original scores."
    else:
        note = "GNN member keeps its original scores (--no-gnn)."
    pb_stressed = p_b_new if p_b_new is not None else base_b
    rows.append(
        {
            "model": f"{cfg.explain.champion} (champion)",
            "original": M.pr_auc(y, ens(base_a.to_numpy(), base_b.reindex(ids).to_numpy())),
            "stressed": M.pr_auc(
                y, ens(p_a_new.reindex(ids).to_numpy(), pb_stressed.reindex(ids).to_numpy())
            ),
        }
    )
    table = pd.DataFrame(rows)
    table["change"] = table["stressed"] - table["original"]
    rep = Report(f"Stress test: mules' outgoing laundering delayed by {delay_hours:g} h ({window})")
    rep.item("Mules in window", len(mules))
    rep.item("Laundering transfers delayed", moved)
    rep.note(note)
    rep.table(table)
    path = cfg.paths.reports_dir / "stress_test.md"
    path.write_text(rep.text())
    return rep.text()
