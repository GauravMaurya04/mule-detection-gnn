"""Day 8: reason codes for the account champion, plus the explanation checks.

For each explained account:
  - SHAP (exact, seed-averaged) on the XGBoost member gives group contributions -> catalog codes;
  - the ensemble is split into each member's share of the account's added risk; when the GNN
    member supplies >= gnn_share_min, RC08 joins the codes (first if the GNN leads);
  - the GNN explainer picks the evidence transfers and runs fidelity / ground-truth / stability;
  - the audit compares the two members: do they agree on why the account looks risky?
The GNN parts need PyTorch; without it (or with_gnn=False) codes are still produced.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from mule.config import Config
from mule.eval.combine import ensemble_components, seed_mean
from mule.explain.codes import combine, fire, gnn_code, load_catalog, risk_shares, thresholds
from mule.explain.shap_part import group_sums, seed_shap
from mule.io import TX_SCHEMA, read_parquet
from mule.report import Report


def _accounts_text(cfg: Config) -> pd.Series:
    index = read_parquet(cfg.paths.interim_dir / "account_index.parquet").set_index("account_id")
    return index["bank"].astype(str) + ":" + index["account"].astype(str)


def explain_accounts(cfg: Config, window: str, account_ids, with_gnn: bool = True) -> pd.DataFrame:
    ecfg = cfg.explain
    comps = ensemble_components(cfg, "node", ecfg.champion)
    feats = read_parquet(cfg.paths.processed_dir / "account_features.parquet")
    feats["window"] = feats["window"].astype(str)
    train = feats[feats["window"].str.startswith("train_s")]
    catalog = load_catalog(ecfg.catalog)
    cuts = thresholds(train, catalog, ecfg.q_high, ecfg.q_low)

    xw = feats[feats["window"] == window].set_index("account_id")
    ids = np.asarray(account_ids, dtype=np.int64)
    X = xw.reindex(ids).drop(columns=["window"])
    groups = group_sums(seed_shap(cfg, "node", comps["a"], X))

    pa = seed_mean(cfg, "node", comps["a"], window).set_index("account_id")["score"]
    pb = seed_mean(cfg, "node", comps["b"], window).set_index("account_id")["score"]
    pa = pd.Series(comps["cal_a"].predict(pa.to_numpy()), index=pa.index)
    pb = pd.Series(comps["cal_b"].predict(pb.reindex(pa.index).to_numpy()), index=pa.index)
    w = comps["weight_a"]
    shares = risk_shares(
        pa.reindex(ids).to_numpy(),
        pb.reindex(ids).to_numpy(),
        w,
        float(pa.mean()),
        float(pb.mean()),
    )

    evidence, edge_tx = {}, None
    tx = None
    if with_gnn:
        try:
            from mule.explain.gnn_part import gnn_evidence
        except ImportError:
            print("WARNING: PyTorch not available; evidence transfers skipped")
            with_gnn = False
    if with_gnn:
        tx = read_parquet(cfg.paths.interim_dir / "transactions.parquet", TX_SCHEMA)
        evidence, edge_tx = gnn_evidence(
            cfg, comps["b"], window, ids, tx.set_index("tx_id")["is_laundering"]
        )

    rc08_min = ecfg.gnn_share_min if ecfg.gnn_share_min is not None else 1 - w
    names = _accounts_text(cfg)
    by_id = tx.set_index("tx_id") if tx is not None else None
    rows = []
    for i, aid in enumerate(ids):
        fired = fire(X.loc[aid], groups.loc[aid], catalog, cuts, ecfg.top_groups, ecfg.max_codes)
        codes = combine(fired, gnn_code(float(shares[i]), rc08_min), ecfg.max_codes)
        ev = evidence.get(int(aid), {})
        transfers = []
        if ev.get("evidence_edges") and by_id is not None:
            for pos, weight in zip(ev["evidence_edges"], ev["evidence_weights"], strict=True):
                t = tx.loc[tx["tx_id"] == int(edge_tx[pos])].iloc[0]
                incoming = int(t["dst_id"]) == int(aid)
                other = int(t["src_id"] if incoming else t["dst_id"])
                transfers.append(
                    {
                        "tx_id": int(t["tx_id"]),
                        "ts": str(t["ts"]),
                        "direction": "in" if incoming else "out",
                        "counterparty": names.get(other, str(other)),
                        "amount_usd": round(float(t["amount_usd"]), 2),
                        "format": str(t["payment_format"]),
                        "weight": weight,
                    }
                )
        rows.append(
            {
                "account_id": int(aid),
                "account": names.get(int(aid), str(aid)),
                "p_xgb": float(pa.get(aid, np.nan)),
                "p_gnn": float(pb.get(aid, np.nan)),
                "gnn_share": float(shares[i]),
                "codes": json.dumps(codes),
                "n_codes": len(codes),
                "n_catalog_codes": len(fired),
                "top_group": groups.loc[aid].idxmax(),
                "evidence": json.dumps(transfers),
                **{k: v for k, v in ev.items() if k not in ("evidence_edges", "evidence_weights")},
            }
        )
    return pd.DataFrame(rows)


def run_explain(
    cfg: Config, window: str = "test", top: int | None = None, with_gnn: bool = True
) -> dict:
    ecfg = cfg.explain
    top = top or ecfg.top
    scores = (
        pd.read_parquet(
            cfg.paths.runs_dir / "node" / ecfg.champion / "seed0" / f"scores_{window}.parquet"
        )
        .sort_values("score", ascending=False)
        .reset_index(drop=True)
    )
    scores["rank"] = np.arange(1, len(scores) + 1)
    chosen = scores.head(top)
    out = explain_accounts(cfg, window, chosen["account_id"].to_numpy(), with_gnn)
    out = chosen[["account_id", "score", "rank", "is_mule", "adjacent"]].merge(out, on="account_id")
    out.insert(0, "window", window)
    path = cfg.paths.processed_dir / f"explanations_{window}.parquet"
    out.to_parquet(path, index=False)

    summary = summarise(out)
    rep = Report(f"Explanations: top {top} accounts, window {window}, champion {ecfg.champion}")
    rep.section("Coverage and codes")
    for k in ("accounts", "covered_by_catalog_code", "gnn_led_accounts", "rc08_accounts"):
        rep.item(k.replace("_", " "), summary[k])
    rep.table(pd.Series(summary["code_counts"], name="accounts").rename_axis("code").reset_index())
    if "fidelity" in summary:
        rep.section("Evidence checks (GNN member)")
        for k, v in {
            **summary["fidelity"],
            **summary["ground_truth"],
            **summary["stability"],
        }.items():
            rep.item(k.replace("_", " "), f"{v:.3f}")
    rep.section("Audit: do the two members agree?")
    for k, v in summary["agreement"].items():
        rep.item(k.replace("_", " "), v)
    (cfg.paths.reports_dir / f"explanations_{window}.md").write_text(rep.text())
    return summary


def summarise(df: pd.DataFrame) -> dict:
    codes = [c["code"] for row in df["codes"] for c in json.loads(row)]
    out = {
        "accounts": len(df),
        "covered_by_catalog_code": f"{(df['n_catalog_codes'] > 0).mean():.1%}",
        "gnn_led_accounts": int((df["gnn_share"] >= 0.5).sum()),
        "rc08_accounts": int(sum("RC08" in r for r in df["codes"])),
        "code_counts": pd.Series(codes).value_counts().to_dict() if codes else {},
    }
    if "p_full" in df and df["p_full"].notna().any():
        d = df.dropna(subset=["p_full"])
        out["fidelity"] = {
            "score_full": d["p_full"].mean(),
            "score_without_evidence": d["p_without_evidence"].mean(),
            "score_without_random_transfers": d["p_without_random"].mean(),
            "score_evidence_only": d["p_evidence_only"].mean(),
        }
        mules = d[d["is_mule"].astype(bool)]
        out["ground_truth"] = {
            "true_mules_explained": float(len(mules)),
            "evidence_that_is_laundering": mules["evidence_laundering_share"].mean(),
            "laundering_share_of_neighbourhood": mules["subgraph_laundering_share"].mean(),
        }
        out["stability"] = {"evidence_jaccard_two_seeds": d["stability_jaccard"].mean()}
    gnn_led = df["gnn_share"] >= 0.5
    out["agreement"] = {
        "xgb_led_accounts": int((~gnn_led).sum()),
        "gnn_led_accounts": int(gnn_led.sum()),
        "rank_correlation_of_member_scores": round(
            float(spearmanr(df["p_xgb"], df["p_gnn"]).statistic), 3
        )
        if len(df) > 2
        else None,
        "most_common_top_shap_group": df["top_group"].mode().iat[0] if len(df) else None,
    }
    return out
