"""The audit's weighted metrics match sklearn, and the whole audit runs on synthetic data."""

import json

import numpy as np
import pytest
from sklearn.metrics import average_precision_score

from mule.config import load_config
from mule.eval.audit import Ranked, bootstrap, confusion, run_audit, val_operating_points
from mule.eval.combine import run_ensemble
from mule.eval.evaluate import run_evaluate
from mule.features.stage import run_features
from mule.io import read_parquet
from mule.labels.stage import run_labels
from mule.models.train import run_train
from tests.synthetic import make_interim


def test_weighted_average_precision_matches_sklearn_with_ties_and_weights():
    rng = np.random.default_rng(1)
    y = rng.random(3000) < 0.02
    s = np.round(rng.random(3000) + y * 0.3, 2)  # rounding creates many ties
    rk = Ranked(y, s)
    assert rk.ap() == pytest.approx(average_precision_score(y, s), abs=1e-12)
    w = rng.integers(0, 3, 3000).astype(float)  # integer weights = replicated rows
    keep = w > 0
    rep = np.repeat(np.arange(3000)[keep], w[keep].astype(int))
    assert rk.ap(w) == pytest.approx(average_precision_score(y[rep], s[rep]), abs=1e-12)


def test_hits_at_k_and_confusion():
    y = np.array([1, 0, 1, 0, 0], bool)
    s = np.array([0.9, 0.8, 0.7, 0.2, 0.1])
    rk = Ranked(y, s)
    assert rk.hits_at(1) == 1 and rk.hits_at(3) == 2 and rk.hits_at(2.5) == 1.5
    c = confusion(y, s, 0.75)
    assert (c["tp"], c["fp"], c["fn"], c["tn"]) == (1, 1, 1, 2)
    assert c["f1"] == pytest.approx(0.5)


def test_operating_points_respect_their_constraints():
    rng = np.random.default_rng(2)
    y = rng.random(5000) < 0.05
    s = rng.random(5000) + y * 0.8
    ops = val_operating_points(y, s).set_index("operating_point")["threshold"]
    c = confusion(y, s, ops["B precision >= 0.80"])
    assert c["precision"] >= 0.80
    c = confusion(y, s, ops["F recall >= 0.60"])
    assert c["recall"] >= 0.60


def test_bootstrap_intervals_contain_the_point_estimate():
    rng = np.random.default_rng(3)
    y = rng.random(4000) < 0.03
    s = rng.random(4000) + y * 0.5
    b = bootstrap(y, {"m": s}, {"m": 0.9}, None, 200)
    ci = b[(b["model"] == "m") & (b["metric"] == "pr_auc")].iloc[0]
    assert ci["ci_low"] <= average_precision_score(y, s) <= ci["ci_high"]


@pytest.fixture(scope="module")
def project(tmp_path_factory):
    root = tmp_path_factory.mktemp("audit")
    cfg = load_config(make_interim(root))
    run_labels(cfg)
    run_features(cfg)
    for model in ("xgb_graph", "xgb_pattern"):
        run_train(cfg, "node", model, n_seeds=2)
        run_train(cfg, "edge", model, n_seeds=1)
    run_ensemble(cfg, "node", "xgb_graph", "xgb_pattern")
    run_ensemble(cfg, "edge", "xgb_graph", "xgb_pattern")
    tx = read_parquet(cfg.paths.interim_dir / "transactions.parquet")
    index = read_parquet(cfg.paths.interim_dir / "account_index.parquet").set_index("account_id")
    laundering = tx[tx["is_laundering"] == 1].head(60)
    lines = []
    for i in range(0, 60, 3):
        lines.append("BEGIN LAUNDERING ATTEMPT - STACK")
        for r in laundering.iloc[i : i + 3].itertuples():
            src, dst = index.loc[r.src_id], index.loc[r.dst_id]
            lines.append(
                f"{r.ts:%Y/%m/%d %H:%M},{src.bank},{src.account},{dst.bank},{dst.account},"
                f"{r.amount_received:.2f},{r.currency_received},{r.amount_paid:.2f},"
                f"{r.currency_paid},{r.payment_format},1"
            )
        lines.append("END LAUNDERING ATTEMPT - STACK")
    cfg.paths.raw_dir.mkdir(parents=True, exist_ok=True)
    (cfg.paths.raw_dir / "SYN_Patterns.txt").write_text("\n".join(lines) + "\n")
    return cfg


def test_audit_runs_on_validation_and_guards_test(project):
    kwargs = dict(
        node_champion="ens_xgb_graph__xgb_pattern",
        edge_champion="ens_xgb_graph__xgb_pattern",
        node_extra=("xgb_graph",),
        edge_extra=("xgb_graph",),
        boot_node=30,
        boot_edge=20,
        train_fit=True,
    )
    path = run_audit(project, final=False, **kwargs)
    text = path.read_text()
    for heading in (
        "1. Dataset",
        "4. Accounts: operating points",
        "7. Transactions: ensemble",
        "9. Accounts: by payment channel",
        "10. Transactions: familiar vs unseen",
        "14. Feature importance: node xgb_graph",
        "15. Train vs validation fit",
    ):
        assert heading in text
    assert (
        "test"
        not in json.loads((project.paths.reports_dir / "audit_val.json").read_text())["windows"]
    )
    assert (project.paths.reports_dir / "audit" / "threshold_curve_node_xgb_graph_val.csv").exists()

    run_evaluate(project, "node", ["ens_xgb_graph__xgb_pattern"], final=True)
    final = run_audit(project, final=True, **kwargs).read_text()
    assert "Window test_no_tail" in final
    data = json.loads((project.paths.reports_dir / "audit_final.json").read_text())
    # the stored ensemble metrics come from the same single score file, so they must agree
    rec = [
        r
        for r in data["node_reconciliation"]
        if r["model"].startswith("ens_") and r["metric"] == "pr_auc"
    ]
    assert rec and all(
        abs(r["audit (seed-averaged)"] - r["stored (mean of seeds)"]) < 1e-9 for r in rec
    )
