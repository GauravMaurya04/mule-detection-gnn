"""Day 7 improvement stage on synthetic data: tuning, calibration, ensemble, then evaluation."""

import json

import numpy as np
import pandas as pd
import pytest

from mule.config import load_config
from mule.eval import metrics as M
from mule.eval.combine import Isotonic, Platt, choose_calibrator, run_calibrate, run_ensemble
from mule.eval.compare import compare
from mule.eval.evaluate import run_evaluate
from mule.features.stage import run_features
from mule.labels.stage import run_labels
from mule.models.train import run_train
from mule.models.tune import run_xgb_tune
from tests.synthetic import make_interim


@pytest.fixture(scope="module")
def project(tmp_path_factory):
    cfg = load_config(make_interim(tmp_path_factory.mktemp("improve")))
    run_labels(cfg)
    run_features(cfg)
    for model in ("xgb_graph", "xgb_pattern"):
        run_train(cfg, "node", model, n_seeds=2)
    return cfg


def test_calibration_metrics_reward_honest_probabilities():
    rng = np.random.default_rng(0)
    p = rng.random(20000)
    y = rng.random(20000) < p  # perfectly calibrated by construction
    honest, inflated = M.calibration(y, p), M.calibration(y, np.sqrt(p))
    assert honest["ece"] < 0.02 and inflated["ece"] > honest["ece"]
    assert inflated["brier"] > honest["brier"]


def test_platt_keeps_the_ranking_and_cv_picks_a_calibrator():
    rng = np.random.default_rng(1)
    y = (rng.random(4000) < 0.05).astype(int)
    s = np.clip(0.3 * y + rng.random(4000) * 0.7, 0, 1)
    platt = Platt().fit(s, y)
    assert np.array_equal(np.argsort(platt.predict(s), kind="stable"), np.argsort(s, kind="stable"))
    assert choose_calibrator(s, y)["choice"] in ("platt", "isotonic")
    assert Isotonic().fit(s, y).predict(s).max() <= 1


def test_tuning_includes_defaults_and_retrains_the_winner(project):
    cfg = project.model_copy(deep=True)
    cfg.models.xgb = cfg.models.xgb.model_copy(update={"n_estimators": 20})
    result = run_xgb_tune(cfg, "node", "xgb_graph", trials=2, n_seeds=2)
    labels = [t["label"] for t in result["trials"]]
    assert labels[0] == "defaults" and len(labels) == 3
    assert result["best"]["val_pr_auc"] == max(t["val_pr_auc"] for t in result["trials"])
    spec = json.loads((cfg.paths.runs_dir / "node" / "_tuning" / "xgb_tuned.json").read_text())
    assert spec["base"] == "xgb_graph" and "gr" in spec["groups"] and "pa" not in spec["groups"]
    assert len(result["final"]["seeds"]) == 2


def test_calibrated_copy_and_ensemble_are_ordinary_models(project):
    cal = run_calibrate(project, "node", "xgb_graph")
    seed_dir = project.paths.runs_dir / "node" / "xgb_graph_cal" / "seed0"
    scores = pd.read_parquet(seed_dir / "scores_test.parquet")
    assert scores["score"].between(0, 1).all() and cal["choice"] in ("platt", "isotonic")

    ens = run_ensemble(project, "node", "xgb_graph", "xgb_pattern")
    assert 0 <= ens["weight_a"] <= 1
    assert ens["val_pr_auc"] >= max(ens["val_pr_auc_a_alone"], ens["val_pr_auc_b_alone"]) - 1e-9
    name = ens["model"]
    assert (
        json.loads((project.paths.runs_dir / "node" / name / "seed0" / "run.json").read_text())[
            "n_features"
        ]
        > 0
    )

    report = run_evaluate(project, "node", ["xgb_graph_cal", name], final=False).read_text()
    assert "xgb_graph_cal" in report and "Brier" in report
    result = compare(project, "node", name, "xgb_graph", n=30)
    assert result["winner"] in (name, "xgb_graph")


def test_stacked_model_uses_the_gnn_column_only_where_asked(project):
    from mule.models.data import load_node
    from mule.models.tune import run_stack

    labels = pd.read_parquet(project.paths.processed_dir / "account_labels.parquet")
    rng = np.random.default_rng(0)
    fake = labels[["window", "account_id"]].assign(
        window=labels["window"].astype(str),
        gnn_score=np.where(labels["is_mule"], 0.7, 0.1) + rng.random(len(labels)) * 0.2,
    )
    fake.to_parquet(project.paths.processed_dir / "gnn_oof_node.parquet", index=False)
    (project.paths.processed_dir / "gnn_oof_node.meta.json").write_text(
        json.dumps(
            {"features_hash": project.section_hash("dataset", "split", "labels", "features")}
        )
    )
    data = load_node(project)
    assert "st_gnn_logit" in data["train"].X.columns and data["val"].X["st_gnn_logit"].notna().all()

    result = run_stack(project, "xgb_graph", n_seeds=1)
    assert result["oof"] is None  # existing scores reused, no GNN training
    info = json.loads(
        (project.paths.runs_dir / "node" / "xgb_stacked" / "seed0" / "run.json").read_text()
    )
    assert "st_gnn_logit" in info["feature_columns"]
    plain = json.loads(
        (project.paths.runs_dir / "node" / "xgb_graph" / "seed0" / "run.json").read_text()
    )
    assert "st_gnn_logit" not in plain["feature_columns"]
