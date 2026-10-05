"""End to end on synthetic interim data: labels -> features -> train -> evaluate -> score."""

import json

import pandas as pd
import pytest

from mule.config import load_config
from mule.eval.evaluate import run_evaluate
from mule.features.stage import run_features
from mule.labels.stage import run_labels
from mule.models.data import load_edge, load_node
from mule.models.train import run_train
from mule.score import score_account, top_accounts
from tests.synthetic import make_interim


@pytest.fixture(scope="module")
def project(tmp_path_factory):
    cfg = load_config(make_interim(tmp_path_factory.mktemp("syn")))
    run_labels(cfg)
    run_features(cfg)
    return cfg


def test_design_matrices_never_contain_labels_or_absolute_time(project):
    node, edge = load_node(project), load_edge(project, names=("train", "val"))
    for data in (node, edge):
        cols = list(data["train"].X.columns)
        assert not [c for c in cols if "laundering" in c or "mule" in c or c in ("ts", "e_ts")]
    assert node["train"].y.sum() > 0 and node["val"].y.sum() > 0
    assert edge["train"].X.shape[1] == 16 + 2 * 32
    assert len(edge["train"].y) == len(edge["train"].meta)


@pytest.mark.parametrize("task", ["node", "edge"])
def test_train_evaluate_and_final_guard(project, task):
    for model in ("rule_ach", "rules", "xgb"):
        summary = run_train(project, task, model, n_seeds=2)
        assert len(summary["seeds"]) == (2 if model == "xgb" else 1)

    run_evaluate(project, task, None, final=False)
    metrics = json.loads((project.paths.runs_dir / task / "xgb" / "metrics.json").read_text())
    assert list(metrics["windows"]) == ["val"]  # no test numbers without --final

    report = run_evaluate(project, task, None, final=True)
    final_path = project.paths.runs_dir / task / "xgb" / "metrics_final.json"
    metrics = json.loads(final_path.read_text())
    assert metrics["final"] and set(metrics["windows"]) == {"val", "test", "test_no_tail"}
    assert "Window: test_no_tail" in report.read_text()

    # A later normal run must not erase the recorded final results.
    report = run_evaluate(project, task, None, final=False)
    assert final_path.exists() and "Window: test_no_tail" in report.read_text()
    if task == "node":
        assert "pr_auc_ci_low" in metrics["windows"]["test"]
        assert metrics["windows"]["val"]["pr_auc"]["mean"] > 0.05


def test_scorer_explains_one_account_and_lists_the_top(project):
    run_train(project, "node", "xgb", n_seeds=1)
    scores = pd.read_parquet(
        project.paths.runs_dir / "node" / "xgb" / "seed0" / "scores_val.parquet"
    )
    best = int(scores.sort_values("score").iloc[-1]["account_id"])
    index = pd.read_parquet(project.paths.interim_dir / "account_index.parquet").set_index(
        "account_id"
    )
    account = f"{index.loc[best, 'bank']}:{index.loc[best, 'account']}"
    text = score_account(project, account, window="val")
    assert "rank 1 of" in text and "By feature group" in text
    assert "Top 5 accounts" in top_accounts(project, 5, window="val")
