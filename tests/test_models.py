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
    assert edge["train"].X.shape[1] == 16 + 5 + 2 * 55  # own, patterns, sender, receiver
    assert len(edge["train"].y) == len(edge["train"].meta)


@pytest.mark.parametrize("task", ["node", "edge"])
def test_train_evaluate_and_final_guard(project, task):
    for model in ("rule_ach", "rules", "xgb", "xgb_graph", "xgb_full", "xgb_pattern"):
        summary = run_train(project, task, model, n_seeds=2)
        assert len(summary["seeds"]) == (2 if model.startswith("xgb") else 1)
    sizes = {
        m: json.loads((project.paths.runs_dir / task / m / "train_summary.json").read_text())[
            "features"
        ]
        for m in ("xgb", "xgb_graph", "xgb_full", "xgb_pattern")
    }
    own = 0 if task == "node" else 16
    edge_pa = 0 if task == "node" else 5  # transaction pattern columns only in "pa" models
    per_account = 1 if task == "node" else 2
    assert sizes == {
        "xgb": own + per_account * 32,
        "xgb_graph": own + per_account * 40,
        "xgb_full": own + per_account * 48,
        "xgb_pattern": own + edge_pa + per_account * 47,  # 32 + graph 8 + patterns 7
    }

    from mule.eval.compare import compare

    result = compare(project, task, "xgb_pattern", "xgb_graph", n=30)
    assert result["winner"] in ("xgb_pattern", "xgb_graph") and result["window"] == "val"
    with pytest.raises(PermissionError):
        compare(project, task, "xgb_pattern", "xgb_graph", window="test", n=10)

    run_evaluate(project, task, None, final=False)
    metrics = json.loads((project.paths.runs_dir / task / "xgb" / "metrics.json").read_text())
    assert list(metrics["windows"]) == ["val"]  # no test numbers without --final

    report = run_evaluate(project, task, None, final=True)
    final_path = project.paths.runs_dir / task / "xgb" / "metrics_final.json"
    metrics = json.loads(final_path.read_text())
    assert metrics["final"] and set(metrics["windows"]) == {"val", "test", "test_no_tail"}
    assert "Window: test_no_tail" in report.read_text()

    # A later normal run must not erase the recorded final results...
    report = run_evaluate(project, task, None, final=False)
    assert final_path.exists() and "Window: test_no_tail" in report.read_text()
    # ...and a frozen model refuses to retrain without --force.
    with pytest.raises(FileExistsError):
        run_train(project, task, "xgb", n_seeds=1)
    if task == "edge":
        assert "F1 @ 0.5 (IBM protocol)" in report.read_text()
    if task == "node":
        assert "pr_auc_ci_low" in metrics["windows"]["test"]
        assert metrics["windows"]["val"]["pr_auc"]["mean"] > 0.05


def test_scorer_explains_one_account_and_lists_the_top(project):
    run_train(project, "node", "xgb", n_seeds=1, force=True)
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
