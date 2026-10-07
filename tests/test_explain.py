"""Reason codes on hand-built inputs, then the explain stage end to end (without PyTorch)."""

import json

import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

from mule.cli import app
from mule.config import load_config
from mule.eval.combine import ensemble_components, run_ensemble, seed_mean
from mule.explain.codes import Code, combine, fire, gnn_code, risk_shares, thresholds
from mule.explain.stage import run_explain
from mule.features.stage import run_features
from mule.labels.stage import run_labels
from mule.models.train import run_train
from tests.synthetic import make_interim

CATALOG = [
    Code("RC01", "Pass-through", "pt", "pt_fwd_72h_share", "high", "Passed on {value:.0%}"),
    Code(
        "RC02",
        "Many senders",
        "cp",
        "cp_log_unique_senders",
        "high",
        "From {value:.0f} senders (typical {typical:.0f})",
        "expm1",
    ),
    Code("RC04", "Near-zero", "pt", "pt_net_flow", "near_zero", "Balance {value:.0%}"),
]
TRAIN = pd.DataFrame(
    {
        "pt_fwd_72h_share": np.linspace(0, 1, 101),
        "cp_log_unique_senders": np.log1p(np.arange(101)),
        "pt_net_flow": np.linspace(-1, 1, 101),
    }
)


def test_codes_need_both_an_extreme_value_and_an_important_group():
    cuts = thresholds(TRAIN, CATALOG, 0.95, 0.05)
    row = pd.Series(
        {"pt_fwd_72h_share": 0.99, "cp_log_unique_senders": np.log1p(99), "pt_net_flow": 0.0}
    )
    both = fire(row, pd.Series({"pt": 2.0, "cp": 1.0, "vol": -1.0}), CATALOG, cuts)
    assert [c["code"] for c in both] == ["RC01", "RC04", "RC02"]  # strongest group first
    assert "99 senders" in both[2]["text"] and "typical 50" in both[2]["text"]
    # Same extreme values, but pass-through is not a top contributor: no pt codes.
    only_cp = fire(row, pd.Series({"pt": -0.5, "cp": 1.0}), CATALOG, cuts)
    assert [c["code"] for c in only_cp] == ["RC02"]
    # Important group, ordinary value: nothing fires.
    plain = pd.Series({"pt_fwd_72h_share": 0.3, "cp_log_unique_senders": 1.0, "pt_net_flow": 0.5})
    assert fire(plain, pd.Series({"pt": 2.0, "cp": 1.0}), CATALOG, cuts) == []


def test_gnn_code_placement_follows_its_share_of_the_risk():
    fired = [{"code": "RC01", "strength": 2.0}, {"code": "RC02", "strength": 1.0}]
    assert gnn_code(0.1, 0.2) is None
    assert [c["code"] for c in combine(fired, gnn_code(0.3, 0.2))] == ["RC01", "RC02", "RC08"]
    assert [c["code"] for c in combine(fired, gnn_code(0.7, 0.2))] == ["RC08", "RC01", "RC02"]


def test_risk_shares_split_added_risk_between_members():
    shares = risk_shares(np.array([0.9, 0.1, 0.5]), np.array([0.1, 0.9, 0.5]), 0.7, 0.1, 0.1)
    assert shares[0] == 0.0 and shares[1] == 1.0 and 0 < shares[2] < 0.5


@pytest.fixture(scope="module")
def project(tmp_path_factory):
    cfg = load_config(make_interim(tmp_path_factory.mktemp("explain")))
    run_labels(cfg)
    run_features(cfg)
    for model in ("xgb_graph", "xgb_pattern"):
        run_train(cfg, "node", model, n_seeds=2)
    run_ensemble(cfg, "node", "xgb_graph", "xgb_pattern")
    cfg.explain = cfg.explain.model_copy(
        update={
            "champion": "ens_xgb_graph__xgb_pattern",
            "catalog": __import__("pathlib").Path("configs/reason_codes.yaml"),
            "top": 15,
        }
    )
    return cfg


def test_ensemble_components_reproduce_the_saved_scores(project):
    comps = ensemble_components(project, "node", project.explain.champion)
    a = seed_mean(project, "node", comps["a"], "val")
    b = seed_mean(project, "node", comps["b"], "val").set_index("account_id")["score"]
    rebuilt = comps["weight_a"] * comps["cal_a"].predict(a["score"].to_numpy()) + (
        1 - comps["weight_a"]
    ) * comps["cal_b"].predict(b.reindex(a["account_id"]).to_numpy())
    saved = pd.read_parquet(
        project.paths.runs_dir / "node" / project.explain.champion / "seed0" / "scores_val.parquet"
    ).set_index("account_id")["score"]
    assert np.allclose(rebuilt, saved.reindex(a["account_id"]).to_numpy(), atol=1e-9)


def test_explain_stage_writes_codes_and_a_report(project):
    summary = run_explain(project, "val", with_gnn=False)
    assert summary["accounts"] == 15
    df = pd.read_parquet(project.paths.processed_dir / "explanations_val.parquet")
    assert df["rank"].tolist() == list(range(1, 16))
    for codes in df["codes"]:
        assert all({"code", "name", "text"} <= set(c) for c in json.loads(codes))
    assert (project.paths.reports_dir / "explanations_val.md").exists()


def test_why_command_explains_one_account(project, tmp_path):
    df = pd.read_parquet(project.paths.processed_dir / "explanations_val.parquet")
    cfg_path = tmp_path / "c.yaml"
    raw = project.model_dump(mode="json")
    import yaml

    cfg_path.write_text(yaml.safe_dump(raw))
    result = CliRunner().invoke(
        app, ["why", "-c", str(cfg_path), "-a", df["account"].iat[0], "-w", "val", "--no-gnn"]
    )
    assert result.exit_code == 0, result.output
    assert "rank 1 of" in result.output and "GNN share" in result.output


def test_rc08_only_when_the_gnn_supplies_more_than_its_weight(project):
    from mule.eval.combine import ensemble_components
    from mule.explain.stage import explain_accounts

    weight_b = 1 - ensemble_components(project, "node", project.explain.champion)["weight_a"]
    scores = pd.read_parquet(
        project.paths.runs_dir / "node" / project.explain.champion / "seed0" / "scores_val.parquet"
    )
    ids = scores.sort_values("score", ascending=False)["account_id"].head(30).to_numpy()
    df = explain_accounts(project, "val", ids, with_gnn=False)
    has_rc08 = df["codes"].map(lambda c: any(x["code"] == "RC08" for x in json.loads(c)))
    assert (has_rc08 == (df["gnn_share"] > weight_b)).all()
