"""Recall by pattern and the stress test on synthetic data (GNN member kept fixed: no PyTorch)."""

import pandas as pd
import pytest

from mule.config import load_config
from mule.eval.combine import run_ensemble
from mule.eval.day9 import delay_mule_sends, run_pattern_recall, run_stress
from mule.features.stage import run_features
from mule.io import read_parquet
from mule.labels.stage import run_labels
from mule.models.train import run_train
from tests.synthetic import make_interim


def test_only_mule_sends_move_and_they_stay_inside_the_window():
    t0 = pd.Timestamp("2022-09-05")
    tx = pd.DataFrame(
        {
            "tx_id": [0, 1, 2, 3],
            "ts": [
                t0,
                t0 + pd.Timedelta(hours=1),
                t0 + pd.Timedelta(hours=2),
                t0 + pd.Timedelta(hours=40),
            ],
            "src_id": [5, 7, 7, 7],
            "dst_id": [7, 9, 9, 9],
            "is_laundering": [1, 1, 0, 1],
            "is_self": False,
        }
    )
    end = t0 + pd.Timedelta(days=2)
    out, moved = delay_mule_sends(tx, [7], t0, end, delay_hours=24)
    by_id = out.set_index("tx_id")["ts"]
    assert moved == 2
    assert by_id[0] == t0  # receipt from a non-mule: unchanged
    assert by_id[1] == t0 + pd.Timedelta(hours=25)  # mule's laundering send: delayed
    assert by_id[2] == t0 + pd.Timedelta(hours=2)  # mule's normal send: unchanged
    assert by_id[3] == end - pd.Timedelta(minutes=1)  # would leave the window: clipped
    assert out["ts"].is_monotonic_increasing


@pytest.fixture(scope="module")
def project(tmp_path_factory):
    root = tmp_path_factory.mktemp("day9")
    cfg = load_config(make_interim(root))
    run_labels(cfg)
    run_features(cfg)
    for model in ("xgb_graph", "xgb_pattern"):
        run_train(cfg, "node", model, n_seeds=1)
        run_train(cfg, "edge", model, n_seeds=1)
    run_ensemble(cfg, "node", "xgb_graph", "xgb_pattern")
    run_ensemble(cfg, "edge", "xgb_graph", "xgb_pattern")
    # A patterns file in the AMLworld format for some of the synthetic laundering transfers.
    tx = read_parquet(cfg.paths.interim_dir / "transactions.parquet")
    index = read_parquet(cfg.paths.interim_dir / "account_index.parquet").set_index("account_id")
    laundering = tx[tx["is_laundering"] == 1].head(30)
    lines = []
    for i in range(0, 30, 3):
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
    cfg.explain = cfg.explain.model_copy(update={"champion": "ens_xgb_graph__xgb_pattern"})
    return cfg


def test_pattern_recall_needs_final_and_reports_both_tasks(project):
    with pytest.raises(PermissionError):
        run_pattern_recall(project, "ens_xgb_graph__xgb_pattern", "ens_xgb_graph__xgb_pattern")
    text = run_pattern_recall(
        project, "ens_xgb_graph__xgb_pattern", "ens_xgb_graph__xgb_pattern", final=True
    )
    assert "Transactions" in text and "Accounts" in text and "none" in text


def test_stress_test_rescores_the_champion(project):
    with pytest.raises(PermissionError):
        run_stress(project, 48, "test", final=False, with_gnn=False)
    text = run_stress(project, 48, "test", final=True, with_gnn=False)
    assert "champion" in text and "XGBoost member" in text
    assert (project.paths.reports_dir / "stress_test.md").exists()
    delayed = int(text.split("Laundering transfers delayed: ")[1].split()[0].replace(",", ""))
    assert 0 < delayed <= 36  # the synthetic test window holds 36 laundering transfers
