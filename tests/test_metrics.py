import numpy as np
import pytest

from mule.eval import metrics as M
from mule.eval.evaluate import check_window_allowed


def test_precision_at_k_shares_ties_evenly():
    # Four tied top scores, two of them positive: top-2 expects one positive.
    y = np.array([1, 1, 0, 0, 0])
    s = np.array([1.0, 1.0, 1.0, 1.0, 0.0])
    assert M.expected_in_top_k(y, s, 2) == pytest.approx(1.0)
    p, r = M.precision_recall_at_k(y, s, 2)
    assert p == pytest.approx(0.5) and r == pytest.approx(0.5)


def test_r_precision_is_one_for_a_perfect_ranking():
    y = np.array([0, 1, 0, 1, 0])
    s = np.array([0.1, 0.9, 0.2, 0.8, 0.3])
    p, r = M.precision_recall_at_k(y, s, int(y.sum()))
    assert p == r == 1.0


def test_best_f1_threshold_then_threshold_metrics():
    y = np.array([0, 0, 1, 1, 0, 1])
    s = np.array([0.1, 0.2, 0.7, 0.8, 0.3, 0.9])
    t = M.best_f1_threshold(y, s)
    m = M.threshold_metrics(y, s, t)
    assert m["f1"] == 1.0 and m["fpr"] == 0.0


def test_bootstrap_interval_brackets_the_point_estimate():
    rng = np.random.default_rng(0)
    y = (rng.random(2000) < 0.05).astype(int)
    s = y * 0.5 + rng.random(2000)
    lo, hi = M.bootstrap_pr_auc_ci(y, s, n=200)
    assert lo <= M.pr_auc(y, s) <= hi


@pytest.mark.parametrize("window", ["test", "test_no_tail"])
def test_test_windows_need_final(window):
    with pytest.raises(PermissionError):
        check_window_allowed(window, final=False)
    check_window_allowed(window, final=True)
    check_window_allowed("val", final=False)


def test_multignn_log_reports_test_f1_at_the_best_validation_epoch():
    from mule.eval.multignn import parse_log

    log = "\n".join(
        f"INFO Validation F1: {v}\nINFO Test F1: {t}"
        for v, t in [(0.40, 0.30), (0.55, 0.50), (0.55, 0.70), (0.52, 0.65)]
    )
    parsed = parse_log(log)
    assert parsed["best_epoch"] == 1 and parsed["test_f1"] == 0.50  # first epoch reaching 0.55
    assert parsed["epochs"] == 4


def test_paired_bootstrap_detects_a_real_difference_and_not_a_fake_one():
    from mule.eval.compare import paired_bootstrap

    rng = np.random.default_rng(1)
    y = rng.random(3000) < 0.05
    good = y * 1.0 + rng.normal(0, 0.6, 3000)
    weak = y * 0.4 + rng.normal(0, 0.6, 3000)
    diffs = paired_bootstrap(y, good, weak, n=200)
    assert np.percentile(diffs, 2.5) > 0
    same = paired_bootstrap(y, good, good + rng.normal(0, 1e-9, 3000), n=200)
    assert np.percentile(same, 2.5) <= 0 <= np.percentile(same, 97.5)


def test_paper_rows_appear_in_the_transaction_results(tmp_path):
    from mule.config import load_config
    from mule.eval.evaluate import write_results
    from mule.eval.multignn import add_paper_rows

    (tmp_path / "c.yaml").write_text(
        f"paths:\n  raw_dir: {tmp_path}\n  interim_dir: {tmp_path}\n"
        f"  processed_dir: {tmp_path}\n  reports_dir: {tmp_path}/reports\n"
        f"  runs_dir: {tmp_path}/runs\n"
    )
    cfg = load_config(tmp_path / "c.yaml")
    assert add_paper_rows(cfg)["paper_multipna_eu"] == pytest.approx(0.6816)
    text = write_results(cfg, "edge").read_text()
    assert "paper_multigin_eu" in text and "0.682" in text


def test_results_table_shows_tiny_values_instead_of_zero(tmp_path):
    import json

    from mule.config import load_config
    from mule.eval.evaluate import write_results

    (tmp_path / "c.yaml").write_text(
        f"paths:\n  raw_dir: {tmp_path}\n  interim_dir: {tmp_path}\n"
        f"  processed_dir: {tmp_path}\n  reports_dir: {tmp_path}/reports\n"
        f"  runs_dir: {tmp_path}/runs\n"
    )
    cfg = load_config(tmp_path / "c.yaml")
    model = cfg.paths.runs_dir / "node" / "m"
    model.mkdir(parents=True)
    cell = {"mean": 0.0005, "std": 0.0}
    metrics = {
        "model": "m",
        "seeds": 1,
        "windows": {
            "val": {
                "pr_auc": {"mean": 0.7, "std": 0.0},
                "brier": cell,
                "ece": cell,
                "fpr@val_threshold": {"mean": 0.0002, "std": 0.0},
                "positives": {"mean": 242, "std": 0},
            }
        },
    }
    (model / "metrics.json").write_text(json.dumps(metrics))
    text = write_results(cfg, "node").read_text()
    assert "5.0e-04" in text and "2.0e-04" in text and "| 0.000 |" not in text
