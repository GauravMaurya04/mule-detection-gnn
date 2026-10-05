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
