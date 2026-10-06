"""Ranking metrics for rare positives. Top-K counts are tie-aware, so rule baselines with many
equal scores are judged by their expected value rather than by an arbitrary tie order."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve


def expected_in_top_k(flag: np.ndarray, scores: np.ndarray, k: int) -> float:
    """Expected number of flagged rows among the k highest scores (ties shared evenly)."""
    flag = np.asarray(flag, dtype=bool)
    scores = np.asarray(scores, dtype=float)
    k = int(min(max(k, 0), len(scores)))
    if k == 0:
        return 0.0
    kth = np.partition(scores, len(scores) - k)[len(scores) - k]
    above = scores > kth
    tied = scores == kth
    need = k - int(above.sum())
    return float(flag[above].sum() + need * flag[tied].sum() / tied.sum())


def precision_recall_at_k(y, scores, k: int) -> tuple[float, float]:
    y = np.asarray(y, dtype=bool)
    hits = expected_in_top_k(y, scores, k)
    return (hits / k if k else np.nan), (hits / y.sum() if y.sum() else np.nan)


def pr_auc(y, scores) -> float:
    y = np.asarray(y)
    return float(average_precision_score(y, scores)) if y.sum() else np.nan


def best_f1_threshold(y, scores) -> float:
    """Threshold maximising F1 (chosen on validation, then frozen for test)."""
    y = np.asarray(y)
    if not y.sum():
        return float("inf")
    precision, recall, thresholds = precision_recall_curve(y, scores)
    f1 = 2 * precision[:-1] * recall[:-1] / np.clip(precision[:-1] + recall[:-1], 1e-12, None)
    return float(thresholds[int(np.argmax(f1))])


def threshold_metrics(y, scores, threshold: float) -> dict[str, float]:
    y = np.asarray(y, dtype=bool)
    pred = np.asarray(scores) >= threshold
    tp, fp = int((pred & y).sum()), int((pred & ~y).sum())
    fn, tn = int((~pred & y).sum()), int((~pred & ~y).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else np.nan
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "fpr": fp / (fp + tn) if fp + tn else np.nan,
    }


def bootstrap_pr_auc_ci(y, scores, n: int, seed: int = 0) -> tuple[float, float]:
    """95% interval for PR-AUC from resampling rows with replacement."""
    y, scores = np.asarray(y), np.asarray(scores)
    if not y.sum() or n == 0:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(n):
        idx = rng.integers(0, len(y), len(y))
        if y[idx].sum():
            values.append(average_precision_score(y[idx], scores[idx]))
    return float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))


def calibration(y, scores, bins: int = 10) -> dict[str, float]:
    """Brier score and expected calibration error (equal-width bins), for probability scores."""
    y = np.asarray(y, dtype=float)
    p = np.clip(np.asarray(scores, dtype=float), 0.0, 1.0)
    which = np.minimum((p * bins).astype(int), bins - 1)
    ece = 0.0
    for b in range(bins):
        m = which == b
        if m.any():
            ece += m.mean() * abs(y[m].mean() - p[m].mean())
    return {"brier": float(np.mean((p - y) ** 2)), "ece": float(ece)}
