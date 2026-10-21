"""`mule audit`: the measurements behind the system audit, from saved artifacts only.

Nothing here trains or changes a model. Every threshold and operating point is chosen on the
validation window and applied unchanged to the test windows, which are read only with --final
(the same guard as the evaluator). Outputs: reports/audit_<val|final>.md, a JSON with every
number, and CSVs (threshold curves, reliability tables) under reports/audit/.

Conventions, stated once:
- A model's scores are averaged over its seeds first (as the ensembles do), so "xgb_graph" here is
  the 3-seed average, and its threshold is the max-F1 threshold of that average on validation.
  The results tables instead average per-seed metrics; the reconciliation section shows both.
- PR-AUC is average precision (step-wise), identical to sklearn's average_precision_score,
  re-implemented with sample weights so bootstraps run in O(n) per resample.
- Cluster bootstrap: laundering transfers of one pattern attempt (and mules tagged with that
  attempt) are resampled together; everything else is resampled row by row.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve, roc_auc_score

from mule.config import Config
from mule.eval import metrics as M
from mule.eval.combine import ID, Platt, seed_mean
from mule.eval.evaluate import check_window_allowed
from mule.io import TX_SCHEMA, read_parquet
from mule.labels.split import load_manifest, window_bounds
from mule.report import Report

K_LIST = (10, 25, 50, 100, 500, 1000)
OPERATING_POINTS = (
    ("A max F1", None, None),
    ("B precision >= 0.80", "precision", 0.80),
    ("C precision >= 0.85", "precision", 0.85),
    ("D precision >= 0.90", "precision", 0.90),
    ("E recall >= 0.50", "recall", 0.50),
    ("F recall >= 0.60", "recall", 0.60),
    ("G recall >= 0.70", "recall", 0.70),
)
PROFILE_FEATURES = [
    "vol_active_days",
    "vol_in_per_day",
    "vol_out_per_day",
    "cp_log_unique_senders",
    "cp_log_unique_receivers",
    "pt_fwd_24h_share",
    "pt_fwd_72h_share",
    "pt_median_delay_h",
    "pt_net_flow",
    "ch_share_ach",
    "pa_log_fan_in_24h_max",
    "gr_coreness",
]
FORMATS = ("ACH", "Bitcoin", "Cash", "Cheque", "Credit Card", "Reinvestment", "Wire")


# ----------------------------------------------------------------------------------------------
# Weighted metrics (fast enough for bootstraps)
# ----------------------------------------------------------------------------------------------


class Ranked:
    """Rows sorted once by score (descending) so weighted metrics are cumulative sums."""

    def __init__(self, y: np.ndarray, s: np.ndarray):
        self.order = np.argsort(-np.asarray(s, float), kind="stable")
        self.s = np.asarray(s, float)[self.order]
        self.y = np.asarray(y, bool)[self.order]
        self.last_of_tie = np.r_[self.s[1:] != self.s[:-1], True]

    def ap(self, w: np.ndarray | None = None) -> float:
        """Average precision with sample weights (w in the original row order)."""
        wt = np.ones(len(self.s)) if w is None else np.asarray(w, float)[self.order]
        tp = np.cumsum(wt * self.y)[self.last_of_tie]
        allw = np.cumsum(wt)[self.last_of_tie]
        total = tp[-1] if len(tp) else 0.0
        if total <= 0:
            return np.nan
        precision = np.divide(tp, allw, out=np.zeros_like(tp), where=allw > 0)
        recall = tp / total
        return float(np.sum(np.diff(np.r_[0.0, recall]) * precision))

    def hits_at(self, k: float, w: np.ndarray | None = None) -> float:
        """Weighted positives among the first k units of weight (ties broken by sort order)."""
        wt = np.ones(len(self.s)) if w is None else np.asarray(w, float)[self.order]
        cw = np.cumsum(wt)
        i = int(np.searchsorted(cw, k, side="left"))
        if i >= len(cw):
            return float(np.sum(wt * self.y))
        full = float(np.sum((wt * self.y)[:i]))
        before = cw[i - 1] if i else 0.0
        part = (k - before) / wt[i] if wt[i] else 0.0
        return full + part * wt[i] * float(self.y[i])


def confusion(y, s, thr, w=None) -> dict:
    y = np.asarray(y, bool)
    w = np.ones(len(y)) if w is None else np.asarray(w, float)
    pred = np.asarray(s) >= thr
    tp, fp = float(w[pred & y].sum()), float(w[pred & ~y].sum())
    fn, tn = float(w[~pred & y].sum()), float(w[~pred & ~y].sum())
    precision = tp / (tp + fp) if tp + fp else np.nan
    recall = tp / (tp + fn) if tp + fn else np.nan
    f1 = 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else np.nan
    n = tp + fp + fn + tn
    return {
        "threshold": float(thr),
        "flagged": tp + fp,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "accuracy": (tp + tn) / n if n else np.nan,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "fpr": fp / (fp + tn) if fp + tn else np.nan,
        "fnr": fn / (fn + tp) if fn + tp else np.nan,
    }


def calibration_detail(y, p, top_bins: int = 10) -> tuple[dict, pd.DataFrame]:
    """Slope/intercept of logit-recalibration, log loss, and equal-mass reliability over the
    highest scores (where rare-event calibration matters), plus one bin for everything else."""
    y = np.asarray(y, float)
    p = np.clip(np.asarray(p, float), 1e-7, 1 - 1e-7)
    z = np.log(p / (1 - p))
    out = {
        "log_loss": float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))),
        "brier_no_skill": float(y.mean() * (1 - y.mean())),
    }
    if 0 < y.sum() < len(y):
        lr = LogisticRegression(C=1e6, max_iter=1000).fit(z[:, None], y)
        out["cal_slope"], out["cal_intercept"] = float(lr.coef_[0][0]), float(lr.intercept_[0])
    n_top = int(min(len(p), max(10 * y.sum(), 2000)))
    order = np.argsort(-p, kind="stable")
    rows = []
    for b, idx in enumerate(np.array_split(order[:n_top], top_bins)):
        rows.append(_rel_row(f"top {b + 1}/{top_bins}", y[idx], p[idx]))
    rest = order[n_top:]
    if len(rest):
        rows.append(_rel_row("rest", y[rest], p[rest]))
    rel = pd.DataFrame(rows)
    out["ece_top"] = float(
        np.sum(rel["rows"] * (rel["mean_predicted"] - rel["observed_rate"]).abs()) / len(p)
    )
    return out, rel


def _rel_row(name, y, p) -> dict:
    return {
        "bin": name,
        "rows": len(y),
        "positives": int(y.sum()),
        "mean_predicted": float(p.mean()) if len(p) else np.nan,
        "observed_rate": float(y.mean()) if len(y) else np.nan,
    }


def val_operating_points(y, s) -> pd.DataFrame:
    """The seven operating points, chosen on validation scores."""
    precision, recall, thresholds = precision_recall_curve(np.asarray(y, bool), s)
    p, r = precision[:-1], recall[:-1]
    f1 = 2 * p * r / np.clip(p + r, 1e-12, None)
    rows = []
    for name, kind, level in OPERATING_POINTS:
        if kind is None:
            i = int(np.argmax(f1))
        elif kind == "precision":
            ok = np.flatnonzero(p >= level)
            i = int(ok[np.argmax(r[ok])]) if len(ok) else -1
        else:
            ok = np.flatnonzero(r >= level)
            i = int(ok[np.argmax(p[ok])]) if len(ok) else -1
        rows.append(
            {"operating_point": name, "threshold": float(thresholds[i]) if i >= 0 else np.nan}
        )
    return pd.DataFrame(rows)


def threshold_curve(y, s, max_rows: int = 400) -> pd.DataFrame:
    """Validation threshold curve over the top of the ranking (CSV) ."""
    y = np.asarray(y, bool)
    precision, recall, thresholds = precision_recall_curve(y, s)
    p, r, t = precision[:-1], recall[:-1], thresholds
    s_sorted = np.sort(np.asarray(s, float))
    flagged = len(s_sorted) - np.searchsorted(s_sorted, t, side="left")
    tp = r * y.sum()
    fp = flagged - tp
    neg = len(y) - y.sum()
    df = pd.DataFrame(
        {
            "threshold": t,
            "flagged": flagged,
            "precision": p,
            "recall": r,
            "f1": 2 * p * r / np.clip(p + r, 1e-12, None),
            "fpr": fp / max(neg, 1),
            "fnr": 1 - r,
        }
    )
    df = df[df["flagged"] <= max(20 * y.sum(), 5000)]
    if len(df) > max_rows:
        df = df.iloc[np.unique(np.linspace(0, len(df) - 1, max_rows).astype(int))]
    return df.sort_values("flagged").reset_index(drop=True)


def metric_block(y, s, thr, k_extra: dict | None = None, probabilistic=True) -> dict:
    y = np.asarray(y, bool)
    s = np.asarray(s, float)
    rk = Ranked(y, s)
    out = {
        "rows": len(y),
        "positives": int(y.sum()),
        "prevalence": float(y.mean()),
        "pr_auc": rk.ap(),
        "roc_auc": float(roc_auc_score(y, s)) if 0 < y.sum() < len(y) else np.nan,
    }
    ks = {f"K={k}": k for k in K_LIST}
    ks.update(k_extra or {})
    ks["R"] = int(y.sum())
    for name, k in ks.items():
        k = int(min(k, len(y)))
        hits = M.expected_in_top_k(y, s, k)  # tie-aware, as in the main tables
        out[f"P@{name}"] = hits / k if k else np.nan
        out[f"R@{name}"] = hits / y.sum() if y.sum() else np.nan
        if name == "R":
            out["K_R"] = k
    out.update({f"thr_{k}": v for k, v in confusion(y, s, thr).items()})
    if probabilistic:
        out.update(M.calibration(y, s))
        cal, _ = calibration_detail(y, s)
        out.update(cal)
    return out


# ----------------------------------------------------------------------------------------------
# Bootstrap (row and cluster), paired
# ----------------------------------------------------------------------------------------------


def bootstrap(
    y: np.ndarray,
    scores: dict[str, np.ndarray],
    thresholds: dict[str, float],
    units: np.ndarray | None,
    n: int,
    k_fixed: int = 100,
    seed: int = 0,
    paired: tuple[str, str] | None = None,
) -> pd.DataFrame:
    """Percentile 95% intervals of PR-AUC, P@k_fixed, R-precision, precision/recall/F1 at the
    frozen threshold, per model; plus PR-AUC(a) - PR-AUC(b) when `paired` is given.
    `units`: cluster id per row (None = rows)."""
    y = np.asarray(y, bool)
    rng = np.random.default_rng(seed)
    if units is None:
        units = np.arange(len(y))
    _, unit_index = np.unique(units, return_inverse=True)
    n_units = int(unit_index.max()) + 1
    ranked = {m: Ranked(y, s) for m, s in scores.items()}
    preds = {m: np.asarray(scores[m]) >= thresholds[m] for m in scores}
    draws = {
        m: {
            "pr_auc": [],
            f"P@{k_fixed}": [],
            "R-precision": [],
            "precision": [],
            "recall": [],
            "f1": [],
        }
        for m in scores
    }
    diffs = []
    for _ in range(n):
        counts = np.bincount(rng.integers(0, n_units, n_units), minlength=n_units)
        w = counts[unit_index].astype(float)
        pos = float(w[y].sum())
        if pos <= 0:
            continue
        aps = {}
        for m, rk in ranked.items():
            aps[m] = rk.ap(w)
            d = draws[m]
            d["pr_auc"].append(aps[m])
            d[f"P@{k_fixed}"].append(rk.hits_at(k_fixed, w) / k_fixed)
            d["R-precision"].append(rk.hits_at(pos, w) / pos)
            pr = preds[m]
            tp, fp, fn = w[pr & y].sum(), w[pr & ~y].sum(), w[~pr & y].sum()
            d["precision"].append(tp / (tp + fp) if tp + fp else np.nan)
            d["recall"].append(tp / (tp + fn) if tp + fn else np.nan)
            d["f1"].append(2 * tp / (2 * tp + fp + fn) if tp + fp + fn else np.nan)
        if paired:
            diffs.append(aps[paired[0]] - aps[paired[1]])
    rows = []
    for m, d in draws.items():
        for metric, values in d.items():
            v = np.asarray(values, float)
            lo, hi = np.nanpercentile(v, [2.5, 97.5]) if len(v) else (np.nan, np.nan)
            rows.append({"model": m, "metric": metric, "ci_low": float(lo), "ci_high": float(hi)})
    if paired and diffs:
        v = np.asarray(diffs)
        lo, hi = np.nanpercentile(v, [2.5, 97.5])
        rows.append(
            {
                "model": f"{paired[0]} - {paired[1]}",
                "metric": "pr_auc difference",
                "ci_low": float(lo),
                "ci_high": float(hi),
                "share_above_0": float(np.mean(v > 0)),
            }
        )
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------------------------
# Data helpers
# ----------------------------------------------------------------------------------------------


class Context:
    """Loads shared inputs lazily, once."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.manifest = load_manifest(cfg.paths.processed_dir / "split_manifest.json")
        self._tx = self._labels = self._feats = self._ptx = self._attempt = None
        self._txi = None

    @property
    def tx_by_id(self) -> pd.DataFrame:
        """Transactions indexed by tx_id (tx_id equals the sorted position, so this is cheap)."""
        if self._txi is None:
            self._txi = self.tx.set_index("tx_id")
        return self._txi

    @property
    def tx(self) -> pd.DataFrame:
        if self._tx is None:
            self._tx = read_parquet(self.cfg.paths.interim_dir / "transactions.parquet", TX_SCHEMA)
        return self._tx

    @property
    def labels(self) -> pd.DataFrame:
        if self._labels is None:
            lab = read_parquet(self.cfg.paths.processed_dir / "account_labels.parquet")
            self._labels = lab.assign(window=lab["window"].astype(str))
        return self._labels

    @property
    def feats(self) -> pd.DataFrame:
        if self._feats is None:
            f = read_parquet(self.cfg.paths.processed_dir / "account_features.parquet")
            self._feats = f.assign(window=f["window"].astype(str))
        return self._feats

    def patterns(self) -> tuple[pd.Series, pd.Series] | None:
        """(pattern type, attempt id) per laundering tx_id; None if the patterns file is absent."""
        if self._ptx is None:
            path = self.cfg.paths.raw_dir / f"{self.cfg.dataset}_Patterns.txt"
            if not path.exists():
                return None
            from mule.data.patterns import match_patterns, read_patterns

            index = read_parquet(self.cfg.paths.interim_dir / "account_index.parquet")
            found = match_patterns(read_patterns(path), self.tx, index).dropna(subset=["tx_id"])
            found = found.drop_duplicates("tx_id")
            tx_ids = found["tx_id"].astype(int).to_numpy()
            laundering = self.tx.loc[self.tx["is_laundering"] == 1, "tx_id"].to_numpy()
            self._ptx = pd.Series(found["pattern"].to_numpy(), index=tx_ids).reindex(laundering)
            self._ptx = self._ptx.fillna("none")
            self._attempt = pd.Series(found["attempt_id"].to_numpy(), index=tx_ids)
        return self._ptx, self._attempt

    def window_tx(self, window: str) -> pd.DataFrame:
        start, end = window_bounds(self.manifest, window)
        t = self.tx
        return t.loc[(t["ts"] >= start) & (t["ts"] < end)]

    def train_accounts(self) -> np.ndarray:
        lab = self.labels
        return lab.loc[lab["window"].isin(self.manifest["train_snapshots"]), "account_id"].unique()


def model_exists(cfg: Config, task: str, model: str) -> bool:
    return any((cfg.paths.runs_dir / task / model).glob("seed*/scores_val.parquet"))


def scores_for(cfg: Config, task: str, model: str, window: str) -> pd.DataFrame:
    return seed_mean(cfg, task, model, window)


def account_units(ctx: Context, window: str, df: pd.DataFrame) -> np.ndarray | None:
    """Cluster id per account row: a mule takes the attempt of its earliest patterned
    laundering transfer in the window; everyone else is their own unit."""
    pt = ctx.patterns()
    if pt is None:
        return None
    _, attempt = pt
    w = ctx.window_tx(window)
    lt = w[(w["is_laundering"] == 1) & w["tx_id"].isin(attempt.index)]
    tags = pd.concat(
        [
            pd.DataFrame(
                {
                    "account_id": lt["src_id"].to_numpy(),
                    "ts": lt["ts"].to_numpy(),
                    "tx_id": lt["tx_id"].to_numpy(),
                }
            ),
            pd.DataFrame(
                {
                    "account_id": lt["dst_id"].to_numpy(),
                    "ts": lt["ts"].to_numpy(),
                    "tx_id": lt["tx_id"].to_numpy(),
                }
            ),
        ]
    ).sort_values("ts")
    first = tags.drop_duplicates("account_id").set_index("account_id")["tx_id"]
    att = attempt.reindex(first.to_numpy()).to_numpy()
    by_account = pd.Series(att, index=first.index)
    unit = np.arange(len(df), dtype=np.int64) + 1_000_000
    mule = df["is_mule"].to_numpy(bool)
    a = by_account.reindex(df["account_id"].to_numpy()).to_numpy(float)
    use = mule & ~np.isnan(a)
    unit[use] = a[use].astype(np.int64)
    return unit


def edge_units(ctx: Context, df: pd.DataFrame) -> np.ndarray | None:
    pt = ctx.patterns()
    if pt is None:
        return None
    _, attempt = pt
    unit = np.arange(len(df), dtype=np.int64) + 1_000_000
    a = attempt.reindex(df["tx_id"].to_numpy()).to_numpy(float)
    use = ~np.isnan(a)
    unit[use] = a[use].astype(np.int64)
    return unit


# ----------------------------------------------------------------------------------------------
# Sections
# ----------------------------------------------------------------------------------------------


def _roles(lab_window: pd.DataFrame, ids: np.ndarray) -> np.ndarray:
    """mule / adjacent / other for account ids, from one window's labels (indexed by id)."""
    mule = lab_window["is_mule"].reindex(ids).astype("boolean").fillna(False).to_numpy(bool)
    adj = lab_window["adjacent"].reindex(ids).astype("boolean").fillna(False).to_numpy(bool)
    return np.where(mule, "mule", np.where(adj, "adjacent", "other"))


def dataset_section(ctx: Context, windows: list[str]) -> dict:
    tx = ctx.tx
    key = [
        "ts",
        "src_id",
        "dst_id",
        "amount_paid",
        "currency_paid",
        "amount_received",
        "currency_received",
        "payment_format",
    ]
    dup = tx.duplicated(subset=key, keep=False)
    dup_label = tx.duplicated(subset=[*key, "is_laundering"], keep=False)
    same_minute_src = tx.duplicated(subset=["ts", "src_id"], keep=False)
    out = {
        "transactions": len(tx),
        "accounts": int(np.union1d(tx["src_id"], tx["dst_id"]).size),
        "laundering": int(tx["is_laundering"].sum()),
        "duplicate_rows_all_fields": int(dup.sum()),
        "duplicate_groups_with_conflicting_label": int((dup & ~dup_label).sum()),
        "self_transfers": int(tx["is_self"].sum()),
        "laundering_self_transfers": int((tx["is_self"] & (tx["is_laundering"] == 1)).sum()),
        "rows_sharing_minute_with_same_sender": int(same_minute_src.sum()),
        "missing_values": {c: int(v) for c, v in tx.isna().sum().items() if v},
        "reinvestment_self_share": float(
            tx.loc[tx["payment_format"].astype(str) == "Reinvestment", "is_self"].mean()
        ),
    }
    lab = ctx.labels
    rows = []
    for w in [*ctx.manifest["train_snapshots"], *windows]:
        t = ctx.window_tx(w)
        lw = lab[lab["window"] == w]
        rows.append(
            {
                "window": w,
                "transactions": len(t),
                "laundering_tx": int(t["is_laundering"].sum()),
                "tx_prevalence (= no-skill PR-AUC)": float(t["is_laundering"].mean()),
                "accounts": len(lw),
                "mules": int(lw["is_mule"].sum()),
                "adjacent": int(lw["adjacent"].sum()),
                "account_prevalence (= no-skill PR-AUC)": float(lw["is_mule"].mean()),
                "ach_share_of_traffic": float((t["payment_format"].astype(str) == "ACH").mean()),
                "ach_share_of_laundering": float(
                    (t.loc[t["is_laundering"] == 1, "payment_format"].astype(str) == "ACH").mean()
                ),
            }
        )
    out["windows"] = rows
    return out


def lone_provenance(ctx: Context, windows: list[str]) -> pd.DataFrame | None:
    """Where 'lone' laundering (in no pattern attempt) sits relative to other laundering."""
    pt = ctx.patterns()
    if pt is None:
        return None
    pattern, _ = pt
    tx = ctx.tx
    lt = tx[(tx["is_laundering"] == 1) & ~tx["is_self"]]
    first_in = lt.groupby("dst_id")["ts"].min()
    rows = []
    lab = ctx.labels
    for w in [*ctx.manifest["train_snapshots"], *windows]:
        t = ctx.window_tx(w)
        launder = t[t["is_laundering"] == 1].copy()
        launder["kind"] = np.where(
            pattern.reindex(launder["tx_id"]).to_numpy() == "none", "lone", "patterned"
        )
        lw = lab[lab["window"] == w].set_index("account_id")
        launder["src_role"] = _roles(lw, launder["src_id"].to_numpy())
        launder["dst_role"] = _roles(lw, launder["dst_id"].to_numpy())
        fi = first_in.reindex(launder["src_id"].to_numpy()).to_numpy()
        ts = launder["ts"].to_numpy()
        launder["src_received_laundering_before"] = ~pd.isna(fi) & (fi <= ts)
        launder["src_received_within_72h"] = launder["src_received_laundering_before"] & (
            (ts - fi) <= np.timedelta64(72, "h")
        )
        launder["ach"] = launder["payment_format"].astype(str) == "ACH"
        launder["self"] = launder["is_self"]
        for kind, g in launder.groupby("kind"):
            rows.append(
                {
                    "window": w,
                    "kind": kind,
                    "transfers": len(g),
                    "self_transfer": g["self"].mean(),
                    "ACH": g["ach"].mean(),
                    "sender_is_mule": (g["src_role"] == "mule").mean(),
                    "sender_is_adjacent": (g["src_role"] == "adjacent").mean(),
                    "receiver_is_mule": (g["dst_role"] == "mule").mean(),
                    "sender_got_laundering_earlier": g["src_received_laundering_before"].mean(),
                    "sender_got_laundering_within_72h": g["src_received_within_72h"].mean(),
                }
            )
    return pd.DataFrame(rows)


def model_section(ctx, task, model, windows, boot_n, units_fn) -> dict:
    """Metrics, operating points, curves, calibration and CIs for one (seed-averaged) model."""
    cfg = ctx.cfg
    key, label = ID[task]
    frames = {w: scores_for(cfg, task, model, w) for w in windows}
    val = frames["val"]
    yv, sv = val[label].to_numpy(bool), val["score"].to_numpy(float)
    thr = M.best_f1_threshold(yv, sv)
    ops = val_operating_points(yv, sv)
    out = {"model": model, "val_threshold_seed_avg": thr, "windows": {}, "operating_points": {}}
    seed_thr = []
    for d in sorted((cfg.paths.runs_dir / task / model).glob("seed*")):
        seed_thr.append(json.loads((d / "run.json").read_text()).get("threshold"))
    out["per_seed_thresholds"] = seed_thr
    curve = threshold_curve(yv, sv)
    for w, df in frames.items():
        y, s = df[label].to_numpy(bool), df["score"].to_numpy(float)
        k_extra = (
            {"K=0.1%": max(1, round(len(df) * cfg.evaluation.k_share))} if task == "node" else {}
        )
        out["windows"][w] = metric_block(y, s, thr, k_extra)
        _, rel = calibration_detail(y, s)
        out["windows"][w]["reliability"] = rel.to_dict(orient="records")
        op_rows = []
        for r in ops.itertuples(index=False):
            c = confusion(y, s, r.threshold) if np.isfinite(r.threshold) else {}
            op_rows.append({"operating_point": r.operating_point, **c})
        out["operating_points"][w] = op_rows
        if boot_n:
            units = units_fn(w, df)
            boots = []
            for mode, u in (("row", None), ("cluster", units)):
                if mode == "cluster" and u is None:
                    continue
                b = bootstrap(y, {model: s}, {model: thr}, u, boot_n, cfg.evaluation.k_fixed)
                boots.append(b.assign(resampling=mode))
            out["windows"][w]["bootstrap"] = pd.concat(boots).to_dict(orient="records")
    out["val_curve"] = curve
    return out


def ensemble_section(ctx, task, ens, windows, boot_n, units_fn) -> dict | None:
    cfg = ctx.cfg
    path = cfg.paths.runs_dir / task / ens / "seed0" / "run.json"
    if not path.exists():
        return None
    info = json.loads(path.read_text())
    if "members" not in info:
        return None
    a, b = info["members"]
    key, label = ID[task]
    fa = {w: scores_for(cfg, task, a, w) for w in windows}
    fb = {w: scores_for(cfg, task, b, w).set_index(key)["score"] for w in windows}
    fe = {w: scores_for(cfg, task, ens, w).set_index(key)["score"] for w in windows}
    yv = fa["val"][label].to_numpy(int)
    ca = Platt().fit(fa["val"]["score"].to_numpy(float), yv)
    cb = Platt().fit(fb["val"].reindex(fa["val"][key]).to_numpy(float), yv)
    thr = {}
    out = {"members": [a, b], "weight_a": info.get("weight_a"), "windows": {}}
    out["val_pr_auc_by_weight"] = info.get("val_pr_auc_by_weight")
    for w in windows:
        ids = fa[w][key]
        y = fa[w][label].to_numpy(bool)
        sa = ca.predict(fa[w]["score"].to_numpy(float))
        sb = cb.predict(fb[w].reindex(ids).to_numpy(float))
        se = fe[w].reindex(ids).to_numpy(float)
        if w == "val":
            thr = {m: M.best_f1_threshold(y, s) for m, s in ((a, sa), (b, sb), (ens, se))}
        res = {
            "pr_auc": {a: Ranked(y, sa).ap(), b: Ranked(y, sb).ap(), ens: Ranked(y, se).ap()},
            "spearman_all": float(spearmanr(sa, sb).statistic),
            "spearman_positives": float(spearmanr(sa[y], sb[y]).statistic)
            if y.sum() > 2
            else np.nan,
        }
        best_member = max((a, b), key=lambda m: res["pr_auc"][m])
        res["best_member"] = best_member
        res["ensemble_gain"] = res["pr_auc"][ens] - res["pr_auc"][best_member]
        pa_, pb_ = sa >= thr[a], sb >= thr[b]
        pe_ = se >= thr[ens]
        res["overlap_positives_at_val_thresholds"] = {
            "both_caught": int((pa_ & pb_ & y).sum()),
            f"only_{a}": int((pa_ & ~pb_ & y).sum()),
            f"only_{b}": int((~pa_ & pb_ & y).sum()),
            "both_missed": int((~pa_ & ~pb_ & y).sum()),
            "ensemble_caught": int((pe_ & y).sum()),
        }
        res["overlap_negatives_at_val_thresholds"] = {
            "both_flagged": int((pa_ & pb_ & ~y).sum()),
            f"only_{a}": int((pa_ & ~pb_ & ~y).sum()),
            f"only_{b}": int((~pa_ & pb_ & ~y).sum()),
            "ensemble_flagged": int((pe_ & ~y).sum()),
        }
        k = int(y.sum())
        top_a = set(np.argsort(-sa, kind="stable")[:k])
        top_b = set(np.argsort(-sb, kind="stable")[:k])
        pos = set(np.flatnonzero(y))
        res["top_R_overlap"] = {
            "jaccard_of_top_R": len(top_a & top_b) / max(len(top_a | top_b), 1),
            "mules_in_both_top_R": len(top_a & top_b & pos),
            f"mules_only_in_{a}_top_R": len((top_a - top_b) & pos),
            f"mules_only_in_{b}_top_R": len((top_b - top_a) & pos),
            "mules_in_neither": len(pos - top_a - top_b),
        }
        if boot_n:
            units = units_fn(w, fa[w])
            b_ = bootstrap(
                y,
                {ens: se, best_member: sa if best_member == a else sb},
                {ens: thr[ens], best_member: thr[best_member]},
                units,
                boot_n,
                cfg.evaluation.k_fixed,
                paired=(ens, best_member),
            )
            res["gain_ci"] = b_[b_["metric"] == "pr_auc difference"].to_dict(orient="records")
        out["windows"][w] = res
    return out


def pattern_section(ctx, task, model, windows) -> dict | None:
    pt = ctx.patterns()
    if pt is None:
        return None
    pattern, _ = pt
    cfg = ctx.cfg
    key, label = ID[task]
    val = scores_for(cfg, task, model, "val")
    thr = M.best_f1_threshold(val[label].to_numpy(bool), val["score"].to_numpy(float))
    out = {}
    for w in windows:
        df = scores_for(cfg, task, model, w)
        y, s = df[label].to_numpy(bool), df["score"].to_numpy(float)
        flagged = s >= thr
        total_pos, total_caught = int(y.sum()), int((flagged & y).sum())
        rows = []
        if task == "edge":
            p = pattern.reindex(df.loc[y, key].to_numpy()).fillna("none").to_numpy()
            caught = flagged[y]
            for name in sorted(set(p)):
                m = p == name
                rows.append(
                    _pattern_row(name, int(m.sum()), int(caught[m].sum()), total_pos, total_caught)
                )
        else:
            t = ctx.window_tx(w)
            lt = t[t["is_laundering"] == 1]
            tags = pd.concat(
                [
                    pd.DataFrame(
                        {
                            "account_id": lt["src_id"].to_numpy(),
                            "pattern": pattern.reindex(lt["tx_id"]).to_numpy(),
                        }
                    ),
                    pd.DataFrame(
                        {
                            "account_id": lt["dst_id"].to_numpy(),
                            "pattern": pattern.reindex(lt["tx_id"]).to_numpy(),
                        }
                    ),
                ]
            ).drop_duplicates()
            mule_ids = df.loc[y, key].to_numpy()
            tags = tags[tags["account_id"].isin(mule_ids)]
            only_lone = tags.groupby("account_id")["pattern"].apply(lambda x: set(x) == {"none"})
            ids_all = df[key].to_numpy()
            k_r = int(y.sum())
            for name, g in tags.groupby("pattern"):
                m = np.isin(ids_all, g["account_id"].to_numpy()) & y
                r = _pattern_row(
                    name, int(m.sum()), int((flagged & m).sum()), total_pos, total_caught
                )
                r["recall@R"] = M.expected_in_top_k(m, s, k_r) / max(m.sum(), 1)
                rows.append(r)
            lone_ids = only_lone.index[only_lone.to_numpy()]
            m = np.isin(ids_all, lone_ids) & y
            r = _pattern_row(
                "lone only", int(m.sum()), int((flagged & m).sum()), total_pos, total_caught
            )
            r["recall@R"] = M.expected_in_top_k(m, s, k_r) / max(m.sum(), 1)
            rows.append(r)
        out[w] = rows
    return out


def _pattern_row(name, positives, caught, total_pos, total_caught) -> dict:
    return {
        "pattern": name,
        "positives": positives,
        "share_of_positives": positives / max(total_pos, 1),
        "recall@val_threshold": caught / positives if positives else np.nan,
        "false_negatives": positives - caught,
        "share_of_all_caught": caught / max(total_caught, 1),
        "small_sample": positives < 30,
    }


def temporal_section(ctx, model, windows) -> dict:
    """Transaction task, per calendar day, at the frozen validation threshold."""
    cfg = ctx.cfg
    val = scores_for(cfg, "edge", model, "val")
    thr = M.best_f1_threshold(val["is_laundering"].to_numpy(bool), val["score"].to_numpy(float))
    tx = ctx.tx_by_id["ts"]
    out = {}
    for w in [x for x in windows if x != "test_no_tail"]:
        df = scores_for(cfg, "edge", model, w)
        day = tx.reindex(df["tx_id"].to_numpy()).dt.floor("D").to_numpy()
        rows = []
        for d in np.unique(day):
            m = day == d
            y, s = df["is_laundering"].to_numpy(bool)[m], df["score"].to_numpy(float)[m]
            c = confusion(y, s, thr)
            rows.append(
                {
                    "day": str(pd.Timestamp(d).date()),
                    "transactions": int(m.sum()),
                    "laundering": int(y.sum()),
                    "pr_auc": Ranked(y, s).ap() if y.sum() else np.nan,
                    "precision": c["precision"],
                    "recall": c["recall"],
                    "f1": c["f1"],
                    "small_sample": y.sum() < 30,
                }
            )
        out[w] = rows
    return out


def channel_section(ctx, task, model, windows) -> dict:
    cfg = ctx.cfg
    key, label = ID[task]
    val = scores_for(cfg, task, model, "val")
    thr = M.best_f1_threshold(val[label].to_numpy(bool), val["score"].to_numpy(float))
    out = {}
    for w in windows:
        df = scores_for(cfg, task, model, w)
        y, s = df[label].to_numpy(bool), df["score"].to_numpy(float)
        if task == "edge":
            fmt = (
                ctx.tx_by_id["payment_format"]
                .astype(str)
                .reindex(df["tx_id"].to_numpy())
                .to_numpy()
            )
            groups = {f: fmt == f for f in FORMATS}
        else:
            f = ctx.feats[ctx.feats["window"] == w].set_index("account_id")
            share_cols = [c for c in f.columns if c.startswith("ch_share_")]
            sh = f[share_cols].reindex(df["account_id"].to_numpy())
            dominant = np.where(
                sh.notna().any(axis=1).to_numpy(),
                sh.fillna(-1).to_numpy().argmax(axis=1).astype(str),
                "none",
            )
            names = {str(i): c.replace("ch_share_", "") for i, c in enumerate(share_cols)}
            dominant = np.array([names.get(d, d) for d in dominant])
            groups = {f"dominant {n}": dominant == n for n in sorted(set(dominant))}
            ach = (
                sh["ch_share_ach"].to_numpy() if "ch_share_ach" in sh else np.full(len(df), np.nan)
            )
            groups["ACH share >= 0.5"] = ach >= 0.5
            groups["ACH share < 0.5"] = ~(ach >= 0.5)
        rows = []
        for name, m in groups.items():
            if not m.any():
                continue
            c = confusion(y[m], s[m], thr)
            rows.append(
                {
                    "group": name,
                    "rows": int(m.sum()),
                    "positives": int(y[m].sum()),
                    "flagged": c["flagged"],
                    "precision": c["precision"],
                    "recall": c["recall"],
                    "f1": c["f1"],
                    "pr_auc_within_group": Ranked(y[m], s[m]).ap() if y[m].sum() >= 20 else np.nan,
                    "small_sample": y[m].sum() < 30,
                }
            )
        out[w] = rows
    return out


def unseen_section(ctx, task, model, windows) -> dict:
    cfg = ctx.cfg
    key, label = ID[task]
    val = scores_for(cfg, task, model, "val")
    thr = M.best_f1_threshold(val[label].to_numpy(bool), val["score"].to_numpy(float))
    train_acc = ctx.train_accounts()
    snaps = ctx.manifest["train_snapshots"]
    start_train = window_bounds(ctx.manifest, snaps[0])[0]
    end_train = window_bounds(ctx.manifest, snaps[-1])[1]
    ttx = ctx.tx[(ctx.tx["ts"] >= start_train) & (ctx.tx["ts"] < end_train)]
    out = {}
    for w in windows:
        df = scores_for(cfg, task, model, w)
        y, s = df[label].to_numpy(bool), df["score"].to_numpy(float)
        groups = {}
        if task == "node":
            ids = df["account_id"].to_numpy()
            seen = np.isin(ids, train_acc)
            groups["active in training"] = seen
            groups["never active in training"] = ~seen
            groups["repeat mule (prior window) + all non-mules"] = (
                df["prior_mule"].to_numpy(bool) | ~y
            )
            groups["first-time mules + all non-mules"] = ~df["prior_mule"].to_numpy(bool) | ~y
            t = ctx.window_tx(w)
            t = t[~t["is_self"]]
            cp = pd.concat(
                [
                    pd.DataFrame({"a": t["src_id"].to_numpy(), "c": t["dst_id"].to_numpy()}),
                    pd.DataFrame({"a": t["dst_id"].to_numpy(), "c": t["src_id"].to_numpy()}),
                ]
            ).drop_duplicates()
            cp["seen"] = np.isin(cp["c"].to_numpy(), train_acc)
            frac = cp.groupby("a")["seen"].mean().reindex(ids).to_numpy()
            groups["most counterparties unseen in training (<50% seen)"] = frac < 0.5
            groups["most counterparties seen in training"] = frac >= 0.5
        else:
            tw = ctx.tx_by_id.reindex(df["tx_id"].to_numpy())
            src_seen = np.isin(tw["src_id"].to_numpy(), train_acc)
            dst_seen = np.isin(tw["dst_id"].to_numpy(), train_acc)
            pairs = pd.MultiIndex.from_arrays(
                [ttx["src_id"].to_numpy(), ttx["dst_id"].to_numpy()]
            ).unique()
            pair_seen = pd.MultiIndex.from_arrays(
                [tw["src_id"].to_numpy(), tw["dst_id"].to_numpy()]
            ).isin(pairs)
            groups["sender and receiver seen in training"] = src_seen & dst_seen
            groups["sender or receiver new"] = ~(src_seen & dst_seen)
            groups["both endpoints new"] = ~src_seen & ~dst_seen
            groups["pair traded in training"] = pair_seen
            groups["new pair"] = ~pair_seen
        rows = []
        for name, m in groups.items():
            if not m.any():
                continue
            c = confusion(y[m], s[m], thr)
            k = min(100, int(m.sum()))
            rows.append(
                {
                    "group": name,
                    "rows": int(m.sum()),
                    "positives": int(y[m].sum()),
                    "pr_auc_within_group": Ranked(y[m], s[m]).ap() if y[m].sum() >= 10 else np.nan,
                    "P@100_within_group": M.expected_in_top_k(y[m], s[m], k) / k if k else np.nan,
                    "precision": c["precision"],
                    "recall": c["recall"],
                    "f1": c["f1"],
                    "small_sample": y[m].sum() < 30,
                }
            )
        out[w] = rows
    return out


def error_section(ctx, task, model, windows) -> dict:
    cfg = ctx.cfg
    key, label = ID[task]
    val = scores_for(cfg, task, model, "val")
    thr = M.best_f1_threshold(val[label].to_numpy(bool), val["score"].to_numpy(float))
    pt = ctx.patterns()
    out = {}
    for w in windows:
        df = scores_for(cfg, task, model, w)
        y, s = df[label].to_numpy(bool), df["score"].to_numpy(float)
        pred = s >= thr
        outcome = np.where(pred & y, "TP", np.where(pred & ~y, "FP", np.where(y, "FN", "TN")))
        res = {"counts": pd.Series(outcome).value_counts().to_dict()}
        if task == "node":
            adj = df["adjacent"].to_numpy(bool)
            res["false_positives_adjacent"] = int((pred & ~y & adj).sum())
            res["false_positives_uninvolved"] = int((pred & ~y & ~adj).sum())
            f = ctx.feats[ctx.feats["window"] == w].set_index("account_id")
            cols = [c for c in PROFILE_FEATURES if c in f.columns]
            prof = f[cols].reindex(df["account_id"].to_numpy()).reset_index(drop=True)
            prof["outcome"] = outcome
            tn = prof[prof["outcome"] == "TN"]
            sample = tn.sample(min(len(tn), 20000), random_state=0)
            prof = pd.concat([prof[prof["outcome"] != "TN"], sample])
            res["median_profile"] = (
                prof.groupby("outcome")[cols].median().reset_index().to_dict(orient="records")
            )
            # false-negative clusters: pattern tag x receipt timing x dominant channel
            start, _ = window_bounds(ctx.manifest, w)
            tx = ctx.tx
            lt = tx[
                (tx["is_laundering"] == 1)
                & ~tx["is_self"]
                & (tx["ts"] < window_bounds(ctx.manifest, w)[1])
            ]
            first_in = lt.groupby("dst_id")["ts"].min()
            fn_ids = df.loc[outcome == "FN", "account_id"].to_numpy()
            tp_ids = df.loc[outcome == "TP", "account_id"].to_numpy()
            rows = []
            for name, ids in (("FN", fn_ids), ("TP", tp_ids)):
                fi = first_in.reindex(ids)
                ach = (
                    f["ch_share_ach"].reindex(ids)
                    if "ch_share_ach" in f
                    else pd.Series(np.nan, index=ids)
                )
                rows.append(
                    {
                        "outcome": name,
                        "mules": len(ids),
                        "first_laundering_receipt_before_window": float((fi < start).mean())
                        if len(ids)
                        else np.nan,
                        "ach_share_median": float(ach.median()) if len(ids) else np.nan,
                        "repeat_mule_share": float(
                            df.set_index("account_id")["prior_mule"].reindex(ids).mean()
                        )
                        if len(ids)
                        else np.nan,
                    }
                )
            res["fn_vs_tp"] = rows
        else:
            tw = ctx.tx_by_id.reindex(df["tx_id"].to_numpy())
            lab = ctx.labels[ctx.labels["window"] == w].set_index("account_id")

            frame = pd.DataFrame(
                {
                    "outcome": outcome,
                    "pattern": (
                        pt[0].reindex(df["tx_id"].to_numpy()).fillna("none").to_numpy()
                        if pt is not None
                        else "unknown"
                    ),
                    "format": tw["payment_format"].astype(str).to_numpy(),
                    "sender": _roles(lab, tw["src_id"].to_numpy()),
                    "receiver": _roles(lab, tw["dst_id"].to_numpy()),
                    "self": tw["is_self"].to_numpy(),
                    "log_usd": np.log10(tw["amount_usd"].to_numpy() + 1),
                }
            )
            fn = frame[frame["outcome"] == "FN"]
            res["fn_clusters"] = (
                fn.groupby(["pattern", "format", "sender", "receiver"])
                .size()
                .rename("false_negatives")
                .sort_values(ascending=False)
                .head(15)
                .reset_index()
                .to_dict(orient="records")
            )
            pos = frame[frame["outcome"].isin(["TP", "FN"])]
            res["positives_by_outcome"] = (
                pos.groupby("outcome")
                .agg(
                    transfers=("format", "size"),
                    ach=("format", lambda x: (x == "ACH").mean()),
                    self_transfer=("self", "mean"),
                    median_log10_usd=("log_usd", "median"),
                    sender_is_mule=("sender", lambda x: (x == "mule").mean()),
                    receiver_is_mule=("receiver", lambda x: (x == "mule").mean()),
                )
                .reset_index()
                .to_dict(orient="records")
            )
            fp = frame[frame["outcome"] == "FP"]
            res["fp_clusters"] = (
                fp.groupby(["format", "sender", "receiver"])
                .size()
                .rename("false_positives")
                .sort_values(ascending=False)
                .head(10)
                .reset_index()
                .to_dict(orient="records")
            )
        out[w] = res
    return out


def importance_section(ctx, task, model, permutation_repeats: int = 3) -> dict | None:
    """XGBoost gain / split count / cover (seed mean, normalised), SHAP mean |value| and
    group permutation importance on validation (account task only: the edge matrix is large)."""
    import xgboost as xgb

    cfg = ctx.cfg
    seeds = sorted((cfg.paths.runs_dir / task / model).glob("seed*"))
    if not seeds or not (seeds[0] / "model.json").exists():
        return None
    cols = json.loads((seeds[0] / "run.json").read_text())["feature_columns"]
    tables = []
    for d in seeds:
        b = xgb.Booster()
        b.load_model(d / "model.json")
        row = {}
        for kind in ("total_gain", "weight", "total_cover"):
            sc = b.get_score(importance_type=kind)
            total = sum(sc.values()) or 1.0
            row[kind] = pd.Series({c: sc.get(c, 0.0) / total for c in cols})
        tables.append(pd.DataFrame(row))
    imp = sum(tables) / len(tables)
    imp = imp.rename(
        columns={"total_gain": "gain_share", "weight": "split_share", "total_cover": "cover_share"}
    )
    out = {
        "top_gain_share": float(imp["gain_share"].max()),
        "top3_gain_share": float(imp["gain_share"].nlargest(3).sum()),
    }
    if task == "node":
        from mule.explain.shap_part import seed_shap
        from mule.models.data import load_node

        data = load_node(cfg)["val"]
        X = data.X
        sh = seed_shap(cfg, task, model, X)
        imp["shap_mean_abs"] = sh.abs().mean().reindex(imp.index)
        base_scores = scores_for(cfg, task, model, "val")
        y = data.y.astype(bool)
        base = Ranked(y, base_scores["score"].to_numpy(float)).ap()
        rng = np.random.default_rng(0)
        models = []
        for d in seeds:
            b = xgb.Booster()
            b.load_model(d / "model.json")
            info = json.loads((d / "run.json").read_text())
            models.append((b, int(info["best_iteration"])))

        def predict(frame):
            dm = xgb.DMatrix(frame[cols])
            return np.mean([b.predict(dm, iteration_range=(0, it + 1)) for b, it in models], axis=0)

        groups = sorted({c.split("_")[0] for c in cols})
        perm = []
        for g in groups:
            gcols = [c for c in cols if c.split("_")[0] == g]
            drops = []
            for _ in range(permutation_repeats):
                Xp = X.copy()
                idx = rng.permutation(len(Xp))
                Xp[gcols] = Xp[gcols].to_numpy()[idx]
                drops.append(base - Ranked(y, predict(Xp)).ap())
            perm.append(
                {
                    "group": g,
                    "features": len(gcols),
                    "pr_auc_drop_mean": float(np.mean(drops)),
                    "pr_auc_drop_std": float(np.std(drops)),
                }
            )
        out["group_permutation"] = perm
        out["val_pr_auc_unpermuted"] = base
    out["features"] = (
        imp.sort_values("gain_share", ascending=False)
        .reset_index(names="feature")
        .to_dict(orient="records")
    )
    return out


def train_fit_section(ctx, model) -> dict | None:
    """In-sample (training snapshot) vs validation PR-AUC per seed, account task XGBoost."""
    import xgboost as xgb

    from mule.models.data import load_node

    cfg = ctx.cfg
    seeds = sorted((cfg.paths.runs_dir / "node" / model).glob("seed*"))
    if not seeds or not (seeds[0] / "model.json").exists():
        return None
    data = load_node(cfg)
    rows = []
    for d in seeds:
        info = json.loads((d / "run.json").read_text())
        b = xgb.Booster()
        b.load_model(d / "model.json")
        it = int(info["best_iteration"])
        row = {"seed": d.name, "best_iteration": it}
        for part in ("train", "val"):
            X = data[part].X[info["feature_columns"]]
            p = b.predict(xgb.DMatrix(X), iteration_range=(0, it + 1))
            row[f"{part}_pr_auc"] = Ranked(data[part].y.astype(bool), p).ap()
        meta = data["train"].meta
        for snap in ctx.manifest["train_snapshots"]:
            m = (meta["window"].astype(str) == snap).to_numpy()
            X = data["train"].X.loc[m, info["feature_columns"]]
            p = b.predict(xgb.DMatrix(X), iteration_range=(0, it + 1))
            row[f"{snap}_pr_auc"] = Ranked(data["train"].y[m].astype(bool), p).ap()
        rows.append(row)
    return {"rows": rows}


def reconciliation(cfg, task, models, windows, sections) -> list[dict]:
    """Audit values (seed-averaged scores) next to the stored metrics (per-seed means)."""
    rows = []
    for model in models:
        d = cfg.paths.runs_dir / task / model
        stored_file = d / "metrics_final.json"
        if not stored_file.exists():
            stored_file = d / "metrics.json"
        if not stored_file.exists() or model not in sections:
            continue
        stored = json.loads(stored_file.read_text())["windows"]
        for w in windows:
            if w not in stored or w not in sections[model]["windows"]:
                continue
            a = sections[model]["windows"][w]
            st = stored[w]
            pairs = [("pr_auc", "pr_auc")]
            if task == "node":
                pairs += [
                    ("P@K=100", "precision@K=100"),
                    ("P@R", "precision@R-precision"),
                    ("thr_f1", "f1@val_threshold"),
                ]
            else:
                pairs += [
                    ("thr_f1", "minority_f1"),
                    ("thr_precision", "precision@val_threshold"),
                    ("thr_recall", "recall@val_threshold"),
                ]
            for ak, sk in pairs:
                if sk in st:
                    rows.append(
                        {
                            "model": model,
                            "window": w,
                            "metric": ak,
                            "audit (seed-averaged)": a.get(ak),
                            "stored (mean of seeds)": st[sk]["mean"],
                        }
                    )
    return rows


# ----------------------------------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------------------------------


def _df(rows) -> pd.DataFrame:
    return pd.DataFrame(rows) if not isinstance(rows, pd.DataFrame) else rows


def _clean(obj):
    if isinstance(obj, dict):
        return {str(k): _clean(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [_clean(v) for v in obj]
    if isinstance(obj, pd.DataFrame):
        return _clean(obj.to_dict(orient="records"))
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating | float):
        return None if not np.isfinite(obj) else float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


def run_audit(
    cfg: Config,
    final: bool = False,
    node_champion: str = "ens_xgb_graph__gnn_sage",
    edge_champion: str = "ens_xgb_tuned__gnn_sage",
    node_extra: tuple[str, ...] = ("xgb_graph", "gnn_sage", "xgb_pattern"),
    edge_extra: tuple[str, ...] = ("xgb_tuned", "gnn_sage", "xgb_pattern"),
    boot_node: int = 1000,
    boot_edge: int = 200,
    train_fit: bool = False,
) -> Path:
    windows = ["val", *(("test", "test_no_tail") if final else ())]
    for w in windows:
        check_window_allowed(w, final)
    ctx = Context(cfg)
    out_dir = cfg.paths.reports_dir / "audit"
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = "final" if final else "val"
    result: dict = {"windows": windows}
    rep = Report(f"Audit measurements ({'validation + test' if final else 'validation only'})")
    rep.note(
        "Scores are seed-averaged. Every threshold and operating point is chosen on validation "
        "and applied unchanged to other windows. small_sample = fewer than 30 positives: "
        "treat those rows as anecdotes."
    )

    print("dataset", flush=True)
    ds = dataset_section(ctx, windows)
    result["dataset"] = ds
    rep.section("1. Dataset")
    for k in (
        "transactions",
        "accounts",
        "laundering",
        "duplicate_rows_all_fields",
        "duplicate_groups_with_conflicting_label",
        "self_transfers",
        "laundering_self_transfers",
        "rows_sharing_minute_with_same_sender",
        "reinvestment_self_share",
    ):
        rep.item(k, ds[k])
    rep.item("missing values", ds["missing_values"] or "none")
    rep.table(_df(ds["windows"]))

    print("lone laundering provenance", flush=True)
    lone = lone_provenance(ctx, windows)
    if lone is not None:
        result["lone_provenance"] = lone
        rep.section("2. Lone vs patterned laundering (shares of transfers)")
        rep.table(lone)

    plans = [
        ("node", node_champion, [m for m in node_extra if model_exists(cfg, "node", m)], boot_node),
        ("edge", edge_champion, [m for m in edge_extra if model_exists(cfg, "edge", m)], boot_edge),
    ]
    for task, champ, extra, boot_n in plans:
        if not model_exists(cfg, task, champ):
            print(f"skipping {task}: {champ} not found", flush=True)
            continue
        name = "Accounts" if task == "node" else "Transactions"
        units_fn = (
            (lambda w, df: account_units(ctx, w, df))
            if task == "node"
            else (lambda w, df: edge_units(ctx, df))
        )
        sections = {}
        for model in [champ, *extra]:
            print(f"{task}: metrics for {model}", flush=True)
            sec = model_section(
                ctx, task, model, windows, boot_n if model == champ else 0, units_fn
            )
            sec["val_curve"].to_csv(
                out_dir / f"threshold_curve_{task}_{model}_val.csv", index=False
            )
            sections[model] = sec
        result[f"{task}_models"] = {
            m: {k: v for k, v in s.items() if k != "val_curve"} for m, s in sections.items()
        }

        rep.section(f"3. {name}: metrics (seed-averaged; threshold = max F1 on val)")
        for w in windows:
            rows = []
            for model, sec in sections.items():
                m = sec["windows"][w]
                rows.append(
                    {
                        "model": model,
                        **{k: v for k, v in m.items() if not isinstance(v, list | dict)},
                    }
                )
            rep.note(f"Window {w}")
            rep.table(
                _df(rows).set_index("model").T.reset_index().rename(columns={"index": "metric"})
            )
        rep.section(f"4. {name}: operating points chosen on val ({champ})")
        for w in windows:
            rep.note(f"Window {w}")
            rep.table(
                _df(sections[champ]["operating_points"][w])[
                    [
                        "operating_point",
                        "threshold",
                        "flagged",
                        "tp",
                        "fp",
                        "precision",
                        "recall",
                        "f1",
                    ]
                ]
            )
        rep.section(f"5. {name}: 95% intervals for {champ} (row vs cluster bootstrap)")
        for w in windows:
            b = sections[champ]["windows"][w].get("bootstrap")
            if b:
                rep.note(f"Window {w}")
                rep.table(_df(b))
        rep.section(f"6. {name}: reliability of {champ} (equal-mass bins over the top scores)")
        for w in windows:
            rep.note(f"Window {w}")
            rep.table(_df(sections[champ]["windows"][w]["reliability"]))

        print(f"{task}: ensemble", flush=True)
        ens = ensemble_section(ctx, task, champ, windows, boot_n, units_fn)
        if ens:
            result[f"{task}_ensemble"] = ens
            rep.section(
                f"7. {name}: ensemble audit ({champ}, "
                f"weight on {ens['members'][0]} = {ens['weight_a']})"
            )
            for w in windows:
                e = ens["windows"][w]
                rep.note(f"Window {w}")
                rep.item(
                    "PR-AUC (seed-averaged, Platt)",
                    {k: round(v, 4) for k, v in e["pr_auc"].items()},
                )
                rep.item("ensemble gain over best member", round(e["ensemble_gain"], 4))
                if e.get("gain_ci"):
                    ci = e["gain_ci"][0]
                    rep.item("gain 95% CI (paired)", f"[{ci['ci_low']:.4f}, {ci['ci_high']:.4f}]")
                rep.item(
                    "Spearman (all rows / positives)",
                    f"{e['spearman_all']:.3f} / {e['spearman_positives']:.3f}",
                )
                rep.table(_df([e["overlap_positives_at_val_thresholds"]]))
                rep.table(_df([e["overlap_negatives_at_val_thresholds"]]))
                rep.table(_df([e["top_R_overlap"]]))

        print(f"{task}: patterns, channels, unseen, errors", flush=True)
        pat = pattern_section(ctx, task, champ, windows)
        if pat:
            result[f"{task}_patterns"] = pat
            rep.section(f"8. {name}: pattern-level recall ({champ})")
            for w in windows:
                rep.note(f"Window {w}")
                rep.table(_df(pat[w]))
        ch = channel_section(ctx, task, champ, windows)
        result[f"{task}_channels"] = ch
        rep.section(f"9. {name}: by payment channel ({champ})")
        for w in windows:
            rep.note(f"Window {w}")
            rep.table(_df(ch[w]))
        un = unseen_section(ctx, task, champ, windows)
        result[f"{task}_unseen"] = un
        rep.section(f"10. {name}: familiar vs unseen entities ({champ})")
        for w in windows:
            rep.note(f"Window {w}")
            rep.table(_df(un[w]))
        er = error_section(ctx, task, champ, windows)
        result[f"{task}_errors"] = er
        rep.section(f"11. {name}: error analysis ({champ}, val threshold)")
        for w in windows:
            rep.note(f"Window {w}: {er[w]['counts']}")
            for k, v in er[w].items():
                if k == "counts":
                    continue
                if isinstance(v, list):
                    rep.note(k)
                    rep.table(_df(v))
                else:
                    rep.item(k, v)
        if task == "edge":
            tm = temporal_section(ctx, champ, windows)
            result["edge_temporal"] = tm
            rep.section("12. Transactions: by calendar day (frozen val threshold)")
            for w, rows in tm.items():
                rep.note(f"Window {w}")
                rep.table(_df(rows))
        rec = reconciliation(cfg, task, [champ, *extra], windows, sections)
        if rec:
            result[f"{task}_reconciliation"] = rec
            rep.section(f"13. {name}: reconciliation with stored metrics")
            rep.table(_df(rec))

    print("feature importance", flush=True)
    for task, model in (("node", "xgb_graph"), ("edge", "xgb_tuned")):
        if not model_exists(cfg, task, model):
            continue
        imp = importance_section(ctx, task, model)
        if imp:
            result[f"{task}_importance_{model}"] = imp
            rep.section(f"14. Feature importance: {task} {model}")
            rep.item("largest single-feature gain share", round(imp["top_gain_share"], 3))
            rep.item("top-3 gain share", round(imp["top3_gain_share"], 3))
            if "group_permutation" in imp:
                rep.item("val PR-AUC before permutation", round(imp["val_pr_auc_unpermuted"], 4))
                rep.table(_df(imp["group_permutation"]))
            rep.table(_df(imp["features"]).head(25))
    if train_fit and model_exists(cfg, "node", "xgb_graph"):
        print("train fit", flush=True)
        tf = train_fit_section(ctx, "xgb_graph")
        if tf:
            result["node_train_fit_xgb_graph"] = tf
            rep.section("15. Train vs validation fit: node xgb_graph (in-sample PR-AUC)")
            rep.table(_df(tf["rows"]))

    (cfg.paths.reports_dir / f"audit_{tag}.json").write_text(json.dumps(_clean(result), indent=1))
    path = cfg.paths.reports_dir / f"audit_{tag}.md"
    path.write_text(rep.text())
    return path
