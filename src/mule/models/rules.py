"""Hand-set rule baselines: the floor every learned model has to beat.

rule_ach: rank by ACH share alone. Phase 0 found ACH carries 87% of laundering transactions,
so this tests whether a model learns anything beyond "uses a lot of ACH".
rules: count of three analyst-style conditions (mostly ACH, most money forwarded within 72 h,
little money left behind), with ACH share breaking ties.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from mule.config import RulesConfig
from mule.models.base import Model


def _col(X: pd.DataFrame, name: str) -> np.ndarray:
    return X[name].fillna(0).to_numpy(dtype=float)


class AchShareRule(Model):
    name, deterministic = "rule_ach", True

    def __init__(self, task: str):
        self.prefix = "" if task == "node" else "src_"
        self.task = task

    def fit(self, train, val, seed: int) -> dict:
        return {}

    def _parts(self, X: pd.DataFrame) -> dict[str, np.ndarray]:
        if self.task == "node":
            return {"ch_share_ach": _col(X, "ch_share_ach")}
        # transaction task: is this transfer ACH, ties broken by the sender's ACH share
        return {
            "e_fmt_ach": _col(X, "e_fmt_ach"),
            "src_ch_share_ach": 1e-3 * _col(X, "src_ch_share_ach"),
        }

    def predict_scores(self, X: pd.DataFrame) -> np.ndarray:
        return np.sum(list(self._parts(X).values()), axis=0)

    def explain(self, X: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(self._parts(X), index=X.index).reindex(
            columns=X.columns, fill_value=0.0
        )


class AnalystRules(Model):
    name, deterministic = "rules", True

    def __init__(self, task: str, cfg: RulesConfig):
        self.task, self.cfg = task, cfg

    def fit(self, train, val, seed: int) -> dict:
        return {}

    def params(self) -> dict:
        return self.cfg.model_dump()

    def _parts(self, X: pd.DataFrame) -> dict[str, np.ndarray]:
        p = "" if self.task == "node" else "dst_"  # transaction task: judge the receiver
        net = X[f"{p}pt_net_flow"].to_numpy(dtype=float)
        parts = {
            f"{p}ch_share_ach": (_col(X, f"{p}ch_share_ach") >= self.cfg.ach_share_min) * 1.0,
            f"{p}pt_fwd_72h_share": (_col(X, f"{p}pt_fwd_72h_share") >= self.cfg.fwd_72h_share_min)
            * 1.0,
            f"{p}pt_net_flow": (np.nan_to_num(np.abs(net), nan=1.0) <= self.cfg.net_flow_abs_max)
            * 1.0,
        }
        if self.task == "edge":
            parts["e_fmt_ach"] = _col(X, "e_fmt_ach")
        parts["tiebreak"] = 0.01 * _col(X, f"{p}ch_share_ach")
        return parts

    def predict_scores(self, X: pd.DataFrame) -> np.ndarray:
        return np.sum(list(self._parts(X).values()), axis=0)

    def explain(self, X: pd.DataFrame) -> pd.DataFrame:
        parts = self._parts(X)
        parts.pop("tiebreak")
        return pd.DataFrame(parts, index=X.index).reindex(columns=X.columns, fill_value=0.0)
