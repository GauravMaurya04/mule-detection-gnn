"""XGBoost baseline. Explanations are exact TreeSHAP values from XGBoost itself (pred_contribs)."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

from mule.config import XGBConfig
from mule.models.base import Model


class XGBModel(Model):
    name = "xgb"
    probabilistic = True

    def __init__(self, cfg: XGBConfig, name: str = "xgb", groups: tuple[str, ...] | None = None):
        self.cfg, self.name, self.groups = cfg, name, groups
        self.booster: xgb.Booster | None = None
        self.best_iteration = 0

    def params(self) -> dict:
        return {**self.cfg.model_dump(), "groups": ",".join(self.groups or ("all",))}

    def fit(self, train, val, seed: int) -> dict:
        pos = int(train.y.sum())
        neg = len(train.y) - pos
        params = {
            "objective": "binary:logistic",
            "eval_metric": "aucpr",
            "tree_method": "hist",
            "device": self.cfg.device,
            "learning_rate": self.cfg.learning_rate,
            "max_depth": self.cfg.max_depth,
            "subsample": self.cfg.subsample,
            "colsample_bytree": self.cfg.colsample_bytree,
            "min_child_weight": self.cfg.min_child_weight,
            # Moderate reweighting: sqrt keeps rare positives visible without swamping calibration.
            "scale_pos_weight": math.sqrt(neg / max(pos, 1)),
            "seed": seed,
        }
        dtrain = xgb.QuantileDMatrix(train.X, label=train.y)
        rounds = self.cfg.n_estimators
        if val is not None and val.y.sum() > 0:
            dval = xgb.QuantileDMatrix(val.X, label=val.y, ref=dtrain)
            self.booster = xgb.train(
                params,
                dtrain,
                num_boost_round=rounds,
                evals=[(dval, "val")],
                early_stopping_rounds=self.cfg.early_stopping_rounds,
                verbose_eval=False,
            )
            self.best_iteration = int(self.booster.best_iteration)
        else:  # no validation positives (tiny test data): fixed number of rounds
            self.booster = xgb.train(params, dtrain, num_boost_round=min(rounds, 100))
            self.best_iteration = self.booster.num_boosted_rounds() - 1
        return {
            "best_iteration": self.best_iteration,
            "scale_pos_weight": params["scale_pos_weight"],
        }

    def _dmatrix(self, X: pd.DataFrame) -> xgb.DMatrix:
        return xgb.DMatrix(X)

    def predict_scores(self, X: pd.DataFrame) -> np.ndarray:
        return self.booster.predict(self._dmatrix(X), iteration_range=(0, self.best_iteration + 1))

    def explain(self, X: pd.DataFrame) -> pd.DataFrame:
        contribs = self.booster.predict(
            self._dmatrix(X), pred_contribs=True, iteration_range=(0, self.best_iteration + 1)
        )
        return pd.DataFrame(contribs[:, :-1], columns=X.columns, index=X.index)  # drop bias

    def save(self, folder: Path) -> None:
        super().save(folder)
        self.booster.save_model(folder / "model.json")

    def load(self, folder: Path, info: dict) -> None:
        self.booster = xgb.Booster()
        self.booster.load_model(folder / "model.json")
        self.best_iteration = int(info["best_iteration"])
