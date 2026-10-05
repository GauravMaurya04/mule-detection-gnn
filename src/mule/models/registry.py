"""Model names used on the command line."""

from __future__ import annotations

from mule.config import Config
from mule.models.base import Model

MODEL_NAMES = ("rule_ach", "rules", "xgb")
TASKS = ("node", "edge")


def make_model(name: str, task: str, cfg: Config) -> Model:
    if task not in TASKS:
        raise ValueError(f"task must be one of {TASKS}")
    if name == "rule_ach":
        from mule.models.rules import AchShareRule

        return AchShareRule(task)
    if name == "rules":
        from mule.models.rules import AnalystRules

        return AnalystRules(task, cfg.models.rules)
    if name == "xgb":
        from mule.models.xgb import XGBModel

        return XGBModel(cfg.models.xgb)
    raise ValueError(f"unknown model {name!r}; choose from {MODEL_NAMES}")
