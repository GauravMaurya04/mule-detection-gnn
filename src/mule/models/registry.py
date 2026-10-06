"""Model names used on the command line."""

from __future__ import annotations

from mule.config import Config
from mule.models.base import Model

MODEL_NAMES = (
    "rule_ach",
    "rules",
    "xgb",
    "xgb_graph",
    "xgb_full",
    "xgb_pattern",
    "xgb_pattern_full",
)
# XGBoost variants differ only in the feature groups they see, so their gaps are clean ablations.
XGB_GROUPS = {
    "xgb": ("vol", "cp", "pt", "ch", "xb"),  # Tier 1 (v1 baseline)
    "xgb_graph": ("vol", "cp", "pt", "ch", "xb", "gr"),  # + graph features
    "xgb_full": ("vol", "cp", "pt", "ch", "xb", "gr", "en"),  # + entity features
    "xgb_pattern": ("vol", "cp", "pt", "ch", "xb", "gr", "pa"),  # graph + pattern features
    "xgb_pattern_full": ("vol", "cp", "pt", "ch", "xb", "gr", "en", "pa"),  # everything
}
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
    if name in XGB_GROUPS:
        from mule.models.xgb import XGBModel

        return XGBModel(cfg.models.xgb, name, XGB_GROUPS[name])
    raise ValueError(f"unknown model {name!r}; choose from {MODEL_NAMES}")
