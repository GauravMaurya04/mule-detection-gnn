"""The shared model interface. The evaluator and scorer only ever call these methods."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import pandas as pd


class Model(ABC):
    name: str = "base"
    deterministic: bool = False  # deterministic models need only one seed
    probabilistic: bool = False  # scores are probabilities, so a 0.5 cut is meaningful
    groups: tuple[str, ...] | None = None  # feature groups used; None = every column

    def select_columns(self, columns) -> list[str]:
        """Columns this model sees; transaction columns are e_*, src_<group>_*, dst_<group>_*."""
        if self.groups is None:
            return list(columns)
        keep = []
        for c in columns:
            base = c[4:] if c.startswith(("src_", "dst_")) else c
            if c.startswith("e_") or any(base.startswith(g + "_") for g in self.groups):
                keep.append(c)
        return keep

    @abstractmethod
    def fit(self, train, val, seed: int) -> dict:
        """Train on `train` (TaskData), use `val` only for early stopping. Returns run info."""

    @abstractmethod
    def predict_scores(self, X: pd.DataFrame) -> np.ndarray:
        """Higher = more suspicious. Only the ranking matters to most metrics."""

    @abstractmethod
    def explain(self, X: pd.DataFrame) -> pd.DataFrame:
        """Per-row, per-feature contributions to the score (same columns as X)."""

    def save(self, folder: Path) -> None:
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "model_params.json").write_text(json.dumps(self.params(), indent=2))

    def load(self, folder: Path, info: dict) -> None:  # noqa: B027 - optional for rules
        pass

    def params(self) -> dict:
        return {}
