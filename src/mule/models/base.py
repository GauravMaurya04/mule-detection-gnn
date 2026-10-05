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
