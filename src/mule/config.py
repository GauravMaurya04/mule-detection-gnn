"""Load configs/base.yaml into validated settings; unknown keys are errors, not silent typos."""

from __future__ import annotations

import hashlib
import json
from datetime import date
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Paths(_Strict):
    raw_dir: Path
    interim_dir: Path
    processed_dir: Path
    reports_dir: Path


class SplitConfig(_Strict):
    shares: tuple[float, float, float] = (0.6, 0.2, 0.2)
    no_tail_end: date | None = None

    @model_validator(mode="after")
    def _shares_sum_to_one(self) -> SplitConfig:
        if abs(sum(self.shares) - 1) > 1e-6 or min(self.shares) <= 0:
            raise ValueError("split.shares must be three positive numbers summing to 1")
        return self


class LabelsConfig(_Strict):
    rule: Literal["completed", "within"] = "completed"


class FeaturesConfig(_Strict):
    lookback_hours: int = Field(72, ge=0)
    forward_hours: list[int] = [24, 72]


class Config(_Strict):
    dataset: str = "HI-Small"
    seed: int = 42
    paths: Paths
    split: SplitConfig = SplitConfig()
    labels: LabelsConfig = LabelsConfig()
    features: FeaturesConfig = FeaturesConfig()

    def section_hash(self, *sections: str) -> str:
        """Short hash of the named config sections, used to detect stale artifacts."""
        payload = {s: self.model_dump(mode="json")[s] for s in sections}
        return hashlib.sha1(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:8]


def load_config(path: str | Path = "configs/base.yaml") -> Config:
    with open(path) as f:
        return Config.model_validate(yaml.safe_load(f))
