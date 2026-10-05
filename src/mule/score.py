"""`mule score`: score one account (with its strongest SHAP evidence) or list the top accounts."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from mule.config import Config
from mule.features.account import GROUPS
from mule.io import read_parquet
from mule.models.registry import make_model


def _load(cfg: Config, model_name: str):
    seed_dir = cfg.paths.runs_dir / "node" / model_name / "seed0"
    if not seed_dir.exists():
        raise FileNotFoundError(
            f"{seed_dir} not found: run `mule train --task node --model {model_name}`"
        )
    info = json.loads((seed_dir / "run.json").read_text())
    model = make_model(model_name, "node", cfg)
    model.load(seed_dir, info)
    return model, info


def _window_features(cfg: Config, window: str, columns: list[str]) -> pd.DataFrame:
    feats = read_parquet(cfg.paths.processed_dir / "account_features.parquet")
    feats = feats[feats["window"].astype(str) == window]
    if feats.empty:
        raise ValueError(f"no features for window {window!r}")
    return feats.set_index("account_id")[columns].astype("float32")


def score_account(
    cfg: Config, account: str, window: str = "test", model_name: str = "xgb", top_n: int = 5
) -> str:
    model, info = _load(cfg, model_name)
    columns = info["feature_columns"]
    bank_txt, _, acct = account.partition(":")
    index = read_parquet(cfg.paths.interim_dir / "account_index.parquet")
    match = index[(index["bank"] == int(bank_txt)) & (index["account"] == acct)]
    if match.empty:
        raise ValueError(f"account {account} not found (format: BANK:ACCOUNT, e.g. 10:8000EBD30)")
    account_id = int(match["account_id"].iloc[0])
    feats = _window_features(cfg, window, columns)
    if account_id not in feats.index:
        return f"{account} had no transactions in window {window}."

    all_scores = pd.Series(model.predict_scores(feats), index=feats.index)
    score = float(all_scores.loc[account_id])
    rank = int((all_scores > score).sum()) + 1
    row = feats.loc[[account_id]]
    contrib = model.explain(row).iloc[0]
    top = contrib.reindex(contrib.abs().sort_values(ascending=False).index).head(top_n)
    groups = {
        GROUPS[p]: float(contrib[[c for c in columns if c.startswith(p + "_")]].sum())
        for p in GROUPS
    }

    lines = [
        f"Account {account} (id {account_id}), window {window}, model {model_name}",
        f"Score {score:.4f}  rank {rank:,} of {len(all_scores):,}  "
        f"(top {100 * rank / len(all_scores):.2f}%)",
        "",
        "Strongest evidence (contribution to the score, then the feature value):",
    ]
    for name, value in top.items():
        direction = "raises" if value > 0 else "lowers"
        lines.append(f"  {direction:6} {value:+.3f}  {name} = {row.iloc[0][name]:.4g}")
    lines += ["", "By feature group:"]
    lines += [f"  {g:22} {v:+.3f}" for g, v in sorted(groups.items(), key=lambda kv: -abs(kv[1]))]
    return "\n".join(lines)


def top_accounts(cfg: Config, n: int = 10, window: str = "test", model_name: str = "xgb") -> str:
    model, info = _load(cfg, model_name)
    feats = _window_features(cfg, window, info["feature_columns"])
    scores = pd.Series(model.predict_scores(feats), index=feats.index).nlargest(n)
    index = read_parquet(cfg.paths.interim_dir / "account_index.parquet").set_index("account_id")
    rows = [f"Top {n} accounts in window {window} ({model_name}):"]
    for rank, (aid, s) in enumerate(scores.items(), start=1):
        bank, acct = index.loc[aid, "bank"], index.loc[aid, "account"]
        rows.append(f"  {rank:3}. {bank}:{acct}  score {s:.4f}")
    return "\n".join(rows) + ("" if np.isfinite(scores).all() else "\n(some scores are NaN)")
