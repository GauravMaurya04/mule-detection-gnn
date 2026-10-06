"""Out-of-fold GNN scores for stacking (account task).

For each train snapshot, a GraphSAGE is trained on the other two snapshots and scores the held-out
one, so every training row gets a score from a model that never saw it (in-sample GNN scores are
overfit, training loss ~0.002). Validation and test rows get the average of the three fold
models, so train and evaluation scores come from the same kind of model. Hyperparameters are
the GNN's own tuned ones. The result is one extra XGBoost feature, st_gnn_logit.
Note: a fold may train on later snapshots and score an earlier one; this only builds a training
feature and never touches validation or test labels.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from mule.config import Config
from mule.labels.split import EVAL_WINDOWS
from mule.models.gnn.graph import build_graphs
from mule.models.gnn.train import _device, _params, fit, run_name, score_graph

OOF_FILE = "gnn_oof_node.parquet"


def run_gnn_oof(cfg: Config, kind: str = "sage", seed: int = 0) -> dict:
    gcfg = cfg.gnn
    name = run_name(kind, gcfg.node_features)
    tuned = cfg.paths.runs_dir / "node" / "_tuning" / f"{name}.json"
    if not tuned.exists():
        raise FileNotFoundError(f"{tuned} not found: run `mule gnn-tune -m {kind}` first")
    params = _params(gcfg, json.loads(tuned.read_text())["best_params"])
    graphs, _ = build_graphs(cfg, gcfg.node_features)
    snaps = [w for w in graphs if w.startswith("train_s")]
    score_device = _device(gcfg.score_device)
    frames, eval_scores, folds = [], {w: [] for w in EVAL_WINDOWS}, []
    for held in snaps:
        train_on = [s for s in snaps if s != held]
        print(f"fold: train on {train_on}, score {held}", flush=True)
        model, info = fit(kind, graphs, train_on, gcfg, params, seed)
        g = graphs[held]
        frames.append(
            pd.DataFrame(
                {
                    "window": held,
                    "account_id": g.meta["account_id"].to_numpy(),
                    "gnn_score": score_graph(model, g, score_device, gcfg.edge_chunk),
                }
            )
        )
        for w in EVAL_WINDOWS:
            eval_scores[w].append(score_graph(model, graphs[w], score_device, gcfg.edge_chunk))
        folds.append(
            {"held_out": held, "val_pr_auc": info["val_pr_auc"], "best_epoch": info["best_epoch"]}
        )
    for w in EVAL_WINDOWS:
        frames.append(
            pd.DataFrame(
                {
                    "window": w,
                    "account_id": graphs[w].meta["account_id"].to_numpy(),
                    "gnn_score": np.mean(eval_scores[w], axis=0),
                }
            )
        )
    out = pd.concat(frames, ignore_index=True)
    out.to_parquet(cfg.paths.processed_dir / OOF_FILE, index=False)
    meta = {
        "kind": kind,
        "params": params,
        "folds": folds,
        "features_hash": cfg.section_hash("dataset", "split", "labels", "features"),
    }
    (cfg.paths.processed_dir / "gnn_oof_node.meta.json").write_text(json.dumps(meta, indent=2))
    return meta
