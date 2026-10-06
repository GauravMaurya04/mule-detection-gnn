"""Transaction-task GNN: the account encoder plus an edge head.

score(transaction) = MLP(sender embedding, receiver embedding, the transaction's own features).
Training draws every laundering transaction in each train snapshot plus `edge_neg_ratio`
random normal ones per positive, embeds both endpoints from a sampled neighbourhood, and
trains with BCE. Scoring embeds every account of a window once, then scores all of the
window's transactions from those embeddings in chunks: one pass, not a subgraph per edge.
Hyperparameters transfer from the account-task tuning of the same encoder (no second grid).
"""

from __future__ import annotations

import copy
import json
import shutil
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn

from mule.config import Config
from mule.eval.metrics import best_f1_threshold, pr_auc
from mule.features.patterns import EDGE_COLUMNS as PATTERN_EDGE_COLUMNS
from mule.io import TX_SCHEMA, read_parquet
from mule.labels.split import EVAL_WINDOWS, load_manifest, window_bounds
from mule.models.gnn.graph import CONTINUOUS_EDGE, build_graphs, edge_table
from mule.models.gnn.model import AccountGNN, _mlp
from mule.models.gnn.sampler import NeighborSampler
from mule.models.gnn.train import _device, _params, fanouts_for, run_name
from mule.tracking import log_run

ROW_CHUNK = 1_000_000


@dataclass
class EdgeRows:
    tx_id: np.ndarray
    src: torch.Tensor  # node positions in the window graph
    dst: torch.Tensor
    attr: torch.Tensor  # standardised edge features + is_self
    y: torch.Tensor


def _pattern_block(patterns: pd.DataFrame, tx_ids: np.ndarray) -> np.ndarray:
    """The 5 per-transaction pattern features (log scale, 0 when missing) + 5 missing flags."""
    values = patterns.reindex(tx_ids).to_numpy(dtype=np.float32)
    missing = np.isnan(values).astype(np.float32)
    return np.hstack([np.nan_to_num(values, nan=0.0), missing])


def build_edge_rows(cfg: Config, graphs: dict, stats: dict) -> dict[str, EdgeRows]:
    """Every transaction of each window, mapped onto that window's graph. Row features are the
    graph's edge features, an is_self flag and the per-transaction pattern features, so the GNN
    sees the same transaction-level information as xgb_pattern."""
    manifest = load_manifest(cfg.paths.processed_dir / "split_manifest.json")
    pattern_path = cfg.paths.processed_dir / "edge_patterns.parquet"
    if not pattern_path.exists():
        raise FileNotFoundError(f"{pattern_path} not found: rerun `mule features`")
    edge_pa = read_parquet(pattern_path)
    edge_pa["window"] = edge_pa["window"].astype(str)
    pa_by_window = {
        w: g.set_index("tx_id")[PATTERN_EDGE_COLUMNS] for w, g in edge_pa.groupby("window")
    }
    del edge_pa
    tx = read_parquet(cfg.paths.interim_dir / "transactions.parquet", TX_SCHEMA)
    feats = edge_table(tx)
    mean = np.asarray(stats["edge_mean"], dtype=np.float32)
    std = np.asarray(stats["edge_std"], dtype=np.float32)
    feats[:, CONTINUOUS_EDGE] = (feats[:, CONTINUOUS_EDGE] - mean) / std
    is_self = tx["is_self"].to_numpy(np.float32)[:, None]
    src_all, dst_all = tx["src_id"].to_numpy(), tx["dst_id"].to_numpy()
    out = {}
    for w, g in graphs.items():
        start, end = window_bounds(manifest, w)
        m = ((tx["ts"] >= start) & (tx["ts"] < end)).to_numpy()
        src = np.searchsorted(g.nodes, src_all[m])
        dst = np.searchsorted(g.nodes, dst_all[m])
        src_c, dst_c = np.clip(src, 0, len(g.nodes) - 1), np.clip(dst, 0, len(g.nodes) - 1)
        if not ((g.nodes[src_c] == src_all[m]).all() and (g.nodes[dst_c] == dst_all[m]).all()):
            raise RuntimeError(f"window {w}: a transaction endpoint is missing from the graph")
        tx_ids = tx["tx_id"].to_numpy()[m]
        attr = np.hstack([feats[m], is_self[m], _pattern_block(pa_by_window[w], tx_ids)])
        out[w] = EdgeRows(
            tx_id=tx_ids,
            src=torch.from_numpy(src.astype(np.int64)),
            dst=torch.from_numpy(dst.astype(np.int64)),
            attr=torch.from_numpy(np.ascontiguousarray(attr, dtype=np.float32)),
            y=torch.from_numpy(tx["is_laundering"].to_numpy()[m].astype(np.float32)),
        )
    return out


class EdgeGNN(nn.Module):
    def __init__(self, kind, in_dim, edge_dim, row_dim, hidden, layers, dropout):
        super().__init__()
        self.encoder = AccountGNN(kind, in_dim, edge_dim, hidden, layers, dropout)
        self.head = _mlp(2 * hidden + row_dim, hidden, 1, dropout)

    def score_rows(self, h, src, dst, attr):
        return self.head(torch.cat([h[src], h[dst], attr], dim=-1)).squeeze(-1)


def score_window(model: EdgeGNN, g, rows: EdgeRows, device, chunk: int) -> np.ndarray:
    model.eval()

    def run(m, dev):
        m = m.to(dev)
        with torch.no_grad():
            h = m.encoder.embed(g.x.to(dev), g.edge_index.to(dev), g.edge_attr.to(dev), chunk=chunk)
            parts = []
            for s in range(0, len(rows.tx_id), ROW_CHUNK):
                sl = slice(s, s + ROW_CHUNK)
                logits = m.score_rows(
                    h, rows.src[sl].to(dev), rows.dst[sl].to(dev), rows.attr[sl].to(dev)
                )
                parts.append(torch.sigmoid(logits).cpu())
            return torch.cat(parts).numpy() if parts else np.empty(0, np.float32)

    try:
        return run(model, device)
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        print("WARNING: GPU out of memory while scoring; falling back to CPU for this pass")
        return run(copy.deepcopy(model).cpu(), torch.device("cpu"))


def _edge_epoch(model, opt, items, gcfg, fanouts, device, gen) -> float:
    model.train()
    losses = []
    for g, rows, sampler in items:
        pos = torch.nonzero(rows.y == 1).squeeze(1)
        neg_pool = torch.nonzero(rows.y == 0).squeeze(1)
        if pos.numel() == 0:
            continue
        n_neg = min(neg_pool.numel(), gcfg.edge_neg_ratio * pos.numel())
        neg = neg_pool[torch.randperm(neg_pool.numel(), generator=gen)[:n_neg]]
        idx = torch.cat([pos, neg])
        target = torch.cat([torch.ones(pos.numel()), torch.zeros(neg.numel())])
        order = torch.randperm(idx.numel(), generator=gen)
        idx, target = idx[order], target[order]
        for s in range(0, idx.numel(), gcfg.edge_batch_size):
            batch, y = idx[s : s + gcfg.edge_batch_size], target[s : s + gcfg.edge_batch_size]
            src, dst = rows.src[batch], rows.dst[batch]
            seeds = torch.unique(torch.cat([src, dst]))
            sub_nodes, sub_ei, edge_ids, _ = sampler.sample(seeds, fanouts, gen)
            h = model.encoder.embed(
                g.x[sub_nodes].to(device), sub_ei.to(device), g.edge_attr[edge_ids].to(device)
            )
            sp = torch.searchsorted(sub_nodes, src).to(device)
            dp = torch.searchsorted(sub_nodes, dst).to(device)
            logits = model.score_rows(h, sp, dp, rows.attr[batch].to(device))
            loss = F.binary_cross_entropy_with_logits(logits, y.to(device))
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(float(loss))
    return float(np.mean(losses)) if losses else float("nan")


def fit_edge(kind, graphs, rows, snapshots, gcfg, params, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    gen = torch.Generator().manual_seed(seed)
    device, score_device = _device(gcfg.device), _device(gcfg.score_device)
    val_g = graphs["val"]
    model = EdgeGNN(
        kind,
        val_g.x.shape[1],
        val_g.edge_attr.shape[1],
        rows["val"].attr.shape[1],
        int(params["hidden"]),
        int(params["layers"]),
        float(params["dropout"]),
    ).to(device)
    opt = torch.optim.AdamW(
        model.parameters(), lr=float(params["lr"]), weight_decay=float(params["weight_decay"])
    )
    items = [
        (graphs[s], rows[s], NeighborSampler(graphs[s].edge_index, len(graphs[s].nodes)))
        for s in snapshots
    ]
    fanouts = fanouts_for(gcfg, int(params["layers"]))
    best, best_state, best_epoch, waited, history = -1.0, None, 0, 0, []
    t0 = time.perf_counter()
    for epoch in range(1, gcfg.edge_max_epochs + 1):
        loss = _edge_epoch(model, opt, items, gcfg, fanouts, device, gen)
        if epoch % gcfg.eval_every and epoch != gcfg.edge_max_epochs:
            continue
        val_auc = pr_auc(
            rows["val"].y.numpy(),
            score_window(model, val_g, rows["val"], score_device, gcfg.edge_chunk),
        )
        model.to(device)
        history.append({"epoch": epoch, "loss": loss, "val_pr_auc": val_auc})
        print(f"  epoch {epoch:3d}  loss {loss:.4f}  val PR-AUC {val_auc:.4f}", flush=True)
        if val_auc > best:
            best, best_epoch, waited = val_auc, epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            waited += 1
            if waited >= gcfg.edge_patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.to(device)
    return model, {
        "best_epoch": best_epoch,
        "val_pr_auc": best,
        "history": history,
        "train_seconds": time.perf_counter() - t0,
    }


def run_gnn_edge_train(cfg: Config, kind: str, n_seeds: int = 3, force: bool = False) -> dict:
    gcfg = cfg.gnn
    name = run_name(kind, gcfg.node_features)
    out_dir = cfg.paths.runs_dir / "edge" / name
    if (out_dir / "metrics_final.json").exists() and not force:
        raise FileExistsError(f"{out_dir} has recorded --final results; pass --force to retrain")
    tuned = cfg.paths.runs_dir / "node" / "_tuning" / f"{name}.json"
    override = json.loads(tuned.read_text())["best_params"] if tuned.exists() else None
    params = _params(gcfg, override)
    print(
        f"{name} (transaction task) parameters: {params}"
        + ("" if tuned.exists() else " (config defaults: no account-task tuning found)")
    )
    graphs, stats = build_graphs(cfg, gcfg.node_features)
    rows = build_edge_rows(cfg, graphs, stats)
    snapshots = [w for w in graphs if w.startswith("train_s")]
    if out_dir.exists():
        shutil.rmtree(out_dir)
    score_device = _device(gcfg.score_device)
    summary = {"task": "edge", "model": name, "params": params, "seeds": []}
    for seed in range(n_seeds):
        print(f"{name} edge seed {seed}", flush=True)
        model, info = fit_edge(kind, graphs, rows, snapshots, gcfg, params, seed)
        seed_dir = out_dir / f"seed{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        scores = {}
        for window in EVAL_WINDOWS:
            t1 = time.perf_counter()
            scores[window] = score_window(
                model, graphs[window], rows[window], score_device, gcfg.edge_chunk
            )
            if window == "val":
                per_1k = 1000 * (time.perf_counter() - t1) / max(len(scores[window]), 1)
            pd.DataFrame(
                {
                    "tx_id": rows[window].tx_id,
                    "is_laundering": rows[window].y.numpy().astype(np.int8),
                    "score": scores[window],
                }
            ).to_parquet(seed_dir / f"scores_{window}.parquet", index=False)
        torch.save(model.state_dict(), seed_dir / "model.pt")
        (seed_dir / "graph_stats.json").write_text(json.dumps(stats))
        run_info = {
            "seed": seed,
            "kind": kind,
            "params": params,
            "node_features": gcfg.node_features,
            "threshold": best_f1_threshold(rows["val"].y.numpy(), scores["val"]),
            "train_seconds": round(info["train_seconds"], 1),
            "score_seconds_per_1k": round(per_1k, 5),
            "probabilistic": True,
            "best_epoch": info["best_epoch"],
            "history": info["history"],
        }
        (seed_dir / "run.json").write_text(json.dumps(run_info, indent=2))
        summary["seeds"].append(
            {
                "seed": seed,
                "val_pr_auc": info["val_pr_auc"],
                "best_epoch": info["best_epoch"],
                "train_seconds": round(info["train_seconds"], 1),
            }
        )
        log_run(
            cfg,
            f"train-edge-{name}-seed{seed}",
            params={"model": name, "seed": seed, **params},
            metrics={"val.pr_auc": info["val_pr_auc"]},
            tags={"stage": "train"},
        )
    (out_dir / "train_summary.json").write_text(json.dumps(summary, indent=2))
    return summary
