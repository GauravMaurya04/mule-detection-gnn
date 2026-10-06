"""Train, tune and time the account-task GNNs.

Each epoch draws every positive row plus `neg_ratio` random negatives per positive from each
train snapshot, samples a 2-hop neighbourhood per batch, and trains with plain BCE. Validation
PR-AUC is measured every `eval_every` epochs by scoring the whole val graph in one pass
(chunked, GPU first, CPU if the GPU runs out of memory); the best epoch is kept.
Outputs match the tabular models (runs/node/<name>/seed<k>/...), so `mule evaluate` and the
results tables need no special cases.
"""

from __future__ import annotations

import copy
import json
import shutil
import time

import numpy as np
import torch
import torch.nn.functional as F

from mule.config import Config, GNNConfig
from mule.eval.metrics import best_f1_threshold, pr_auc
from mule.labels.split import EVAL_WINDOWS
from mule.models.gnn.graph import build_graphs
from mule.models.gnn.model import AccountGNN
from mule.models.gnn.sampler import NeighborSampler
from mule.tracking import log_run

TUNABLE = ("hidden", "layers", "dropout", "lr", "weight_decay", "neg_ratio")


def run_name(kind: str, node_features: str) -> str:
    return f"gnn_{kind}" + ("" if node_features == "engineered" else f"_{node_features}")


def _device(name: str) -> torch.device:
    if name == "cuda" and not torch.cuda.is_available():
        print("WARNING: CUDA not available, using CPU")
        return torch.device("cpu")
    return torch.device(name)


def score_graph(model: AccountGNN, g, device: torch.device, chunk: int) -> np.ndarray:
    """Probabilities for the graph's labelled accounts, from one full-graph pass."""
    model.eval()

    def run(m, dev):
        m = m.to(dev)
        with torch.no_grad():
            logits = m(g.x.to(dev), g.edge_index.to(dev), g.edge_attr.to(dev), chunk=chunk)
            return torch.sigmoid(logits[g.label_nodes.to(dev)]).cpu().numpy()

    try:
        return run(model, device)
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        print("WARNING: GPU out of memory while scoring; falling back to CPU for this pass")
        return run(copy.deepcopy(model).cpu(), torch.device("cpu"))


def fanouts_for(gcfg: GNNConfig, layers: int) -> list[int]:
    """One sampling hop per layer, so training sees as deep a neighbourhood as scoring does.
    Extra layers reuse half the last fanout (at least 5)."""
    base = list(gcfg.fanouts)[:layers]
    return base + [max(5, gcfg.fanouts[-1] // 2)] * (layers - len(base))


def _epoch(model, opt, train_items, gcfg: GNNConfig, neg_ratio: int, fanouts, device, gen) -> float:
    model.train()
    losses = []
    for g, sampler in train_items:
        pos = g.label_nodes[g.y == 1]
        neg_pool = g.label_nodes[g.y == 0]
        if pos.numel() == 0:
            continue
        n_neg = min(neg_pool.numel(), max(neg_ratio * pos.numel(), gcfg.min_negatives))
        neg = neg_pool[torch.randperm(neg_pool.numel(), generator=gen)[:n_neg]]
        seeds = torch.cat([pos, neg])
        target = torch.cat([torch.ones(pos.numel()), torch.zeros(neg.numel())])
        order = torch.randperm(seeds.numel(), generator=gen)
        seeds, target = seeds[order], target[order]
        for s in range(0, seeds.numel(), gcfg.batch_size):
            batch, y = seeds[s : s + gcfg.batch_size], target[s : s + gcfg.batch_size]
            sub_nodes, sub_ei, edge_ids, seed_pos = sampler.sample(batch, fanouts, gen)
            logits = model(
                g.x[sub_nodes].to(device), sub_ei.to(device), g.edge_attr[edge_ids].to(device)
            )[seed_pos.to(device)]
            loss = F.binary_cross_entropy_with_logits(logits, y.to(device))
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(float(loss))
    return float(np.mean(losses)) if losses else float("nan")


def fit(kind: str, graphs: dict, snapshots: list[str], gcfg: GNNConfig, params: dict, seed: int):
    """Train one model; returns (best model, info dict)."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    gen = torch.Generator().manual_seed(seed)
    device = _device(gcfg.device)
    score_device = _device(gcfg.score_device)
    any_graph = graphs["val"]
    model = AccountGNN(
        kind,
        any_graph.x.shape[1],
        any_graph.edge_attr.shape[1],
        hidden=int(params["hidden"]),
        layers=int(params["layers"]),
        dropout=float(params["dropout"]),
    ).to(device)
    opt = torch.optim.AdamW(
        model.parameters(), lr=float(params["lr"]), weight_decay=float(params["weight_decay"])
    )
    train_items = [
        (graphs[s], NeighborSampler(graphs[s].edge_index, len(graphs[s].nodes))) for s in snapshots
    ]
    val = graphs["val"]
    fanouts = fanouts_for(gcfg, int(params["layers"]))
    best, best_state, best_epoch, waited, history = -1.0, None, 0, 0, []
    t0 = time.perf_counter()
    for epoch in range(1, gcfg.max_epochs + 1):
        loss = _epoch(model, opt, train_items, gcfg, int(params["neg_ratio"]), fanouts, device, gen)
        if epoch % gcfg.eval_every and epoch != gcfg.max_epochs:
            continue
        val_auc = pr_auc(val.y.numpy(), score_graph(model, val, score_device, gcfg.edge_chunk))
        model.to(device)
        history.append({"epoch": epoch, "loss": loss, "val_pr_auc": val_auc})
        print(f"  epoch {epoch:3d}  loss {loss:.4f}  val PR-AUC {val_auc:.4f}", flush=True)
        if val_auc > best:
            best, best_epoch, waited = val_auc, epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            waited += 1
            if waited >= gcfg.patience:
                break
    if best_state is not None:  # None only if validation had no positives (tiny test data)
        model.load_state_dict(best_state)
    model.to(device)
    return model, {
        "best_epoch": best_epoch,
        "val_pr_auc": best,
        "history": history,
        "train_seconds": time.perf_counter() - t0,
    }


def _params(gcfg: GNNConfig, override: dict | None) -> dict:
    p = {k: getattr(gcfg, k) for k in TUNABLE}
    p.update(override or {})
    return p


def run_gnn_train(
    cfg: Config,
    kind: str,
    n_seeds: int = 3,
    params: dict | None = None,
    graphs=None,
    stats=None,
    force: bool = False,
) -> dict:
    gcfg = cfg.gnn
    name = run_name(kind, gcfg.node_features)
    out_dir = cfg.paths.runs_dir / "node" / name
    if (out_dir / "metrics_final.json").exists() and not force:
        raise FileExistsError(f"{out_dir} has recorded --final results; pass --force to retrain")
    tuned = cfg.paths.runs_dir / "node" / "_tuning" / f"{name}.json"
    if params is None and tuned.exists():
        params = json.loads(tuned.read_text())["best_params"]
        print(f"using tuned parameters from {tuned}: {params}")
    params = _params(gcfg, params)
    if graphs is None:
        graphs, stats = build_graphs(cfg, gcfg.node_features)
    snapshots = [w for w in graphs if w.startswith("train_s")]
    if out_dir.exists():
        shutil.rmtree(out_dir)
    score_device = _device(gcfg.score_device)
    summary = {"task": "node", "model": name, "params": params, "seeds": []}
    for seed in range(n_seeds):
        print(f"{name} seed {seed}", flush=True)
        model, info = fit(kind, graphs, snapshots, gcfg, params, seed)
        seed_dir = out_dir / f"seed{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        scores = {}
        for window in EVAL_WINDOWS:
            g = graphs[window]
            t1 = time.perf_counter()
            scores[window] = score_graph(model, g, score_device, gcfg.edge_chunk)
            if window == "val":
                per_1k = 1000 * (time.perf_counter() - t1) / max(len(scores[window]), 1)
            g.meta.assign(score=scores[window]).to_parquet(
                seed_dir / f"scores_{window}.parquet", index=False
            )
        torch.save(model.state_dict(), seed_dir / "model.pt")
        (seed_dir / "graph_stats.json").write_text(json.dumps(stats))
        run_info = {
            "seed": seed,
            "kind": kind,
            "params": params,
            "node_features": gcfg.node_features,
            "threshold": best_f1_threshold(graphs["val"].y.numpy(), scores["val"]),
            "train_seconds": round(info["train_seconds"], 1),
            "score_seconds_per_1k": round(per_1k, 5),
            "probabilistic": True,
            "best_epoch": info["best_epoch"],
            "history": info["history"],
            "config_hash": cfg.section_hash("dataset", "split", "labels", "features", "gnn"),
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
            f"train-node-{name}-seed{seed}",
            params={"model": name, "seed": seed, **params},
            metrics={"val.pr_auc": info["val_pr_auc"], "train_seconds": info["train_seconds"]},
            tags={"stage": "train"},
        )
    (out_dir / "train_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def run_gnn_tune(cfg: Config, kind: str, n_seeds: int = 3, force: bool = False) -> dict:
    """Same grid for every variant (seed 0, best validation PR-AUC), then n_seeds of the winner."""
    gcfg = cfg.gnn
    name = run_name(kind, gcfg.node_features)
    graphs, stats = build_graphs(cfg, gcfg.node_features)
    snapshots = [w for w in graphs if w.startswith("train_s")]
    trials = []
    for i, override in enumerate(gcfg.grid):
        params = _params(gcfg, override)
        print(f"{name} config {i + 1}/{len(gcfg.grid)}: {params}", flush=True)
        _, info = fit(kind, graphs, snapshots, gcfg, params, seed=0)
        trials.append(
            {
                "params": params,
                "val_pr_auc": info["val_pr_auc"],
                "best_epoch": info["best_epoch"],
                "seconds": round(info["train_seconds"], 1),
            }
        )
    best = max(trials, key=lambda t: t["val_pr_auc"])
    out = cfg.paths.runs_dir / "node" / "_tuning"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{name}.json").write_text(
        json.dumps({"best_params": best["params"], "trials": trials}, indent=2)
    )
    summary = run_gnn_train(cfg, kind, n_seeds, best["params"], graphs, stats, force)
    return {"trials": trials, "best": best, "final": summary}


def run_gnn_time(cfg: Config, kind: str) -> dict:
    """Time one training epoch and one validation scoring pass, and estimate a full run."""
    gcfg = cfg.gnn
    t0 = time.perf_counter()
    graphs, _ = build_graphs(cfg, gcfg.node_features)
    build = time.perf_counter() - t0
    snapshots = [w for w in graphs if w.startswith("train_s")]
    timing = gcfg.model_copy(update={"max_epochs": 1, "eval_every": 1})
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    _, info = fit(kind, graphs, snapshots, timing, _params(gcfg, None), seed=0)
    evals = gcfg.max_epochs // gcfg.eval_every
    out = {
        "graph_build_seconds": round(build, 1),
        "epoch_plus_val_scoring_seconds": round(info["train_seconds"], 1),
        "worst_case_run_minutes": round(gcfg.max_epochs * info["train_seconds"] / 60, 1),
        "evaluations_per_run": evals,
        "edges_per_window": {w: int(g.edge_index.shape[1]) for w, g in graphs.items()},
    }
    if torch.cuda.is_available():
        out["peak_gpu_memory_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
    return out
