"""GNN evidence transfers: a GNNExplainer-style edge mask, learned per account, plus checks.

For one account the explainer samples its neighbourhood (fixed seed, so the explanation covers a
reproducible subgraph), then learns a weight in (0, 1) for every transfer in it so that the
masked model still calls the account a mule while using as few transfers as possible (size and
entropy penalties). The `evidence_k` highest-weighted transfers are the evidence.

Checks per account (labels are used only to grade, never as input):
  fidelity     score with the evidence removed / kept alone, against removing k random transfers
  ground truth share of evidence transfers that are real laundering, against the subgraph's share
  stability    Jaccard overlap of the evidence from two independent sets of explainer seeds

The evidence mask is the average of `mask_seeds` masks learned from different starting points;
single masks depend heavily on their random start (Jaccard ~0.25 between two of them).
"""

from __future__ import annotations

import json

import numpy as np
import torch
import torch.nn.functional as F

from mule.config import Config, ExplainConfig
from mule.models.gnn.graph import build_graphs
from mule.models.gnn.model import AccountGNN
from mule.models.gnn.sampler import NeighborSampler
from mule.models.gnn.train import _device


def load_gnn(cfg: Config, model: str) -> tuple[AccountGNN, dict]:
    seed_dir = cfg.paths.runs_dir / "node" / model / "seed0"
    info = json.loads((seed_dir / "run.json").read_text())
    stats = json.loads((seed_dir / "graph_stats.json").read_text())
    in_dim = len(stats["feature_columns"]) + 1 if stats["node_features"] == "engineered" else 4
    p = info["params"]
    net = AccountGNN(
        info["kind"],
        in_dim,
        len(stats["edge_columns"]),
        hidden=int(p["hidden"]),
        layers=int(p["layers"]),
        dropout=float(p["dropout"]),
    )
    net.load_state_dict(torch.load(seed_dir / "model.pt", map_location="cpu", weights_only=True))
    return net.eval(), stats


def window_graph(cfg: Config, window: str, stats: dict):
    graphs, _ = build_graphs(cfg, stats["node_features"], stats=stats, windows=[window])
    return graphs[window]


def _jaccard(a, b) -> float:
    a, b = set(map(int, a)), set(map(int, b))
    return len(a & b) / len(a | b) if a | b else 1.0


def _learn_mask(net, x, ei, ea, target, ecfg: ExplainConfig, seed: int) -> torch.Tensor:
    torch.manual_seed(seed)
    logits = torch.nn.Parameter(torch.randn(ei.shape[1], device=x.device) * 0.1)
    opt = torch.optim.Adam([logits], lr=ecfg.mask_lr)
    one = torch.ones((), device=x.device)
    for _ in range(ecfg.mask_steps):
        m = torch.sigmoid(logits)
        out = net(x, ei, ea, edge_mask=m)[target]
        ent = -(m * torch.log(m + 1e-12) + (1 - m) * torch.log(1 - m + 1e-12)).mean()
        loss = F.binary_cross_entropy_with_logits(out, one) + ecfg.size_coef * m.sum()
        loss = loss + ecfg.entropy_coef * ent
        opt.zero_grad()
        loss.backward()
        opt.step()
    return torch.sigmoid(logits).detach()


def explain_account(
    net,
    g,
    sampler: NeighborSampler,
    account_id: int,
    ecfg: ExplainConfig,
    device,
    seed: int = 0,
    laundering: np.ndarray | None = None,
) -> dict:
    """Evidence transfers (window-graph edge positions) and the three checks for one account."""
    pos = int(np.searchsorted(g.nodes, account_id))
    gen = torch.Generator().manual_seed(seed)
    sub_nodes, sub_ei, eids, seed_pos = sampler.sample(
        torch.tensor([pos]), ecfg.subgraph_fanouts, gen
    )
    x, ei = g.x[sub_nodes].to(device), sub_ei.to(device)
    ea, target = g.edge_attr[eids].to(device), int(seed_pos[0])
    n_edges = ei.shape[1]
    out = {"subgraph_edges": int(n_edges), "evidence_edges": [], "p_full": np.nan}
    if n_edges == 0:
        return out

    def prob(mask=None) -> float:
        with torch.no_grad():
            return float(torch.sigmoid(net(x, ei, ea, edge_mask=mask)[target]))

    k = min(ecfg.evidence_k, n_edges)
    n = max(1, ecfg.mask_seeds)
    seeds_a = [seed * 2 * n + j for j in range(n)]  # two disjoint seed sets per account
    seeds_b = [seed * 2 * n + n + j for j in range(n)]
    masks = [
        torch.stack([_learn_mask(net, x, ei, ea, target, ecfg, s) for s in seeds]).mean(dim=0)
        for seeds in (seeds_a, seeds_b)
    ]
    top = [torch.topk(m, k).indices.cpu() for m in masks]
    keep_only = torch.zeros(n_edges, device=device)
    keep_only[top[0]] = 1.0
    without = torch.ones(n_edges, device=device)
    without[top[0]] = 0.0
    rng = np.random.default_rng(seed)
    rand_drop = []
    for _ in range(5):
        r = torch.ones(n_edges, device=device)
        r[torch.from_numpy(rng.choice(n_edges, k, replace=False))] = 0.0
        rand_drop.append(prob(r))
    evidence = eids[top[0]].numpy()
    out.update(
        {
            "evidence_edges": evidence.tolist(),
            "evidence_weights": masks[0][top[0]].cpu().numpy().round(4).tolist(),
            "p_full": prob(),
            "p_without_evidence": prob(without),
            "p_evidence_only": prob(keep_only),
            "p_without_random": float(np.mean(rand_drop)),
            "stability_jaccard": _jaccard(eids[top[0]].numpy(), eids[top[1]].numpy()),
        }
    )
    if laundering is not None:
        sub_lab = laundering[eids.numpy()]
        out["evidence_laundering_share"] = float(laundering[evidence].mean())
        out["subgraph_laundering_share"] = float(sub_lab.mean())
    return out


def gnn_evidence(cfg: Config, model: str, window: str, account_ids, tx_laundering) -> tuple:
    """Run the explainer for each account. `tx_laundering`: Series tx_id -> 0/1 (grading only)."""
    net, stats = load_gnn(cfg, model)
    device = _device(cfg.gnn.device)
    net = net.to(device)
    g = window_graph(cfg, window, stats)
    sampler = NeighborSampler(g.edge_index, len(g.nodes))
    laundering = tx_laundering.reindex(g.edge_tx_id).to_numpy(dtype=float)
    results = {}
    for i, aid in enumerate(account_ids):
        results[int(aid)] = explain_account(
            net, g, sampler, int(aid), cfg.explain, device, seed=cfg.seed + i, laundering=laundering
        )
        if (i + 1) % 25 == 0:
            print(f"  explained {i + 1}/{len(account_ids)} accounts", flush=True)
    return results, g.edge_tx_id
