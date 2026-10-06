"""Neighbour sampling in plain PyTorch (replaces NeighborLoader, so no pyg-lib is needed).

Both edge directions count as neighbours, because every model reads both. Per hop, a node
with at most k incident transfers keeps all of them; a busier node gets k drawn uniformly
(duplicates removed). The result is one merged subgraph, the same shape NeighborLoader gives.
"""

from __future__ import annotations

import torch


class NeighborSampler:
    def __init__(self, edge_index: torch.Tensor, num_nodes: int):
        src, dst = edge_index[0], edge_index[1]
        e = torch.arange(src.numel())
        node = torch.cat([dst, src])
        order = torch.argsort(node, stable=True)
        self.eid = torch.cat([e, e])[order]
        self.other = torch.cat([src, dst])[order]
        counts = torch.bincount(node, minlength=num_nodes)
        self.rowptr = torch.zeros(num_nodes + 1, dtype=torch.long)
        self.rowptr[1:] = torch.cumsum(counts, 0)
        self.edge_index = edge_index

    def _pick(self, frontier: torch.Tensor, k: int, gen: torch.Generator) -> torch.Tensor:
        start = self.rowptr[frontier]
        deg = self.rowptr[frontier + 1] - start
        parts = []
        small = (deg > 0) & (deg <= k)
        if small.any():
            s, d = start[small], deg[small]
            offsets = torch.arange(int(d.sum())) - torch.repeat_interleave(
                torch.cumsum(d, 0) - d, d
            )
            parts.append(torch.repeat_interleave(s, d) + offsets)
        big = deg > k
        if big.any():
            s, d = start[big], deg[big]
            r = (torch.rand(s.numel(), k, generator=gen) * d.unsqueeze(1)).long()
            parts.append((s.unsqueeze(1) + r).flatten())
        if not parts:
            return torch.empty(0, dtype=torch.long)
        return torch.cat(parts).unique()

    def sample(self, seeds: torch.Tensor, fanouts, gen: torch.Generator):
        """Returns (sub_nodes, sub_edge_index, edge_ids, seed_positions)."""
        nodes = seeds.unique()
        frontier = nodes
        picked = []
        for k in fanouts:
            pos = self._pick(frontier, int(k), gen)
            if pos.numel() == 0:
                break
            picked.append(self.eid[pos])
            nbrs = self.other[pos].unique()
            frontier = nbrs[~torch.isin(nbrs, nodes)]
            nodes = torch.cat([nodes, frontier])
            if frontier.numel() == 0:
                break
        edge_ids = torch.cat(picked).unique() if picked else torch.empty(0, dtype=torch.long)
        sub_nodes = torch.sort(nodes).values
        sub_edge_index = torch.searchsorted(sub_nodes, self.edge_index[:, edge_ids])
        return sub_nodes, sub_edge_index, edge_ids, torch.searchsorted(sub_nodes, seeds)
