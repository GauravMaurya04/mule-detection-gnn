"""GraphSAGE, GINE and GINE + reverse message passing, in plain PyTorch.

- sage, gine: every transfer is passed both ways with the same weights, so the model cannot
  tell money in from money out (the "direction-blind" baselines).
- Every layer averages its messages rather than summing them. Training sees at most 15
  sampled neighbours per account while scoring sees all of them, so a sum would make a busy
  account's messages tens of times larger at scoring time than anything seen in training.
  Neighbour counts are already node features, so averaging loses no information.
- gine_rmp: separate weights for "what my senders look like" and "what my receivers look
  like", combined per layer. A test checks that only this variant notices reversed edges.
`edge_mask` (one weight per original edge) scales messages in both directions; the explainer
will learn it on Day 7. `chunk` bounds memory when scoring a whole window at once.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

KINDS = ("sage", "gine", "gine_rmp")


def _mlp(i: int, h: int, o: int, dropout: float = 0.0) -> nn.Sequential:
    return nn.Sequential(nn.Linear(i, h), nn.ReLU(), nn.Dropout(dropout), nn.Linear(h, o))


def _chunks(n: int, chunk: int | None):
    step = chunk or max(n, 1)
    for s in range(0, n, step):
        yield slice(s, s + step)


class GINELayer(nn.Module):
    def __init__(self, dim: int, edge_dim: int):
        super().__init__()
        self.edge = nn.Linear(edge_dim, dim)
        self.mlp = _mlp(dim, dim, dim)
        self.eps = nn.Parameter(torch.zeros(1))

    def forward(self, x, edge_index, edge_attr, edge_mask=None, chunk=None):
        src, dst = edge_index[0], edge_index[1]
        total = torch.zeros_like(x)
        weight = x.new_zeros(x.size(0))
        for sl in _chunks(src.numel(), chunk):
            w = edge_mask[sl] if edge_mask is not None else x.new_ones(src[sl].numel())
            msg = F.relu(x[src[sl]] + self.edge(edge_attr[sl]))
            total.index_add_(0, dst[sl], msg * w.unsqueeze(-1))
            weight.index_add_(0, dst[sl], w)
        agg = total / weight.clamp(min=1).unsqueeze(-1)  # mean, see module docstring
        return self.mlp((1 + self.eps) * x + agg)


class SAGELayer(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.lin_self = nn.Linear(dim, dim)
        self.lin_nbr = nn.Linear(dim, dim, bias=False)

    def forward(self, x, edge_index, edge_attr=None, edge_mask=None, chunk=None):
        src, dst = edge_index[0], edge_index[1]
        total = torch.zeros_like(x)
        weight = x.new_zeros(x.size(0))
        for sl in _chunks(src.numel(), chunk):
            w = edge_mask[sl] if edge_mask is not None else x.new_ones(src[sl].numel())
            total.index_add_(0, dst[sl], x[src[sl]] * w.unsqueeze(-1))
            weight.index_add_(0, dst[sl], w)
        return self.lin_self(x) + self.lin_nbr(total / weight.clamp(min=1).unsqueeze(-1))


class AccountGNN(nn.Module):
    def __init__(
        self,
        kind: str,
        in_dim: int,
        edge_dim: int,
        hidden: int = 64,
        layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        if kind not in KINDS:
            raise ValueError(f"kind must be one of {KINDS}")
        self.kind, self.dropout, self.n_layers = kind, dropout, layers
        self.inp = nn.Linear(in_dim, hidden)
        if kind == "gine_rmp":
            self.fwd = nn.ModuleList(GINELayer(hidden, edge_dim) for _ in range(layers))
            self.rev = nn.ModuleList(GINELayer(hidden, edge_dim) for _ in range(layers))
            self.mix = nn.ModuleList(nn.Linear(2 * hidden, hidden) for _ in range(layers))
        elif kind == "gine":
            self.convs = nn.ModuleList(GINELayer(hidden, edge_dim) for _ in range(layers))
        else:
            self.convs = nn.ModuleList(SAGELayer(hidden) for _ in range(layers))
        self.norms = nn.ModuleList(nn.LayerNorm(hidden) for _ in range(layers))
        self.head = _mlp(hidden, hidden, 1, dropout)

    def embed(self, x, edge_index, edge_attr, edge_mask=None, chunk=None):
        h = F.relu(self.inp(x))
        rev_index = edge_index.flip(0)  # same edge order, so one mask covers both directions
        if self.kind != "gine_rmp":
            both_index = torch.cat([edge_index, rev_index], dim=1)
            both_attr = torch.cat([edge_attr, edge_attr], dim=0)
            both_mask = None if edge_mask is None else torch.cat([edge_mask, edge_mask])
        for i in range(self.n_layers):
            if self.kind == "gine_rmp":
                from_senders = self.fwd[i](h, edge_index, edge_attr, edge_mask, chunk)
                from_receivers = self.rev[i](h, rev_index, edge_attr, edge_mask, chunk)
                new = self.mix[i](torch.cat([from_senders, from_receivers], dim=-1))
            else:
                new = self.convs[i](h, both_index, both_attr, both_mask, chunk)
            h = self.norms[i](F.relu(new) + h)  # residual keeps the account's own features
            h = F.dropout(h, self.dropout, self.training)
        return h

    def forward(self, x, edge_index, edge_attr, edge_mask=None, chunk=None):
        return self.head(self.embed(x, edge_index, edge_attr, edge_mask, chunk)).squeeze(-1)
