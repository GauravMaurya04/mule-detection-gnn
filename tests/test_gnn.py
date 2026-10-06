"""GNN tests. They need PyTorch, so CI (CPU-only, no torch) skips them; run them locally first:
pytest -q tests/test_gnn.py
"""

import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from mule.config import load_config  # noqa: E402
from mule.eval.evaluate import run_evaluate  # noqa: E402
from mule.features.stage import run_features  # noqa: E402
from mule.labels.stage import run_labels  # noqa: E402
from mule.models.gnn.graph import EDGE_COLUMNS, build_graphs  # noqa: E402
from mule.models.gnn.model import AccountGNN, GINELayer  # noqa: E402
from mule.models.gnn.sampler import NeighborSampler  # noqa: E402
from mule.models.gnn.train import run_gnn_train  # noqa: E402
from tests.synthetic import make_interim  # noqa: E402


def _star_graph():
    # node 0 sends to 1..20; node 21 sends to 0; nodes 22 -> 23 are a separate pair
    src = [0] * 20 + [21, 22]
    dst = list(range(1, 21)) + [0, 23]
    return torch.tensor([src, dst]), 24


def test_sampler_keeps_all_edges_of_quiet_nodes_and_caps_busy_ones():
    ei, n = _star_graph()
    sampler = NeighborSampler(ei, n)
    gen = torch.Generator().manual_seed(0)
    nodes, sub_ei, eids, seed_pos = sampler.sample(torch.tensor([22, 0]), [5], gen)
    assert torch.equal(nodes[seed_pos], torch.tensor([22, 0]))
    picked = set(eids.tolist())
    assert 21 in picked  # 22's only edge is always kept
    star_edges = picked & set(range(21))  # node 0 has 21 incident edges; at most 5 drawn
    assert 1 <= len(star_edges) <= 5
    # the subgraph's edges are the original edges, relabelled into the subgraph
    assert torch.equal(nodes[sub_ei], ei[:, eids])


def test_sampler_with_large_fanout_returns_the_full_two_hop_neighbourhood():
    ei, n = _star_graph()
    sampler = NeighborSampler(ei, n)
    nodes, _, eids, _ = sampler.sample(torch.tensor([21]), [100, 100], torch.Generator())
    assert set(nodes.tolist()) == {0, 21, *range(1, 21)}
    assert set(eids.tolist()) == set(range(21))


def test_chunked_messages_match_unchunked():
    torch.manual_seed(0)
    layer = GINELayer(8, 4).eval()
    x, ei, ea = torch.randn(30, 8), torch.randint(0, 30, (2, 200)), torch.randn(200, 4)
    assert torch.allclose(layer(x, ei, ea), layer(x, ei, ea, chunk=17), atol=1e-5)


def test_only_reverse_message_passing_notices_reversed_edges():
    torch.manual_seed(0)
    x, ei, ea = torch.randn(40, 5), torch.randint(0, 40, (2, 120)), torch.randn(120, 3)
    for kind, should_change in (("gine", False), ("sage", False), ("gine_rmp", True)):
        model = AccountGNN(kind, 5, 3, hidden=16, dropout=0.0).eval()
        with torch.no_grad():
            changed = not torch.allclose(model(x, ei, ea), model(x, ei.flip(0), ea), atol=1e-5)
        assert changed == should_change, kind


def test_edge_mask_of_ones_changes_nothing_and_zeros_cut_all_messages():
    torch.manual_seed(0)
    model = AccountGNN("gine_rmp", 5, 3, hidden=16, dropout=0.0).eval()
    x, ei, ea = torch.randn(10, 5), torch.randint(0, 10, (2, 30)), torch.randn(30, 3)
    with torch.no_grad():
        base = model(x, ei, ea)
        assert torch.allclose(base, model(x, ei, ea, edge_mask=torch.ones(30)), atol=1e-6)
        no_edges = model(x, ei[:, :0], ea[:0])
        assert torch.allclose(no_edges, model(x, ei, ea, edge_mask=torch.zeros(30)), atol=1e-6)


@pytest.fixture(scope="module")
def project(tmp_path_factory):
    path = make_interim(tmp_path_factory.mktemp("gnn"))
    with open(path, "a") as f:
        f.write(
            "gnn:\n  hidden: 16\n  max_epochs: 4\n  eval_every: 1\n  patience: 10\n"
            "  device: cpu\n  score_device: cpu\n  batch_size: 64\n  edge_chunk: 1000\n"
            "  edge_max_epochs: 3\n  edge_batch_size: 256\n  edge_patience: 3\n"
        )
    cfg = load_config(path)
    run_labels(cfg)
    run_features(cfg)
    return cfg


def test_graphs_align_labels_and_carry_no_label_information(project):
    graphs, stats = build_graphs(project)
    assert set(graphs) >= {"val", "test", "test_no_tail"}
    for g in graphs.values():
        assert torch.equal(
            torch.from_numpy(g.nodes)[g.label_nodes],
            torch.from_numpy(g.meta["account_id"].to_numpy().copy()).long(),
        )
        assert torch.equal(g.y.bool(), torch.from_numpy(g.meta["is_mule"].to_numpy().copy()))
        assert not torch.isnan(g.x).any() and not torch.isnan(g.edge_attr).any()
        assert g.x.shape[1] == 55 + 1 and g.edge_attr.shape[1] == len(EDGE_COLUMNS)
        assert int(g.edge_index.max()) < len(g.nodes)
    assert not [c for c in stats["feature_columns"] if "mule" in c or "laundering" in c]
    minimal, _ = build_graphs(project, "minimal")
    assert minimal["val"].x.shape[1] == 4  # constant, active, log in-count, log out-count
    assert minimal["val"].x[:, 2:].std(dim=0).min() > 0  # nodes are not all identical


def test_gnn_trains_scores_and_evaluates_like_any_model(project):
    summary = run_gnn_train(project, "gine_rmp", n_seeds=1)
    seed_dir = project.paths.runs_dir / "node" / "gnn_gine_rmp" / "seed0"
    info = json.loads((seed_dir / "run.json").read_text())
    assert info["probabilistic"] and np.isfinite(info["threshold"])
    assert (seed_dir / "model.pt").exists() and (seed_dir / "scores_test_no_tail.parquet").exists()
    assert summary["seeds"][0]["val_pr_auc"] > 0
    report = run_evaluate(project, "node", ["gnn_gine_rmp"], final=False)
    assert "gnn_gine_rmp" in report.read_text()


def test_sampling_depth_follows_the_number_of_layers(project):
    from mule.models.gnn.train import fanouts_for

    assert fanouts_for(project.gnn, 2) == [15, 10]
    assert fanouts_for(project.gnn, 3) == [15, 10, 5]


def test_aggregation_is_an_average_so_sampled_and_full_neighbourhoods_match_in_scale():
    # Duplicating every edge doubles a sum but leaves an average unchanged. Training samples
    # at most 15 neighbours while scoring uses all of them, so only an average is consistent.
    torch.manual_seed(0)
    x, ei, ea = torch.randn(30, 5), torch.randint(0, 30, (2, 90)), torch.randn(90, 3)
    doubled_ei, doubled_ea = torch.cat([ei, ei], dim=1), torch.cat([ea, ea])
    for kind in ("sage", "gine", "gine_rmp"):
        model = AccountGNN(kind, 5, 3, hidden=16, dropout=0.0).eval()
        with torch.no_grad():
            assert torch.allclose(model(x, ei, ea), model(x, doubled_ei, doubled_ea), atol=1e-5), (
                kind
            )


def test_edge_rows_map_every_transaction_onto_its_window_graph(project):
    from mule.models.gnn.edge import build_edge_rows

    graphs, stats = build_graphs(project)
    rows = build_edge_rows(project, graphs, stats)
    for w, r in rows.items():
        g = graphs[w]
        assert len(r.tx_id) == len(r.y) == r.src.numel() == r.attr.shape[0]
        assert int(r.src.max()) < len(g.nodes) and int(r.dst.max()) < len(g.nodes)
        assert r.attr.shape[1] == g.edge_attr.shape[1] + 1 + 10  # + is_self + 5 patterns x2
        assert not torch.isnan(r.attr).any()


def test_transaction_gnn_trains_and_evaluates(project):
    from mule.models.gnn.edge import run_gnn_edge_train

    summary = run_gnn_edge_train(project, "sage", n_seeds=1)
    assert summary["seeds"][0]["val_pr_auc"] >= 0
    seed_dir = project.paths.runs_dir / "edge" / "gnn_sage" / "seed0"
    assert (seed_dir / "scores_test.parquet").exists()
    report = run_evaluate(project, "edge", ["gnn_sage"], final=False)
    assert "gnn_sage" in report.read_text()


def test_ablation_needs_the_tuning_it_borrows(project):
    cfg = project.model_copy(deep=True)
    cfg.gnn = cfg.gnn.model_copy(update={"node_features": "minimal"})
    with pytest.raises(FileNotFoundError):
        run_gnn_train(cfg, "sage", n_seeds=1, params_from="gnn_not_tuned")
