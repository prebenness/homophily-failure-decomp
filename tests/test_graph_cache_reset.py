"""Regression tests for switching between graphs with equal edge counts."""

import torch

from hfd.models.acmgcn import ACMGCNEncoder, acm_adjacencies
from hfd.models import h2gcn as h2gcn_module
from hfd.models.h2gcn import H2GCNEncoder, prepare_h2gcn_operators
from hfd.models.platonov import PlatonovEncoder, mean_adjacency


def _equal_size_graphs() -> tuple[torch.Tensor, torch.Tensor]:
    graph_a = torch.tensor(
        [
            [0, 1, 1, 2, 2, 3, 3, 0],
            [1, 0, 2, 1, 3, 2, 0, 3],
        ]
    )
    graph_b = torch.tensor(
        [
            [0, 1, 1, 3, 3, 2, 2, 0],
            [1, 0, 3, 1, 2, 3, 0, 2],
        ]
    )
    assert graph_a.shape == graph_b.shape
    return graph_a, graph_b


def test_platonov_clear_graph_cache_rebuilds_equal_edge_count_graph() -> None:
    graph_a, graph_b = _equal_size_graphs()
    encoder = PlatonovEncoder(
        in_dim=3,
        hidden_dim=4,
        num_blocks=1,
        dropout=0.0,
        block="sage",
    )
    x = torch.ones(4, 3)

    encoder(x, graph_a)
    cached_a = encoder._adj.to_dense().clone()
    encoder.clear_graph_cache()

    assert encoder._adj is None
    assert encoder._adj_num_edges == -1
    encoder(x, graph_b)
    rebuilt_b = encoder._adj.to_dense()

    assert not torch.equal(rebuilt_b, cached_a)
    assert torch.equal(rebuilt_b, mean_adjacency(graph_b, 4).to_dense())


def test_acmgcn_clear_graph_cache_rebuilds_equal_edge_count_graph() -> None:
    graph_a, graph_b = _equal_size_graphs()
    encoder = ACMGCNEncoder(in_dim=3, hidden_dim=4, dropout=0.0)
    x = torch.ones(4, 3)

    encoder(x, graph_a)
    cached_low_a = encoder._adj[0].to_dense().clone()
    encoder.clear_graph_cache()

    assert encoder._adj is None
    assert encoder._adj_num_edges == -1
    encoder(x, graph_b)
    rebuilt_low_b, rebuilt_high_b = encoder._adj
    expected_low_b, expected_high_b = acm_adjacencies(graph_b, 4)

    assert not torch.equal(rebuilt_low_b.to_dense(), cached_low_a)
    assert torch.equal(rebuilt_low_b.to_dense(), expected_low_b.to_dense())
    assert torch.equal(rebuilt_high_b.to_dense(), expected_high_b.to_dense())


def test_h2gcn_clear_graph_cache_rebuilds_equal_edge_count_graph() -> None:
    graph_a, graph_b = _equal_size_graphs()
    encoder = H2GCNEncoder(in_dim=3, hidden_dim=2, dropout=0.0)
    x = torch.ones(4, 3)

    encoder(x, graph_a)
    cached_hop1_a = encoder._adj[0].to_dense().clone()
    encoder.clear_graph_cache()

    assert encoder._adj is None
    assert encoder._adj_num_edges == -1
    encoder(x, graph_b)
    rebuilt_hop1_b, rebuilt_hop2_b = encoder._adj
    expected_hop1_b, expected_hop2_b = prepare_h2gcn_operators(graph_b, 4)

    assert not torch.equal(rebuilt_hop1_b.to_dense(), cached_hop1_a)
    assert torch.equal(rebuilt_hop1_b.to_dense(), expected_hop1_b.to_dense())
    assert torch.equal(rebuilt_hop2_b.to_dense(), expected_hop2_b.to_dense())


def test_h2gcn_prepared_operators_are_reused_across_checkpoints(
    monkeypatch,
) -> None:
    graph, _ = _equal_size_graphs()
    x = torch.arange(12, dtype=torch.float32).reshape(4, 3) + 1
    original_prepare = h2gcn_module.prepare_h2gcn_operators
    calls = 0

    def counted_prepare(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original_prepare(*args, **kwargs)

    monkeypatch.setattr(
        h2gcn_module,
        "prepare_h2gcn_operators",
        counted_prepare,
    )
    operators = h2gcn_module.prepare_h2gcn_operators(graph.numpy(), 4)
    template = H2GCNEncoder(in_dim=3, hidden_dim=2, dropout=0.0)
    state = template.state_dict()
    outputs = []

    for _ in range(2):
        encoder = H2GCNEncoder(in_dim=3, hidden_dim=2, dropout=0.0)
        encoder.load_state_dict(state)
        encoder.set_graph_cache(operators, num_edges=graph.shape[1])
        outputs.append(encoder(x, graph))
        assert encoder._adj[0] is operators[0]
        assert encoder._adj[1] is operators[1]

    assert calls == 1
    assert torch.equal(outputs[0], outputs[1])
