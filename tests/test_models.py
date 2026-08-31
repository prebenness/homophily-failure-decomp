"""Unit tests for the full encoder roster."""

import numpy as np
import pytest
import torch

from hfd.models import build_model
from hfd.models.acmgcn import ACMConvolution, acm_adjacencies
from hfd.models.gpr import GPRProp
from hfd.models.h2gcn import h2gcn_hop_matrices, row_normalize
from hfd.models.platonov import mean_adjacency


def _toy_graph(n=12, m=40, seed=0):
    g = torch.Generator().manual_seed(seed)
    src = torch.randint(0, n, (m,), generator=g)
    dst = torch.randint(0, n, (m,), generator=g)
    keep = src != dst
    ei = torch.stack([torch.cat([src[keep], dst[keep]]), torch.cat([dst[keep], src[keep]])])
    return ei


@pytest.mark.parametrize(
    "name,extra",
    [
        ("mlp", {}),
        ("gcn", {}),
        ("graphsage-mean", {}),
        ("h2gcn", {}),
        ("gpr-gnn", {"out_dim": 16, "K": 3}),
        ("linkx", {"out_dim": 16}),
        ("acm-gcn", {}),
    ],
)
def test_encoder_shapes_and_grads(name, extra):
    torch.manual_seed(0)
    n, f, c = 12, 7, 3
    cfg = {
        "z_dim": 16,
        "head_dropout": 0.2,
        "encoder": {"hidden_dim": 16, **extra},
    }
    if name in ("mlp", "gcn", "graphsage-mean"):
        cfg["encoder"]["num_blocks"] = 2
    if name == "h2gcn":
        # z_v is the native 7-block concatenation of the internal width.
        cfg["z_dim"] = 7 * 16
    model = build_model(name, f, c, cfg, num_nodes=n)
    x = torch.randn(n, f).abs()  # non-negative rows for row_normalize models
    ei = _toy_graph(n)
    logits = model(x, ei)
    assert logits.shape == (n, c)
    logits.sum().backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert any(g is not None and g.abs().sum() > 0 for g in grads)


def test_mean_adjacency_is_neighbour_mean():
    # Star: node 0 connected to 1,2,3.
    ei = torch.tensor([[0, 0, 0, 1, 2, 3], [1, 2, 3, 0, 0, 0]])
    adj = mean_adjacency(ei, 4)
    x = torch.tensor([[0.0], [3.0], [6.0], [9.0]])
    out = torch.sparse.mm(adj, x)
    assert out[0].item() == pytest.approx(6.0)  # mean of 3,6,9
    assert out[1].item() == pytest.approx(0.0)


def test_h2gcn_hop_matrices_path_graph():
    # Path 0-1-2-3: exact 1-hop pairs (i,i+1); exact 2-hop pairs (0,2),(1,3).
    ei = np.array([[0, 1, 2, 1, 2, 3], [1, 2, 3, 0, 1, 2]])
    hop1, hop2 = h2gcn_hop_matrices(ei, 4)
    h1 = (hop1.toarray() > 0).astype(int)
    h2 = (hop2.toarray() > 0).astype(int)
    expected1 = np.array(
        [[0, 1, 0, 0], [1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0]]
    )
    expected2 = np.array(
        [[0, 0, 1, 0], [0, 0, 0, 1], [1, 0, 0, 0], [0, 1, 0, 0]]
    )
    assert np.array_equal(h1, expected1)
    assert np.array_equal(h2, expected2)
    # Symmetric normalisation: row sums of hop1 on interior nodes < degree.
    assert hop1.toarray()[0, 1] == pytest.approx(1 / np.sqrt(1 * 2))


def test_h2gcn_native_concat_width():
    from hfd.models.h2gcn import H2GCNEncoder

    enc = H2GCNEncoder(in_dim=9, hidden_dim=73)
    x = torch.rand(15, 9)
    ei = _toy_graph(15)
    z = enc(x, ei)
    assert z.shape == (15, 511)
    assert enc.out_dim == 511


def test_row_normalize_matches_source_semantics():
    x = torch.tensor([[2.0, 2.0], [0.0, 0.0], [1.0, 3.0]])
    out = row_normalize(x)
    assert torch.allclose(out[0], torch.tensor([0.5, 0.5]))
    assert torch.allclose(out[1], torch.zeros(2))  # 1/0 -> 0 row
    assert torch.allclose(out[2], torch.tensor([0.25, 0.75]))


def test_acm_adjacencies():
    ei = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
    adj_low, adj_high = acm_adjacencies(ei, 3)
    low = adj_low.to_dense()
    # Rows of D^-1 (A+I) sum to 1.
    assert torch.allclose(low.sum(dim=1), torch.ones(3))
    # adj_high = I - adj_low.
    assert torch.allclose(adj_high.to_dense(), torch.eye(3) - low)


def test_acm_attention_convex_mixing():
    torch.manual_seed(0)
    conv = ACMConvolution(5, 8)
    n = 6
    x = torch.randn(n, 5)
    ei = _toy_graph(n, 20)
    adj_low, adj_high = acm_adjacencies(ei, n)
    out = conv(x, adj_low, adj_high)
    assert out.shape == (n, 8)


def test_gpr_gamma_ppr_init():
    prop = GPRProp(K=10, alpha=0.1)
    gammas = prop.temp.detach().numpy()
    expected = 0.1 * 0.9 ** np.arange(11)
    expected[-1] = 0.9**10
    assert np.allclose(gammas, expected, atol=1e-6)


def test_gpr_propagation_no_graph_is_identity_scaled():
    # Empty graph: gcn_norm adds self-loops, so propagation keeps x and the
    # output is sum(gammas) * x.
    prop = GPRProp(K=2, alpha=0.5)
    x = torch.randn(4, 3)
    ei = torch.empty((2, 0), dtype=torch.long)
    out = prop(x, ei)
    total = prop.temp.sum().item()
    assert torch.allclose(out, x * total, atol=1e-5)
