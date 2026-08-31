"""Unit tests for the Platonov skeleton encoder."""

import numpy as np
import torch

from hfd.models import build_model
from hfd.models.platonov import PlatonovEncoder, normalized_adjacency


def test_normalized_adjacency_with_self_loops():
    # Path graph 0-1-2 (both directions), self-loops added by the function.
    ei = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
    adj = normalized_adjacency(ei, 3, add_self_loops=True).to_dense()
    # Degrees including self-loops: [2, 3, 2].
    expected = np.zeros((3, 3))
    deg = np.array([2.0, 3.0, 2.0])
    for u, v in [(0, 1), (1, 0), (1, 2), (2, 1), (0, 0), (1, 1), (2, 2)]:
        expected[u, v] = 1.0 / np.sqrt(deg[u] * deg[v])
    assert np.allclose(adj.numpy(), expected)


def test_aggregation_includes_self_when_loops_added():
    ei = torch.tensor([[0, 1], [1, 0]])
    adj = normalized_adjacency(ei, 2, add_self_loops=True)
    x = torch.tensor([[1.0], [0.0]])
    out = torch.sparse.mm(adj, x)
    # Node 0: 1/2 * own + 1/2 * neighbour = 0.5; node 1: 0.5 * 1 + 0 = 0.5.
    assert torch.allclose(out, torch.tensor([[0.5], [0.5]]))


def test_encoder_output_shape_and_grad():
    torch.manual_seed(0)
    enc = PlatonovEncoder(in_dim=7, hidden_dim=16, num_blocks=5, block="gcn")
    x = torch.randn(10, 7)
    ei = torch.randint(0, 10, (2, 30))
    z = enc(x, ei)
    assert z.shape == (10, 16)
    z.sum().backward()
    assert enc.input_linear.weight.grad is not None


def test_build_model_gcn_platonov():
    cfg = {
        "z_dim": 16,
        "head_dropout": 0.2,
        "encoder": {"hidden_dim": 16, "num_blocks": 2, "dropout": 0.2},
    }
    model = build_model("gcn", in_dim=5, num_classes=3, model_cfg=cfg)
    x = torch.randn(8, 5)
    ei = torch.randint(0, 8, (2, 20))
    logits = model(x, ei)
    assert logits.shape == (8, 3)
    # Five parameterised pieces per block: LayerNorm + two FFN linears.
    assert isinstance(model.encoder, PlatonovEncoder)
