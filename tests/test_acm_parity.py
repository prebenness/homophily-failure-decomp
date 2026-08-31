"""Math-parity test: our ACM-GCN layer vs the authors' code, verbatim.

GraphConvolution below is copied from SitaoLuan/ACM-GNN,
ACM-Pytorch/models/layers.py (main branch as of 2026-07-08), reduced to the
code paths reachable for model_type="acmgcn", variant=0, structure_info=0 —
the forward/attention3/reset_parameters bodies are verbatim (device pinning
removed so the test runs on CPU):
https://github.com/SitaoLuan/ACM-GNN/blob/main/ACM-Pytorch/models/layers.py

The test initialises our ACMConvolution and the reference with identical
weights and asserts equal outputs on a random graph. It also checks our
adjacency construction against the authors' utils.py recipe
(adj_low = row-normalised (I + A), adj_high = I - adj_low).
"""

import math

import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module

from hfd.models.acmgcn import ACMConvolution, acm_adjacencies


class GraphConvolution(Module):
    """Verbatim acmgcn/variant=0/structure_info=0 paths from the source."""

    def __init__(self, in_features, out_features, nnodes, model_type,
                 output_layer=0, variant=False, structure_info=0):
        super(GraphConvolution, self).__init__()
        (
            self.in_features, self.out_features, self.output_layer,
            self.model_type, self.structure_info, self.variant,
        ) = (in_features, out_features, output_layer, model_type,
             structure_info, variant)
        self.att_low, self.att_high, self.att_mlp = 0, 0, 0
        self.weight_low, self.weight_high, self.weight_mlp = (
            Parameter(torch.FloatTensor(in_features, out_features)),
            Parameter(torch.FloatTensor(in_features, out_features)),
            Parameter(torch.FloatTensor(in_features, out_features)),
        )
        self.att_vec_low, self.att_vec_high, self.att_vec_mlp = (
            Parameter(torch.FloatTensor(1 * out_features, 1)),
            Parameter(torch.FloatTensor(1 * out_features, 1)),
            Parameter(torch.FloatTensor(1 * out_features, 1)),
        )
        self.att_vec = Parameter(torch.FloatTensor(3, 3))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.weight_mlp.size(1))
        std_att = 1.0 / math.sqrt(self.att_vec_mlp.size(1))
        std_att_vec = 1.0 / math.sqrt(self.att_vec.size(1))

        self.weight_low.data.uniform_(-stdv, stdv)
        self.weight_high.data.uniform_(-stdv, stdv)
        self.weight_mlp.data.uniform_(-stdv, stdv)
        self.att_vec_high.data.uniform_(-std_att, std_att)
        self.att_vec_low.data.uniform_(-std_att, std_att)
        self.att_vec_mlp.data.uniform_(-std_att, std_att)
        self.att_vec.data.uniform_(-std_att_vec, std_att_vec)

    def attention3(self, output_low, output_high, output_mlp):
        T = 3
        logits = (
            torch.mm(
                torch.sigmoid(
                    torch.cat(
                        [
                            torch.mm((output_low), self.att_vec_low),
                            torch.mm((output_high), self.att_vec_high),
                            torch.mm((output_mlp), self.att_vec_mlp),
                        ],
                        1,
                    )
                ),
                self.att_vec,
            )
            / T
        )
        att = torch.softmax(logits, 1)
        return att[:, 0][:, None], att[:, 1][:, None], att[:, 2][:, None]

    def forward(self, input, adj_low, adj_high, adj_low_unnormalized):
        output_low = F.relu(torch.spmm(adj_low, (torch.mm(input, self.weight_low))))
        output_high = F.relu(torch.spmm(adj_high, (torch.mm(input, self.weight_high))))
        output_mlp = F.relu(torch.mm(input, self.weight_mlp))

        self.att_low, self.att_high, self.att_mlp = self.attention3(
            (output_low), (output_high), (output_mlp)
        )
        return 3 * (
            self.att_low * output_low
            + self.att_high * output_high
            + self.att_mlp * output_mlp
        )


def _random_graph(n=10, m=30, seed=11):
    g = torch.Generator().manual_seed(seed)
    src = torch.randint(0, n, (m,), generator=g)
    dst = torch.randint(0, n, (m,), generator=g)
    keep = src != dst
    ei = torch.stack(
        [torch.cat([src[keep], dst[keep]]), torch.cat([dst[keep], src[keep]])]
    )
    x = torch.randn(n, 6, generator=g)
    return x, ei, n


def _reference_adjacencies(edge_index, n):
    """Authors' utils.py recipe: adj_low = normalize_tensor(I + A) (row
    normalisation), adj_high = I - adj_low, both dense here."""
    a = torch.zeros(n, n)
    a[edge_index[0], edge_index[1]] = 1.0
    low = torch.eye(n) + a
    rowsum = low.sum(1)
    r_inv = torch.pow(rowsum, -1).flatten()
    r_inv[torch.isinf(r_inv)] = 0.0
    low = torch.diag(r_inv) @ low
    high = torch.eye(n) - low
    return low, high


def test_acm_adjacency_matches_source_recipe():
    _, ei, n = _random_graph()
    ours_low, ours_high = acm_adjacencies(ei, n)
    ref_low, ref_high = _reference_adjacencies(ei, n)
    assert torch.allclose(ours_low.to_dense(), ref_low, atol=1e-6)
    assert torch.allclose(ours_high.to_dense(), ref_high, atol=1e-6)


def test_acm_layer_matches_reference():
    x, ei, n = _random_graph()
    torch.manual_seed(0)
    ours = ACMConvolution(6, 8)
    reference = GraphConvolution(6, 8, n, model_type="acmgcn", variant=0,
                                 structure_info=0)
    with torch.no_grad():
        for name in ("weight_low", "weight_high", "weight_mlp",
                     "att_vec_low", "att_vec_high", "att_vec_mlp", "att_vec"):
            getattr(reference, name).data = getattr(ours, name).data.clone()

    adj_low, adj_high = acm_adjacencies(ei, n)
    out_ours = ours(x, adj_low, adj_high)
    out_ref = reference(x, adj_low, adj_high, None)
    assert torch.allclose(out_ours, out_ref, atol=1e-6), (
        f"max abs diff {(out_ours - out_ref).abs().max().item()}"
    )
