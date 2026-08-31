"""Math-parity test: our GPR propagation vs the authors' code, verbatim.

GPR_prop below is copied unmodified (save for import paths) from
jianhao2016/GPRGNN, src/GNN_models.py, commit on master as of 2026-07-08:
https://github.com/jianhao2016/GPRGNN/blob/master/src/GNN_models.py
It is the reference implementation; the test initialises our GPRProp and the
reference with identical weights and asserts equal outputs on a random graph
to floating-point precision. A silent translation error (wrong normalisation,
extra self-loops, misplaced dropout) would break this equality.
"""

import numpy as np
import pytest
import torch
from torch.nn import Parameter
from torch_geometric.nn import MessagePassing
from torch_geometric.nn.conv.gcn_conv import gcn_norm

from hfd.models.gpr import GPRProp, _gpr_normalized_adjacency


class GPR_prop(MessagePassing):
    '''
    propagation class for GPR_GNN
    (verbatim from jianhao2016/GPRGNN src/GNN_models.py)
    '''

    def __init__(self, K, alpha, Init, Gamma=None, bias=True, **kwargs):
        super(GPR_prop, self).__init__(aggr='add', **kwargs)
        self.K = K
        self.Init = Init
        self.alpha = alpha
        self.Gamma = Gamma

        assert Init in ['SGC', 'PPR', 'NPPR', 'Random', 'WS']
        if Init == 'SGC':
            # SGC-like, note that in this case, alpha has to be a integer. It means where the peak at when initializing GPR weights.
            TEMP = 0.0*np.ones(K+1)
            TEMP[alpha] = 1.0
        elif Init == 'PPR':
            # PPR-like
            TEMP = alpha*(1-alpha)**np.arange(K+1)
            TEMP[-1] = (1-alpha)**K
        elif Init == 'NPPR':
            # Negative PPR
            TEMP = (alpha)**np.arange(K+1)
            TEMP = TEMP/np.sum(np.abs(TEMP))
        elif Init == 'Random':
            # Random
            bound = np.sqrt(3/(K+1))
            TEMP = np.random.uniform(-bound, bound, K+1)
            TEMP = TEMP/np.sum(np.abs(TEMP))
        elif Init == 'WS':
            # Specify Gamma
            TEMP = Gamma

        self.temp = Parameter(torch.tensor(TEMP))

    def reset_parameters(self):
        torch.nn.init.zeros_(self.temp)
        if self.Init == 'SGC':
            self.temp.data[self.alpha] = 1.0
        elif self.Init == 'PPR':
            for k in range(self.K+1):
                self.temp.data[k] = self.alpha*(1-self.alpha)**k
            self.temp.data[-1] = (1-self.alpha)**self.K
        elif self.Init == 'NPPR':
            for k in range(self.K+1):
                self.temp.data[k] = self.alpha**k
            self.temp.data = self.temp.data/torch.sum(torch.abs(self.temp.data))
        elif self.Init == 'Random':
            bound = np.sqrt(3/(self.K+1))
            torch.nn.init.uniform_(self.temp, -bound, bound)
            self.temp.data = self.temp.data/torch.sum(torch.abs(self.temp.data))
        elif self.Init == 'WS':
            self.temp.data = self.Gamma

    def forward(self, x, edge_index, edge_weight=None):
        edge_index, norm = gcn_norm(
            edge_index, edge_weight, num_nodes=x.size(0), dtype=x.dtype)

        hidden = x*(self.temp[0])
        for k in range(self.K):
            x = self.propagate(edge_index, x=x, norm=norm)
            gamma = self.temp[k+1]
            hidden = hidden + gamma*x
        return hidden

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j


def _random_graph(n=10, m=30, seed=7):
    g = torch.Generator().manual_seed(seed)
    src = torch.randint(0, n, (m,), generator=g)
    dst = torch.randint(0, n, (m,), generator=g)
    keep = src != dst
    ei = torch.stack(
        [torch.cat([src[keep], dst[keep]]), torch.cat([dst[keep], src[keep]])]
    )
    x = torch.randn(n, 5, generator=g)
    return x, ei


@pytest.mark.parametrize("K,alpha", [(10, 0.1), (3, 0.5)])
def test_gpr_propagation_matches_reference(K, alpha):
    x, ei = _random_graph()
    ours = GPRProp(K=K, alpha=alpha)
    reference = GPR_prop(K=K, alpha=alpha, Init="PPR")
    # Identical weights (PPR init is deterministic, but copy explicitly).
    with torch.no_grad():
        reference.temp.data = ours.temp.data.clone().to(reference.temp.dtype)

    out_ours = ours(x, ei)
    out_ref = reference(x, ei).float()
    assert torch.allclose(out_ours, out_ref, atol=1e-6), (
        f"max abs diff {(out_ours - out_ref).abs().max().item()}"
    )


def test_gpr_parity_with_isolated_nodes():
    # Isolated node: gcn_norm adds self-loops, both paths must agree.
    x = torch.randn(6, 4)
    ei = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])  # nodes 3,4,5 isolated
    ours = GPRProp(K=4, alpha=0.2)
    reference = GPR_prop(K=4, alpha=0.2, Init="PPR")
    with torch.no_grad():
        reference.temp.data = ours.temp.data.clone().to(reference.temp.dtype)
    assert torch.allclose(ours(x, ei), reference(x, ei).float(), atol=1e-6)


def test_gpr_sparse_propagation_matches_reference_gradients():
    x, ei = _random_graph(n=12, m=40, seed=19)
    ours_x = x.clone().requires_grad_()
    reference_x = x.clone().requires_grad_()
    ours = GPRProp(K=4, alpha=0.2)
    reference = GPR_prop(K=4, alpha=0.2, Init="PPR").float()
    with torch.no_grad():
        reference.temp.copy_(ours.temp)

    ours_output = ours(ours_x, ei)
    reference_output = reference(reference_x, ei)
    upstream = torch.randn_like(ours_output)
    (ours_output * upstream).sum().backward()
    (reference_output * upstream).sum().backward()

    assert torch.allclose(ours_output, reference_output, atol=1e-6)
    assert torch.allclose(ours_x.grad, reference_x.grad, atol=1e-6)
    assert torch.allclose(ours.temp.grad, reference.temp.grad, atol=1e-5)


def test_gpr_normalized_operator_is_sparse_and_edge_sized():
    x, edge_index = _random_graph(n=20, m=70, seed=23)
    adjacency = _gpr_normalized_adjacency(
        edge_index,
        num_nodes=x.shape[0],
        dtype=x.dtype,
    )

    assert adjacency.layout == torch.sparse_coo
    assert adjacency.shape == (x.shape[0], x.shape[0])
    assert adjacency._nnz() <= edge_index.shape[1] + x.shape[0]
