"""GPR-GNN encoder (jianhao2016/GPRGNN, GNN_models.py).

Two-layer MLP feature extractor with feature dropout 0.5, then learned GPR
propagation: gamma-weighted sum of 0..K-hop propagations under GCN
normalisation (self-loops added), K=10, alpha=0.1, PPR initialisation,
propagation dropout dprate=0.5. The source lin2 (hidden -> classes) is
overridden to 512 so z_v is the propagated 512-dimensional output.

Optimiser (source train_model.py): Adam, lr 0.002; weight decay 5e-4 on the
two linear layers and 0 on the propagation weights — exposed via
param_groups().
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.conv.gcn_conv import gcn_norm


def _gpr_normalized_adjacency(
    edge_index: torch.Tensor,
    num_nodes: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Return the source-to-target GCN operator as a sparse COO tensor.

    Rows are target nodes and columns are source nodes, matching PyG's
    source-to-target ``MessagePassing`` convention.  Coalescing duplicate
    entries is mathematically equivalent to summing their messages, while
    keeping storage proportional to the number of edges.
    """
    norm_edge_index, norm = gcn_norm(
        edge_index,
        None,
        num_nodes=num_nodes,
        dtype=dtype,
    )
    src, dst = norm_edge_index[0], norm_edge_index[1]
    return torch.sparse_coo_tensor(
        torch.stack([dst, src]),
        norm,
        (num_nodes, num_nodes),
    ).coalesce()


class GPRProp(nn.Module):
    """GPR_prop with PPR initialisation, as in the source."""

    def __init__(self, K: int = 10, alpha: float = 0.1):
        super().__init__()
        self.K = K
        self.alpha = alpha
        gammas = alpha * (1 - alpha) ** torch.arange(K + 1, dtype=torch.float64)
        gammas[-1] = (1 - alpha) ** K
        self.temp = nn.Parameter(gammas.float())

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        adjacency = _gpr_normalized_adjacency(
            edge_index,
            num_nodes=x.size(0),
            dtype=x.dtype,
        )
        hidden = x * self.temp[0]
        for k in range(self.K):
            # propagate: x[v] = sum_u norm(u,v) * x[u].  Sparse matrix
            # multiplication avoids materialising an |E| x feature-width
            # message tensor, which is several GiB on ogbn-arxiv at width 512.
            x = torch.sparse.mm(adjacency, x)
            hidden = hidden + self.temp[k + 1] * x
        return hidden


class GPRGNNEncoder(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 512,
        out_dim: int = 512,
        K: int = 10,
        alpha: float = 0.1,
        dropout: float = 0.5,
        dprate: float = 0.5,
    ):
        super().__init__()
        self.lin1 = nn.Linear(in_dim, hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, out_dim)
        self.prop1 = GPRProp(K, alpha)
        self.dropout = dropout
        self.dprate = dprate

    def param_groups(self, lr: float, weight_decay: float) -> list[dict]:
        """Source optimiser groups: wd on linears, none on propagation."""
        return [
            {"params": self.lin1.parameters(), "weight_decay": weight_decay, "lr": lr},
            {"params": self.lin2.parameters(), "weight_decay": weight_decay, "lr": lr},
            {"params": self.prop1.parameters(), "weight_decay": 0.0, "lr": lr},
        ]

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)
        if self.dprate > 0.0:
            x = F.dropout(x, p=self.dprate, training=self.training)
        return self.prop1(x, edge_index)
