"""ACM-GCN encoder (plain acmgcn), ported from SitaoLuan/ACM-GNN ACM-Pytorch.

Layer (models/layers.py GraphConvolution, variant=0, structure_info=0):
    output_low  = ReLU(adj_low  @ (X W_low))
    output_high = ReLU(adj_high @ (X W_high))
    output_mlp  = ReLU(X W_mlp)
    att = softmax( sigmoid([low a_l, high a_h, mlp a_m]) @ A_att / 3 , dim=1)
    out = 3 * (att_low * output_low + att_high * output_high + att_mlp * output_mlp)
Weights init uniform(+-1/sqrt(out_dim)); a_* init uniform(+-1);
A_att (3x3) init uniform(+-1/sqrt(3)).

Model (models/models.py GCN, model_type acmgcn): x = Dropout(x);
fea1 = layer0(x); fea1 = Dropout(ReLU(fea1)); layer1 (hidden -> classes) is
the source final classifier, replaced by the shared readout (rule 4), so
z_v = Dropout(ReLU(layer0(x))) at width 512 with the adopted dropout 0.

Preprocessing (utils.py): row-normalised features;
adj_low = D^-1 (A + I) (row-normalised), adj_high = I - adj_low.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .h2gcn import row_normalize


def acm_adjacencies(
    edge_index: torch.Tensor, num_nodes: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """adj_low = row-normalised (A + I); adj_high = I - adj_low (sparse).

    The source builds A densely (a[src, dst] = 1), which collapses duplicate
    edges; dedupe here so multigraph input cannot double-count.
    """
    key = edge_index[0].long() * num_nodes + edge_index[1].long()
    uniq = torch.unique(key)
    dedup = torch.stack([uniq // num_nodes, uniq % num_nodes])
    loops = torch.arange(num_nodes, device=edge_index.device)
    ei = torch.cat([dedup, loops.expand(2, -1)], dim=1)
    src, dst = ei[0], ei[1]
    deg = torch.bincount(src, minlength=num_nodes).float()
    low_weights = 1.0 / deg[src].clamp(min=1.0)
    adj_low = torch.sparse_coo_tensor(
        ei, low_weights, (num_nodes, num_nodes)
    ).coalesce()
    high = torch.sparse_coo_tensor(
        torch.stack([loops, loops]),
        torch.ones(num_nodes, device=edge_index.device),
        (num_nodes, num_nodes),
    )
    adj_high = (high - adj_low).coalesce()
    return adj_low, adj_high


class ACMConvolution(nn.Module):
    """GraphConvolution for model_type acmgcn, variant=0, structure_info=0."""

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.weight_low = nn.Parameter(torch.empty(in_dim, out_dim))
        self.weight_high = nn.Parameter(torch.empty(in_dim, out_dim))
        self.weight_mlp = nn.Parameter(torch.empty(in_dim, out_dim))
        self.att_vec_low = nn.Parameter(torch.empty(out_dim, 1))
        self.att_vec_high = nn.Parameter(torch.empty(out_dim, 1))
        self.att_vec_mlp = nn.Parameter(torch.empty(out_dim, 1))
        self.att_vec = nn.Parameter(torch.empty(3, 3))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.weight_mlp.size(1))
        std_att = 1.0 / math.sqrt(self.att_vec_mlp.size(1))
        std_att_vec = 1.0 / math.sqrt(self.att_vec.size(1))
        for w in (self.weight_low, self.weight_high, self.weight_mlp):
            nn.init.uniform_(w, -stdv, stdv)
        for a in (self.att_vec_low, self.att_vec_high, self.att_vec_mlp):
            nn.init.uniform_(a, -std_att, std_att)
        nn.init.uniform_(self.att_vec, -std_att_vec, std_att_vec)

    def forward(
        self, x: torch.Tensor, adj_low: torch.Tensor, adj_high: torch.Tensor
    ) -> torch.Tensor:
        output_low = F.relu(torch.sparse.mm(adj_low, x @ self.weight_low))
        output_high = F.relu(torch.sparse.mm(adj_high, x @ self.weight_high))
        output_mlp = F.relu(x @ self.weight_mlp)
        logits = (
            torch.sigmoid(
                torch.cat(
                    [
                        output_low @ self.att_vec_low,
                        output_high @ self.att_vec_high,
                        output_mlp @ self.att_vec_mlp,
                    ],
                    dim=1,
                )
            )
            @ self.att_vec
            / 3
        )
        att = torch.softmax(logits, dim=1)
        return 3 * (
            att[:, 0:1] * output_low
            + att[:, 1:2] * output_high
            + att[:, 2:3] * output_mlp
        )


class ACMGCNEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 512, dropout: float = 0.0):
        super().__init__()
        self.conv = ACMConvolution(in_dim, hidden_dim)
        self.dropout = dropout
        self._adj: tuple[torch.Tensor, torch.Tensor] | None = None
        self._adj_num_edges = -1

    def clear_graph_cache(self) -> None:
        """Discard the cached adjacencies before evaluating a different graph."""
        self._adj = None
        self._adj_num_edges = -1

    def _get_adjs(self, edge_index: torch.Tensor, num_nodes: int):
        if self._adj is None or self._adj_num_edges != edge_index.shape[1]:
            self._adj = acm_adjacencies(edge_index, num_nodes)
            self._adj_num_edges = edge_index.shape[1]
        return self._adj

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        adj_low, adj_high = self._get_adjs(edge_index, x.shape[0])
        x = row_normalize(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        h = self.conv(x, adj_low, adj_high)
        return F.dropout(F.relu(h), p=self.dropout, training=self.training)
