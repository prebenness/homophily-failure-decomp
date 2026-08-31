"""LINKX encoder (PyG torch_geometric.nn.models.LINKX).

Repository training defaults (CUAI/Non-Homophily-Large-Scale parse.py):
width 512, num_layers=2, num_edge_layers=1, num_node_layers=1, dropout 0.5.
The source final classifier output is overridden to 512 (out_channels), so
z_v is the LINKX output representation.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch_geometric.nn.models import LINKX


class LINKXEncoder(nn.Module):
    def __init__(
        self,
        in_dim: int,
        num_nodes: int,
        hidden_dim: int = 512,
        out_dim: int = 512,
        num_layers: int = 2,
        num_edge_layers: int = 1,
        num_node_layers: int = 1,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.linkx = LINKX(
            num_nodes=num_nodes,
            in_channels=in_dim,
            hidden_channels=hidden_dim,
            out_channels=out_dim,
            num_layers=num_layers,
            num_edge_layers=num_edge_layers,
            num_node_layers=num_node_layers,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        return self.linkx(x, edge_index)
