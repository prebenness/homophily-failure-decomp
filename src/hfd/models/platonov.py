"""Platonov benchmark GNN skeleton (yandex-research/heterophilous-graphs).

Shared forward pass (model-hyperparameter-defaults.md, rule 2):
input linear to width 512, dropout 0.2, GELU; five pre-norm residual blocks,
each computing x + Block(LayerNorm(x)); output LayerNorm, whose output is z_v.

Blocks are implemented exactly from the repository's modules.py — they are
NOT PyG conv layers:
- GCN block: parameter-free symmetric degree-normalised neighbour
  aggregation (coefficients 1/sqrt(deg_u * deg_v), self-loops included),
  followed by the feed-forward module.
- SAGE block: concatenates the node state with the mean of neighbour states
  (no self-loops) before a feed-forward module with doubled input width.
- MLP (ResNet): feed-forward module only, no aggregation.
- Feed-forward module: linear -> dropout -> GELU -> linear -> dropout.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class FeedForwardModule(nn.Module):
    """linear -> dropout -> GELU -> linear -> dropout (Platonov modules.py)."""

    def __init__(self, dim: int, input_dim_multiplier: int = 1, dropout: float = 0.2):
        super().__init__()
        self.linear_1 = nn.Linear(dim * input_dim_multiplier, dim)
        self.dropout_1 = nn.Dropout(dropout)
        self.act = nn.GELU()
        self.linear_2 = nn.Linear(dim, dim)
        self.dropout_2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout_2(self.linear_2(self.act(self.dropout_1(self.linear_1(x)))))


class GCNBlock(nn.Module):
    """Symmetric degree-normalised aggregation, then the feed-forward module."""

    ADJACENCY = "sym-selfloops"

    def __init__(self, dim: int, dropout: float = 0.2):
        super().__init__()
        self.feed_forward = FeedForwardModule(dim, 1, dropout)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        return self.feed_forward(torch.sparse.mm(adj, x))


class SAGEBlock(nn.Module):
    """Concat of node state and neighbour mean, then the feed-forward module."""

    ADJACENCY = "mean-noloops"

    def __init__(self, dim: int, dropout: float = 0.2):
        super().__init__()
        self.feed_forward = FeedForwardModule(dim, 2, dropout)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        return self.feed_forward(torch.cat([x, torch.sparse.mm(adj, x)], dim=1))


class FFNBlock(nn.Module):
    """Feed-forward module only (the graphless ResNet baseline)."""

    ADJACENCY = None

    def __init__(self, dim: int, dropout: float = 0.2):
        super().__init__()
        self.feed_forward = FeedForwardModule(dim, 1, dropout)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        return self.feed_forward(x)


BLOCKS = {"gcn": GCNBlock, "sage": SAGEBlock, "ffn": FFNBlock}


def normalized_adjacency(
    edge_index: torch.Tensor, num_nodes: int, add_self_loops: bool
) -> torch.Tensor:
    """Sparse A with entries 1/sqrt(deg_u * deg_v); degrees include self-loops
    when they are added."""
    if add_self_loops:
        loops = torch.arange(num_nodes, device=edge_index.device)
        edge_index = torch.cat([edge_index, loops.expand(2, -1)], dim=1)
    src, dst = edge_index[0], edge_index[1]
    deg = torch.bincount(dst, minlength=num_nodes).float()
    weights = (deg[src] * deg[dst]).clamp(min=1.0).rsqrt()
    return torch.sparse_coo_tensor(
        edge_index, weights, (num_nodes, num_nodes)
    ).coalesce()


def mean_adjacency(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """Sparse row-normalised A (neighbour mean), no self-loops."""
    src, dst = edge_index[0], edge_index[1]
    deg = torch.bincount(src, minlength=num_nodes).float()
    weights = 1.0 / deg[src].clamp(min=1.0)
    return torch.sparse_coo_tensor(
        edge_index, weights, (num_nodes, num_nodes)
    ).coalesce()


class PlatonovEncoder(nn.Module):
    """The Platonov GNN skeleton with a pluggable block type."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 512,
        num_blocks: int = 5,
        dropout: float = 0.2,
        block: str = "gcn",
    ):
        super().__init__()
        self.adjacency_kind = BLOCKS[block].ADJACENCY
        self.input_linear = nn.Linear(in_dim, hidden_dim)
        self.input_dropout = nn.Dropout(dropout)
        self.act = nn.GELU()
        self.norms = nn.ModuleList(nn.LayerNorm(hidden_dim) for _ in range(num_blocks))
        self.blocks = nn.ModuleList(
            BLOCKS[block](hidden_dim, dropout) for _ in range(num_blocks)
        )
        self.output_norm = nn.LayerNorm(hidden_dim)
        self._adj: torch.Tensor | None = None
        self._adj_num_edges = -1

    def clear_graph_cache(self) -> None:
        """Discard the cached adjacency before evaluating a different graph."""
        self._adj = None
        self._adj_num_edges = -1

    def _get_adj(self, edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor | None:
        if self.adjacency_kind is None:
            return None
        # The graph is fixed within a run; build the normalised operator once.
        if self._adj is None or self._adj_num_edges != edge_index.shape[1]:
            if self.adjacency_kind == "sym-selfloops":
                self._adj = normalized_adjacency(edge_index, num_nodes, True)
            elif self.adjacency_kind == "mean-noloops":
                self._adj = mean_adjacency(edge_index, num_nodes)
            else:
                raise ValueError(f"Unknown adjacency kind {self.adjacency_kind!r}")
            self._adj_num_edges = edge_index.shape[1]
        return self._adj

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        adj = self._get_adj(edge_index, x.shape[0])
        h = self.act(self.input_dropout(self.input_linear(x)))
        for norm, block in zip(self.norms, self.blocks):
            h = h + block(norm(h), adj)
        return self.output_norm(h)
