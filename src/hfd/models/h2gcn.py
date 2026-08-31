"""H2GCN-2 encoder, ported from the official TensorFlow code (GemsLab/H2GCN).

network_setup M73-R-T1-G-V-T2-G-V-C1-C2-D0.5 decodes (models/__init__.py
parse_network_setup + H2GCN.py) to:

    h0 = ReLU(X W1)                      # dense 73, no bias, tagged "1"
    h1 = [A1n h0 , A2n h0]               # GCN layer over hop mats + flatten,
                                         # tagged "2"        (width 146)
    h2 = [A1n h1 , A2n h1]               #                    (width 292)
    z  = Dropout_0.5([h2, h0, h1])       # C1, C2 concat      (width 511)

Preprocessing (H2GCN.py preprocessing_data + datasets/_dataset.py):
row-normalised features; self-loops removed; A1 = exact-distance-1 indicator,
A2 = exact-distance-2 indicator (nhoodSplit), each symmetrically normalised
D^-1/2 M D^-1/2 on its own degrees.

Adaptation per the defaults doc (rule 4 exception): internal width 73 so the
native 7-block concatenation is z_v in R^511; the source's final dense MO is
the classifier and is removed — the shared readout attaches directly to the
concatenation. L2 regularisation 5e-4 applies to the dense kernel, exposed
via regularization_loss() (TF kernel_regularizer: loss += l2 * sum(W^2)).
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F


def h2gcn_hop_matrices(
    edge_index: np.ndarray, num_nodes: int
) -> tuple[sp.csr_matrix, sp.csr_matrix]:
    """Exact-distance-1 and exact-distance-2 indicator matrices, each
    symmetrically normalised on its own degrees (nhoodSplit + SYM_NORMALIZED).
    Expects a simple undirected edge_index without self-loops."""
    a = sp.csr_matrix(
        (np.ones(edge_index.shape[1]), (edge_index[0], edge_index[1])),
        shape=(num_nodes, num_nodes),
    )
    a = (a > 0).astype(np.float64)
    eye = sp.eye(num_nodes, format="csr")
    reach1 = ((a + eye) > 0).astype(np.float64)  # within distance 1
    reach2 = ((reach1 @ reach1) > 0).astype(np.float64)  # within distance 2
    hop1 = reach1 - eye
    hop2 = reach2 - reach1.minimum(reach2)

    def sym_norm(m: sp.spmatrix) -> sp.csr_matrix:
        m = sp.csr_matrix(m)
        deg = np.asarray(m.sum(axis=1)).ravel()
        with np.errstate(divide="ignore"):
            dinv = np.power(deg, -0.5)
        dinv[np.isinf(dinv)] = 0.0
        d = sp.diags(dinv)
        return sp.csr_matrix(d @ m @ d)

    return sym_norm(hop1), sym_norm(hop2)


def _to_torch_sparse(m: sp.csr_matrix, device, dtype) -> torch.Tensor:
    coo = m.tocoo()
    indices = torch.tensor(np.array([coo.row, coo.col]), dtype=torch.long)
    values = torch.tensor(coo.data, dtype=dtype)
    return torch.sparse_coo_tensor(indices, values, coo.shape).coalesce().to(device)


def prepare_h2gcn_operators(
    edge_index: torch.Tensor | np.ndarray,
    num_nodes: int,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Construct H2GCN's one-hop and two-hop operators for one graph.

    The returned sparse tensors are independent of model parameters and may
    therefore be shared by every checkpoint evaluated on the same graph.
    Callers should prepare them once, after selecting the target device, and
    install them with :meth:`H2GCNEncoder.set_graph_cache`.
    """
    if num_nodes < 0:
        raise ValueError("num_nodes must be non-negative")
    if isinstance(edge_index, torch.Tensor):
        edge_array = edge_index.detach().cpu().numpy()
        target_device = edge_index.device if device is None else torch.device(device)
    else:
        edge_array = np.asarray(edge_index)
        target_device = torch.device("cpu") if device is None else torch.device(device)
    if edge_array.ndim != 2 or edge_array.shape[0] != 2:
        raise ValueError("edge_index must have shape [2, num_edges]")
    if not np.issubdtype(edge_array.dtype, np.integer):
        raise ValueError("edge_index must contain integer node indices")
    edge_array = edge_array.astype(np.int64, copy=False)
    if edge_array.size and (
        int(edge_array.min()) < 0 or int(edge_array.max()) >= num_nodes
    ):
        raise ValueError("edge_index contains an out-of-range node index")

    hop1, hop2 = h2gcn_hop_matrices(edge_array, num_nodes)
    return (
        _to_torch_sparse(hop1, target_device, dtype),
        _to_torch_sparse(hop2, target_device, dtype),
    )


def row_normalize(x: torch.Tensor) -> torch.Tensor:
    """Source row_normalize_features: x_i / sum(x_i), 1/0 -> 0."""
    rowsum = x.sum(dim=1, keepdim=True)
    inv = torch.where(rowsum != 0, 1.0 / rowsum, torch.zeros_like(rowsum))
    return x * inv


class H2GCNEncoder(nn.Module):
    L2_WEIGHT = 5e-4

    def __init__(self, in_dim: int, hidden_dim: int = 73, dropout: float = 0.5):
        super().__init__()
        self.embed = nn.Linear(in_dim, hidden_dim, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.out_dim = 7 * hidden_dim  # native concatenation width (511 at 73)
        self._adj: tuple[torch.Tensor, torch.Tensor] | None = None
        self._adj_num_edges = -1

    def regularization_loss(self) -> torch.Tensor:
        return self.L2_WEIGHT * self.embed.weight.pow(2).sum()

    def clear_graph_cache(self) -> None:
        """Discard operators before evaluating a different graph."""
        self._adj = None
        self._adj_num_edges = -1

    def set_graph_cache(
        self,
        operators: tuple[torch.Tensor, torch.Tensor],
        *,
        num_edges: int,
    ) -> None:
        """Install operators prepared once for reuse across checkpoints.

        The tensors are intentionally retained by reference rather than
        copied, so multiple checkpoint models can share the same graph data.
        Install them after moving the model and operators to their target
        device.
        """
        if len(operators) != 2:
            raise ValueError("operators must contain one-hop and two-hop tensors")
        hop1, hop2 = operators
        for name, operator in (("hop1", hop1), ("hop2", hop2)):
            if not isinstance(operator, torch.Tensor) or not operator.is_sparse:
                raise TypeError(f"{name} must be a sparse COO tensor")
            if operator.ndim != 2 or operator.shape[0] != operator.shape[1]:
                raise ValueError(f"{name} must be a square matrix")
        if hop1.shape != hop2.shape:
            raise ValueError("one-hop and two-hop operators must have equal shapes")
        if hop1.device != hop2.device or hop1.dtype != hop2.dtype:
            raise ValueError(
                "one-hop and two-hop operators must share device and dtype"
            )
        if num_edges < 0:
            raise ValueError("num_edges must be non-negative")
        self._adj = operators
        self._adj_num_edges = int(num_edges)

    def _get_adjs(self, edge_index: torch.Tensor, num_nodes: int):
        if self._adj is None or self._adj_num_edges != edge_index.shape[1]:
            self._adj = prepare_h2gcn_operators(
                edge_index,
                num_nodes,
                device=edge_index.device,
                dtype=torch.float32,
            )
            self._adj_num_edges = edge_index.shape[1]
        return self._adj

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        a1, a2 = self._get_adjs(edge_index, x.shape[0])
        x = row_normalize(x)
        h0 = F.relu(self.embed(x))
        h1 = torch.cat([torch.sparse.mm(a1, h0), torch.sparse.mm(a2, h0)], dim=1)
        h2 = torch.cat([torch.sparse.mm(a1, h1), torch.sparse.mm(a2, h1)], dim=1)
        combined = torch.cat([h2, h0, h1], dim=1)
        return self.dropout(combined)
