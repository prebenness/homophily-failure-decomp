"""Model registry: paper-derived encoders plus the shared readout."""

from __future__ import annotations

import torch.nn as nn

from .acmgcn import ACMGCNEncoder
from .gpr import GPRGNNEncoder
from .h2gcn import H2GCNEncoder
from .head import SharedReadout
from .linkx import LINKXEncoder
from .platonov import PlatonovEncoder


class EncoderWithReadout(nn.Module):
    """Full model: graph encoder a_phi followed by the shared readout.

    The frozen representation z_v used in the decomposition is the encoder
    output; the readout is g_eta (classifier body) and l_omega (final affine).
    """

    def __init__(self, encoder: nn.Module, head: SharedReadout):
        super().__init__()
        self.encoder = encoder
        self.head = head

    def forward(self, x, edge_index):
        return self.head(self.encoder(x, edge_index))

    def regularization_loss(self):
        if hasattr(self.encoder, "regularization_loss"):
            return self.encoder.regularization_loss()
        return None

    def param_groups(self, lr: float, weight_decay: float):
        """Optimiser parameter groups; encoders may split them (GPR-GNN)."""
        if hasattr(self.encoder, "param_groups"):
            return self.encoder.param_groups(lr, weight_decay) + [
                {"params": self.head.parameters(), "weight_decay": weight_decay, "lr": lr}
            ]
        return None


def _platonov(block: str):
    def factory(in_dim: int, num_nodes: int, **kwargs):
        return PlatonovEncoder(in_dim, block=block, **kwargs)

    return factory


def _simple(cls):
    def factory(in_dim: int, num_nodes: int, **kwargs):
        return cls(in_dim, **kwargs)

    return factory


ENCODERS = {
    # Platonov-skeleton models (blocks from their modules.py, not PyG layers).
    "mlp": _platonov("ffn"),
    "gcn": _platonov("gcn"),
    "graphsage-mean": _platonov("sage"),
    # Source-default models.
    "h2gcn": _simple(H2GCNEncoder),
    "gpr-gnn": _simple(GPRGNNEncoder),
    "linkx": lambda in_dim, num_nodes, **kw: LINKXEncoder(in_dim, num_nodes, **kw),
    "acm-gcn": _simple(ACMGCNEncoder),
}


def build_model(
    model_name: str,
    in_dim: int,
    num_classes: int,
    model_cfg: dict,
    num_nodes: int | None = None,
) -> EncoderWithReadout:
    if model_name not in ENCODERS:
        raise KeyError(f"Unknown model {model_name!r}; expected one of {tuple(ENCODERS)}")
    encoder = ENCODERS[model_name](in_dim, num_nodes, **model_cfg.get("encoder", {}))
    head = SharedReadout(
        dim=model_cfg.get("z_dim", 512),
        num_classes=num_classes,
        dropout=model_cfg.get("head_dropout", 0.2),
    )
    return EncoderWithReadout(encoder, head)
