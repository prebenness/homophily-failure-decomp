"""Shared readout attached to every encoder (proposal, model cut points).

    u_v = Dropout_0.2(GELU(LayerNorm(W_g z_v + b_g)))
    s_v = W_l u_v + b_l

g_eta (classifier body) is the linear map, layer norm, GELU, and dropout;
l_omega (final affine) is the output linear layer. Non-residual so that
A_last remains a refit of only the final affine map.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class SharedReadout(nn.Module):
    def __init__(self, dim: int, num_classes: int, dropout: float = 0.2):
        super().__init__()
        # g_eta
        self.lin = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        # l_omega
        self.out = nn.Linear(dim, num_classes)

    def classifier_features(self, z: torch.Tensor) -> torch.Tensor:
        """u_v: output of the classifier body g_eta."""
        return self.drop(self.act(self.norm(self.lin(z))))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.out(self.classifier_features(z))
