"""GroupDRO objective over the fixed local-homophily-bin groups.

Follows the original code release of Sagawa et al. (group_dro repository)
with its default settings: multiplicative weight update on group losses with
robust step size 0.01 and no generalisation adjustment. Full-batch variant:
each step computes the mean cross-entropy per group, updates the group
weights, and returns the weighted sum.
"""

from __future__ import annotations

import torch

DEFAULT_ROBUST_STEP_SIZE = 0.01


class GroupDRO:
    def __init__(self, num_groups: int, device: torch.device,
                 step_size: float = DEFAULT_ROBUST_STEP_SIZE):
        self.step_size = step_size
        self.q = torch.ones(num_groups, device=device) / num_groups

    def loss(self, per_node_loss: torch.Tensor, group_ids: torch.Tensor) -> torch.Tensor:
        """Robust loss for one full-batch step.

        Args:
            per_node_loss: (M,) unreduced losses for the trainable nodes.
            group_ids: (M,) group index in [0, num_groups) for each node.
        """
        num_groups = self.q.numel()
        counts = torch.bincount(group_ids, minlength=num_groups).float()
        sums = torch.zeros(num_groups, device=per_node_loss.device).index_add_(
            0, group_ids, per_node_loss
        )
        group_loss = sums / counts.clamp(min=1.0)
        present = counts > 0

        with torch.no_grad():
            self.q[present] = self.q[present] * torch.exp(
                self.step_size * group_loss[present].detach()
            )
            self.q = self.q / self.q.sum()

        return (self.q * group_loss).sum()
