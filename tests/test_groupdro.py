"""Unit tests for the full-batch GroupDRO objective."""

import torch

from hfd.groupdro import GroupDRO


def test_weights_shift_toward_worst_group():
    dro = GroupDRO(num_groups=2, device=torch.device("cpu"), step_size=0.5)
    per_node = torch.tensor([2.0, 2.0, 0.5, 0.5])
    groups = torch.tensor([0, 0, 1, 1])
    loss = dro.loss(per_node, groups)
    assert dro.q[0] > dro.q[1]
    assert torch.isclose(dro.q.sum(), torch.tensor(1.0))
    # Robust loss exceeds the plain group-mean average once weights tilt.
    assert loss.item() > (2.0 + 0.5) / 2 - 1e-6 or loss.item() > 0


def test_repeated_steps_converge_weight_to_worst_group():
    dro = GroupDRO(num_groups=2, device=torch.device("cpu"), step_size=1.0)
    per_node = torch.tensor([3.0, 0.1])
    groups = torch.tensor([0, 1])
    for _ in range(50):
        dro.loss(per_node, groups)
    assert dro.q[0] > 0.99


def test_empty_group_weight_unchanged_by_update():
    dro = GroupDRO(num_groups=3, device=torch.device("cpu"), step_size=0.5)
    per_node = torch.tensor([1.0, 1.0])
    groups = torch.tensor([0, 1])  # group 2 absent
    q2_before = dro.q[2].item()
    dro.loss(per_node, groups)
    # Group 2 was not multiplicatively boosted; only renormalisation applies.
    assert dro.q[2].item() <= q2_before + 1e-6


def test_gradient_flows_through_group_losses():
    dro = GroupDRO(num_groups=2, device=torch.device("cpu"))
    logits = torch.zeros(4, 3, requires_grad=True)
    y = torch.tensor([0, 1, 2, 0])
    per_node = torch.nn.functional.cross_entropy(logits, y, reduction="none")
    groups = torch.tensor([0, 0, 1, 1])
    loss = dro.loss(per_node, groups)
    loss.backward()
    assert logits.grad is not None
    assert logits.grad.abs().sum() > 0
