"""Unit tests for training-loop rules."""

import pytest
import torch

from hfd.training import (
    RecoveryConfig,
    TrainConfig,
    _weighted_mean_loss,
    make_optimizer,
    train_erm,
    train_recovery,
)


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = torch.nn.Linear(4, 2)

    def forward(self, x, edge_index):
        return self.lin(x)


class GroupedModel(TinyModel):
    def param_groups(self, lr, weight_decay):
        return [
            {"params": self.lin.weight, "lr": lr, "weight_decay": weight_decay},
            {"params": self.lin.bias, "lr": lr, "weight_decay": 0.0},
        ]


def _data(n=20):
    torch.manual_seed(0)
    x = torch.randn(n, 4)
    y = (x[:, 0] > 0).long()
    ei = torch.empty((2, 0), dtype=torch.long)
    idx = torch.arange(n)
    return x, ei, y, idx[: n // 2], idx[n // 2 :]


def test_param_groups_respected():
    model = GroupedModel()
    cfg = TrainConfig(optimizer="adam", lr=0.01, weight_decay=0.5)
    opt = make_optimizer(model, cfg)
    assert len(opt.param_groups) == 2
    assert opt.param_groups[0]["weight_decay"] == 0.5
    assert opt.param_groups[1]["weight_decay"] == 0.0


def test_full_budget_always_trained():
    # No early stopping, in any preparation: every run trains its full
    # budget and checkpoint selection alone determines the model.
    x, ei, y, train_idx, val_idx = _data()
    model = TinyModel()
    cfg = TrainConfig(optimizer="adam", lr=5.0, weight_decay=0.0, max_epochs=60)
    result = train_erm(model, x, ei, y, train_idx, val_idx, cfg)
    assert result.epochs_run == 60  # diverging lr still runs the budget
    assert 0 <= result.best_epoch < 60


def test_weighted_mean_loss_is_normalised_once_and_has_gradients():
    logits = torch.tensor(
        [[2.0, 0.0], [0.5, 1.0], [-1.0, 1.0]], requires_grad=True
    )
    labels = torch.tensor([0, 1, 0])
    per_node = torch.nn.functional.cross_entropy(logits, labels, reduction="none")
    weights = torch.tensor([0.5, 1.0, 2.0])

    loss = _weighted_mean_loss(per_node, weights)
    expected = (per_node * weights).sum() / weights.sum()
    assert torch.allclose(loss, expected)
    assert torch.allclose(
        _weighted_mean_loss(per_node, 7 * weights), expected
    )
    assert torch.allclose(
        _weighted_mean_loss(per_node, torch.ones_like(weights)), per_node.mean()
    )
    loss.backward()
    assert logits.grad is not None
    assert logits.grad.abs().sum() > 0


def test_fg_recovery_uses_weighted_objective_without_groupdro(monkeypatch):
    class UnexpectedGroupDRO:
        def __init__(self, *args, **kwargs):
            raise AssertionError("FG-CCDB must not instantiate GroupDRO")

    monkeypatch.setattr("hfd.training.GroupDRO", UnexpectedGroupDRO)
    x, ei, y, train_idx, val_idx = _data()
    model = TinyModel()
    cfg = RecoveryConfig(
        optimizer="adam", lr=0.01, weight_decay=0.0, max_epochs=1
    )
    weights = torch.linspace(0.5, 1.5, train_idx.numel())

    result = train_recovery(
        model,
        "full",
        x,
        ei,
        y,
        train_idx,
        None,
        val_idx,
        None,
        cfg,
        objective="fg-ccdb",
        train_weights=weights,
    )

    assert result.epochs_run == 1
    assert result.history[0]["objective"] == "fg-ccdb"
    assert "weighted_loss" in result.history[0]
    assert "robust_loss" not in result.history[0]


@pytest.mark.parametrize(
    "weights, message",
    [
        (torch.ones(9), "align"),
        (torch.tensor([float("nan")] + [1.0] * 9), "finite"),
        (torch.tensor([float("inf")] + [1.0] * 9), "finite"),
        (torch.tensor([-1.0] + [1.0] * 9), "strictly positive"),
        (torch.tensor([0.0] + [1.0] * 9), "strictly positive"),
    ],
)
def test_fg_recovery_rejects_invalid_weights(weights, message):
    x, ei, y, train_idx, val_idx = _data()
    with pytest.raises(ValueError, match=message):
        train_recovery(
            TinyModel(),
            "full",
            x,
            ei,
            y,
            train_idx,
            None,
            val_idx,
            None,
            RecoveryConfig(max_epochs=1),
            objective="fg-ccdb",
            train_weights=weights,
        )
