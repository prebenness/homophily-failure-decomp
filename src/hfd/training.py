"""Training loops for the atomic run unit.

No model uses early stopping, in any preparation: every run trains its full
step or epoch budget, and checkpoint selection alone determines the model
(defaults doc, rule 5).

A_erm: full-batch training with cross-entropy on the training split.
Checkpoint selection uses the dataset's standard validation metric.

Recovery preparations (A_last, A_head, A_full): either GroupDRO over the fixed
local-homophily-bin groups or direct FG-CCDB importance weighting. A_full
inherits the parent ERM optimiser settings and step budget; A_last/A_head use
the same optimiser at one tenth of the ERM learning rate for ten times the
budget. Validation macro-F1 on the fixed minority-pattern tail selects the
checkpoint only. Frozen modules run in eval mode, so for A_last and A_head the
frozen features are precomputed once and reused every epoch.
"""

from __future__ import annotations

import copy
import random
import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from .groupdro import GroupDRO
from .metrics import macro_f1


@dataclass
class TrainConfig:
    optimizer: str = "adam"
    lr: float = 0.01
    weight_decay: float = 5e-4
    max_epochs: int = 200


@dataclass
class TrainResult:
    best_state: dict
    best_epoch: int
    best_val_metric: float
    epochs_run: int
    seconds: float
    history: list[dict]


def make_optimizer(model: torch.nn.Module, cfg: TrainConfig) -> torch.optim.Optimizer:
    opts = {"adam": torch.optim.Adam, "adamw": torch.optim.AdamW}
    groups = None
    if hasattr(model, "param_groups"):
        groups = model.param_groups(cfg.lr, cfg.weight_decay)
    if groups is None:
        groups = [
            {"params": model.parameters(), "lr": cfg.lr, "weight_decay": cfg.weight_decay}
        ]
    return opts[cfg.optimizer](groups)


def _model_loss(model: torch.nn.Module, data_loss: torch.Tensor) -> torch.Tensor:
    """Add the model's source regularisation term when it defines one."""
    if hasattr(model, "regularization_loss"):
        reg = model.regularization_loss()
        if reg is not None:
            return data_loss + reg
    return data_loss


def _weighted_mean_loss(
    per_node_loss: torch.Tensor, weights: torch.Tensor
) -> torch.Tensor:
    """Return the deterministic importance-weighted empirical risk."""

    return (per_node_loss * weights).sum() / weights.sum()


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


@torch.no_grad()
def evaluate_split(model, x, edge_index, y, idx) -> tuple[float, float]:
    """Return (loss, accuracy) on the given node indices."""
    model.eval()
    logits = model(x, edge_index)
    loss = F.cross_entropy(logits[idx], y[idx]).item()
    acc = (logits[idx].argmax(dim=-1) == y[idx]).float().mean().item()
    return loss, acc


def train_erm(
    model: torch.nn.Module,
    x: torch.Tensor,
    edge_index: torch.Tensor,
    y: torch.Tensor,
    train_idx: torch.Tensor,
    val_idx: torch.Tensor,
    cfg: TrainConfig,
) -> TrainResult:
    optimizer = make_optimizer(model, cfg)
    best_val_metric = -float("inf")
    best_state, best_epoch = None, -1
    history = []
    start = time.time()

    for epoch in range(cfg.max_epochs):
        model.train()
        optimizer.zero_grad()
        logits = model(x, edge_index)
        loss = _model_loss(model, F.cross_entropy(logits[train_idx], y[train_idx]))
        loss.backward()
        optimizer.step()

        val_loss, val_acc = evaluate_split(model, x, edge_index, y, val_idx)
        history.append(
            {"epoch": epoch, "train_loss": loss.item(), "val_loss": val_loss, "val_acc": val_acc}
        )

        # Checkpoint selection: standard validation metric (accuracy).
        if val_acc > best_val_metric:
            best_val_metric = val_acc
            best_epoch = epoch
            best_state = copy.deepcopy(
                {k: v.detach().cpu() for k, v in model.state_dict().items()}
            )

    return TrainResult(
        best_state=best_state,
        best_epoch=best_epoch,
        best_val_metric=best_val_metric,
        epochs_run=len(history),
        seconds=time.time() - start,
        history=history,
    )


@dataclass
class RecoveryConfig:
    """Recovery-preparation settings derived from the parent ERM cell.

    Contract rule: A_full inherits the parent ERM optimiser type, learning
    rate, schedule, and step budget; A_last and A_head use the same optimiser
    and at least the same step budget with the learning rate divided by 10.
    No early stopping in any recovery run.
    """

    optimizer: str = "adamw"
    lr: float = 3e-5
    weight_decay: float = 0.0
    robust_step_size: float = 0.01  # GroupDRO code-release default
    max_epochs: int = 1000

    @staticmethod
    def from_erm(train_cfg: "TrainConfig", preparation: str) -> "RecoveryConfig":
        if preparation == "full":
            lr, budget = train_cfg.lr, train_cfg.max_epochs
        else:
            # lr/10 with 10x the parent budget: the contract requires at
            # least the parent budget, and the debug run showed last/head
            # still improving at the parent budget's edge under lr/10.
            lr, budget = train_cfg.lr / 10, train_cfg.max_epochs * 10
        return RecoveryConfig(
            optimizer=train_cfg.optimizer,
            lr=lr,
            weight_decay=train_cfg.weight_decay,
            max_epochs=budget,
        )


def _reset_module(module: torch.nn.Module) -> None:
    """Re-randomise every parameterised submodule in place."""
    for m in module.modules():
        if hasattr(m, "reset_parameters"):
            m.reset_parameters()


def train_recovery(
    model: torch.nn.Module,
    preparation: str,  # "last" | "head" | "full"
    x: torch.Tensor,
    edge_index: torch.Tensor,
    y: torch.Tensor,
    train_idx: torch.Tensor,
    train_group_ids: torch.Tensor | None,
    val_tail_idx: torch.Tensor,
    num_groups: int | None,
    cfg: RecoveryConfig,
    *,
    objective: str = "groupdro",
    train_weights: torch.Tensor | None = None,
) -> TrainResult:
    """Retrain the preparation under the selected tail-balancing objective.

    The caller prepares `model`: for "last"/"head" it holds the parent ERM
    checkpoint with the retrained part freshly re-initialised; for "full" it
    is a fresh random initialisation.
    """
    device = x.device
    if objective not in ("groupdro", "fg-ccdb"):
        raise ValueError(f"Unknown recovery objective {objective!r}")
    if objective == "groupdro":
        if train_group_ids is None or num_groups is None:
            raise ValueError("GroupDRO requires train_group_ids and num_groups")
        if train_group_ids.shape != train_idx.shape:
            raise ValueError("train_group_ids must align with train_idx")
    else:
        if train_weights is None:
            raise ValueError("FG-CCDB requires train_weights")
        train_weights = train_weights.detach().to(device=device, dtype=x.dtype)
        if train_weights.shape != train_idx.shape:
            raise ValueError("train_weights must align with train_idx")
        if not torch.isfinite(train_weights).all():
            raise ValueError("train_weights must all be finite")
        if torch.any(train_weights <= 0):
            raise ValueError("train_weights must all be strictly positive")

    y_val_tail = y[val_tail_idx].cpu().numpy()
    validation_tail_labels = np.unique(y_val_tail)

    # Freeze and precompute per preparation. Frozen modules run in eval mode.
    if preparation == "last":
        for p in model.parameters():
            p.requires_grad_(False)
        for p in model.head.out.parameters():
            p.requires_grad_(True)
        model.eval()
        with torch.no_grad():
            features = model.head.classifier_features(model.encoder(x, edge_index))
        trainable = model.head.out
        forward_train = lambda: trainable(features)  # noqa: E731
        forward_eval = forward_train
    elif preparation == "head":
        for p in model.parameters():
            p.requires_grad_(False)
        for p in model.head.parameters():
            p.requires_grad_(True)
        model.eval()
        with torch.no_grad():
            features = model.encoder(x, edge_index)
        trainable = model.head
        forward_train = lambda: trainable(features)  # noqa: E731
        forward_eval = forward_train
    elif preparation == "full":
        for p in model.parameters():
            p.requires_grad_(True)
        trainable = model
        forward_train = lambda: model(x, edge_index)  # noqa: E731
        forward_eval = forward_train
    else:
        raise ValueError(f"Unknown preparation {preparation!r}")

    opts = {"adam": torch.optim.Adam, "adamw": torch.optim.AdamW}
    groups = None
    if preparation == "full" and hasattr(model, "param_groups"):
        # A_full inherits the parent's optimiser settings, including any
        # source-specific parameter grouping (e.g. GPR-GNN propagation
        # weights train without weight decay).
        groups = model.param_groups(cfg.lr, cfg.weight_decay)
    if groups is None:
        groups = [
            {
                "params": [p for p in trainable.parameters() if p.requires_grad],
                "lr": cfg.lr,
                "weight_decay": cfg.weight_decay,
            }
        ]
    optimizer = opts[cfg.optimizer](groups)
    dro = (
        GroupDRO(num_groups, device, step_size=cfg.robust_step_size)
        if objective == "groupdro"
        else None
    )

    best_val_metric = -float("inf")
    best_state, best_epoch = None, -1
    history = []
    start = time.time()

    for epoch in range(cfg.max_epochs):
        trainable.train()
        optimizer.zero_grad()
        logits = forward_train()
        per_node = F.cross_entropy(logits[train_idx], y[train_idx], reduction="none")
        if objective == "groupdro":
            loss = dro.loss(per_node, train_group_ids)
        else:
            loss = _weighted_mean_loss(per_node, train_weights)
        if preparation == "full":
            # A_full inherits the full ERM protocol, including any source
            # regularisation term; last/head train on frozen features where
            # the regularised encoder weights are constants.
            loss = _model_loss(model, loss)
        loss.backward()
        optimizer.step()

        # Checkpoint selection: validation macro-F1 on the fixed tail.
        trainable.eval()
        with torch.no_grad():
            val_pred = forward_eval()[val_tail_idx].argmax(dim=-1).cpu().numpy()
        val_tail_f1 = macro_f1(y_val_tail, val_pred, validation_tail_labels)
        history_row = {
            "epoch": epoch,
            "objective": objective,
            "objective_loss": loss.item(),
            "val_tail_macro_f1": val_tail_f1,
        }
        if objective == "groupdro":
            # Preserve the existing history key used by the registered runs.
            history_row["robust_loss"] = loss.item()
        else:
            history_row["weighted_loss"] = loss.item()
        history.append(history_row)
        if val_tail_f1 > best_val_metric:
            best_val_metric = val_tail_f1
            best_epoch = epoch
            best_state = copy.deepcopy(
                {k: v.detach().cpu() for k, v in model.state_dict().items()}
            )

    return TrainResult(
        best_state=best_state,
        best_epoch=best_epoch,
        best_val_metric=best_val_metric,
        epochs_run=len(history),
        seconds=time.time() - start,
        history=history,
    )
