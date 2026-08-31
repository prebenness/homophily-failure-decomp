"""FG-CCDB weights for a discrete nuisance variable and discrete label.

For empirical training counts ``n_by`` in nuisance-bin ``b`` and class ``y``,
the per-observation importance ratio

    w_by = n_b * n_y / (N * n_by)

changes the empirical joint distribution to the product of its marginals.
The function below is deliberately strict about support: if a bin and a class
both have positive marginal count but their joint cell is empty, the requested
product distribution cannot be obtained by reweighting the observed samples.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class FGCCDBWeights:
    """Per-observation weights and their auditable empirical diagnostics."""

    weights: np.ndarray
    diagnostics: dict


def _integer_vector(name: str, values: np.ndarray) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional; got shape {array.shape}")
    if array.size == 0:
        raise ValueError(f"{name} must not be empty")
    if not np.issubdtype(array.dtype, np.integer):
        if not np.all(np.isfinite(array)) or not np.all(array == np.floor(array)):
            raise ValueError(f"{name} must contain integer values")
    return array.astype(np.int64, copy=False)


def compute_fg_ccdb_weights(
    labels: np.ndarray,
    bin_ids: np.ndarray,
    *,
    num_classes: int,
    num_bins: int,
) -> FGCCDBWeights:
    """Return direct importance weights targeting ``P(bin) P(label)``.

    ``labels`` and ``bin_ids`` must already be restricted to the training
    observations on which the loss is evaluated. Bin ids below zero therefore
    need to be removed by the caller before this function is used.
    """

    y = _integer_vector("labels", labels)
    b = _integer_vector("bin_ids", bin_ids)
    if y.shape != b.shape:
        raise ValueError(
            f"labels and bin_ids must have the same shape; got {y.shape} and {b.shape}"
        )
    if num_classes <= 0 or num_bins <= 0:
        raise ValueError("num_classes and num_bins must both be positive")
    if np.any((y < 0) | (y >= num_classes)):
        bad = np.unique(y[(y < 0) | (y >= num_classes)]).tolist()
        raise ValueError(f"labels outside [0, {num_classes}): {bad}")
    if np.any((b < 0) | (b >= num_bins)):
        bad = np.unique(b[(b < 0) | (b >= num_bins)]).tolist()
        raise ValueError(f"bin_ids outside [0, {num_bins}): {bad}")

    n = int(y.size)
    cell_counts = np.zeros((num_bins, num_classes), dtype=np.int64)
    np.add.at(cell_counts, (b, y), 1)
    bin_counts = cell_counts.sum(axis=1)
    class_counts = cell_counts.sum(axis=0)

    active_target = (bin_counts[:, None] > 0) & (class_counts[None, :] > 0)
    unsupported = np.argwhere(active_target & (cell_counts == 0))
    if unsupported.size:
        cells = [
            {"bin": int(bin_index), "class": int(class_index)}
            for bin_index, class_index in unsupported
        ]
        raise ValueError(
            "FG-CCDB product-marginal target is infeasible because active "
            f"bin/class cells are empty: {cells}"
        )

    cell_weights = np.zeros_like(cell_counts, dtype=np.float64)
    observed = cell_counts > 0
    numerator = bin_counts[:, None] * class_counts[None, :]
    cell_weights[observed] = (
        numerator[observed] / (float(n) * cell_counts[observed])
    )
    weights = cell_weights[b, y]

    weight_sum = float(weights.sum(dtype=np.float64))
    if not np.isfinite(weights).all() or weight_sum <= 0.0:
        raise RuntimeError("FG-CCDB produced non-finite weights or zero total weight")

    weighted_cell_mass = np.zeros_like(cell_counts, dtype=np.float64)
    np.add.at(weighted_cell_mass, (b, y), weights)
    weighted_joint = weighted_cell_mass / weight_sum
    empirical_bin = bin_counts.astype(np.float64) / n
    empirical_class = class_counts.astype(np.float64) / n
    target_joint = np.outer(empirical_bin, empirical_class)
    weighted_bin = weighted_joint.sum(axis=1)
    weighted_class = weighted_joint.sum(axis=0)
    ess = weight_sum**2 / float(np.square(weights).sum(dtype=np.float64))

    diagnostics = {
        "name": "fg-ccdb",
        "spurious_variable": "fixed local-homophily bin",
        "target": "empirical P(bin) * empirical P(label) on eligible training nodes",
        "weight_formula": "n_bin * n_class / (N * n_bin_class)",
        "loss": "sum(weight * per-node cross-entropy) / sum(weight)",
        "n_training_nodes": n,
        "num_bins": int(num_bins),
        "num_classes": int(num_classes),
        "cell_counts": cell_counts.tolist(),
        "bin_counts": bin_counts.tolist(),
        "class_counts": class_counts.tolist(),
        "cell_weights": cell_weights.tolist(),
        "support_complete_for_active_marginals": True,
        "weight_summary": {
            "minimum": float(weights.min()),
            "median": float(np.quantile(weights, 0.5)),
            "p95": float(np.quantile(weights, 0.95)),
            "p99": float(np.quantile(weights, 0.99)),
            "maximum": float(weights.max()),
            "mean": float(weights.mean(dtype=np.float64)),
            "sum": weight_sum,
            "effective_sample_size": float(ess),
            "effective_sample_size_fraction": float(ess / n),
        },
        "balance_audit": {
            "empirical_bin_marginal": empirical_bin.tolist(),
            "empirical_class_marginal": empirical_class.tolist(),
            "target_joint": target_joint.tolist(),
            "weighted_joint": weighted_joint.tolist(),
            "weighted_bin_marginal": weighted_bin.tolist(),
            "weighted_class_marginal": weighted_class.tolist(),
            "max_abs_joint_target_error": float(
                np.max(np.abs(weighted_joint - target_joint))
            ),
            "max_abs_bin_marginal_error": float(
                np.max(np.abs(weighted_bin - empirical_bin))
            ),
            "max_abs_class_marginal_error": float(
                np.max(np.abs(weighted_class - empirical_class))
            ),
        },
    }
    return FGCCDBWeights(weights=weights, diagnostics=diagnostics)
