"""Regression tests for the equation-defined FG-CCDB importance weights."""

import numpy as np
import pytest

from hfd.fg_ccdb import compute_fg_ccdb_weights


def _observations(cell_counts):
    bins = []
    labels = []
    for bin_index, row in enumerate(cell_counts):
        for class_index, count in enumerate(row):
            bins.extend([bin_index] * count)
            labels.extend([class_index] * count)
    return np.asarray(labels), np.asarray(bins)


def test_weights_recover_product_of_empirical_marginals():
    cell_counts = np.asarray([[1, 3], [2, 2], [5, 1]])
    labels, bins = _observations(cell_counts)

    result = compute_fg_ccdb_weights(
        labels, bins, num_classes=2, num_bins=3
    )
    diagnostics = result.diagnostics

    expected_cell_weights = np.asarray(
        [[16 / 7, 4 / 7], [8 / 7, 6 / 7], [24 / 35, 18 / 7]]
    )
    expected_joint = np.asarray(
        [[8 / 49, 6 / 49], [8 / 49, 6 / 49], [12 / 49, 9 / 49]]
    )
    assert np.allclose(diagnostics["cell_weights"], expected_cell_weights)
    assert np.isclose(result.weights.sum(), 14.0)
    assert np.allclose(
        diagnostics["balance_audit"]["weighted_joint"], expected_joint
    )
    assert np.allclose(
        diagnostics["balance_audit"]["weighted_bin_marginal"], [2 / 7, 2 / 7, 3 / 7]
    )
    assert np.allclose(
        diagnostics["balance_audit"]["weighted_class_marginal"], [4 / 7, 3 / 7]
    )
    assert np.isclose(
        diagnostics["weight_summary"]["effective_sample_size"], 12005 / 1179
    )
    assert diagnostics["balance_audit"]["max_abs_joint_target_error"] < 1e-15


def test_only_selected_training_observations_determine_weights():
    labels = np.asarray([0, 1, 1, 0, 0, 1, 1, 0])
    bins = np.asarray([0, 0, 1, 1, 0, 1, 0, 1])
    train_idx = np.asarray([6, 5, 4, 3])

    original = compute_fg_ccdb_weights(
        labels[train_idx], bins[train_idx], num_classes=2, num_bins=2
    )
    changed_labels = labels.copy()
    changed_bins = bins.copy()
    outside = np.setdiff1d(np.arange(labels.size), train_idx)
    changed_labels[outside] = 1 - changed_labels[outside]
    changed_bins[outside] = 1 - changed_bins[outside]
    changed = compute_fg_ccdb_weights(
        changed_labels[train_idx],
        changed_bins[train_idx],
        num_classes=2,
        num_bins=2,
    )

    assert np.array_equal(
        original.diagnostics["cell_counts"], changed.diagnostics["cell_counts"]
    )
    assert np.allclose(original.weights, changed.weights)


def test_undefined_bins_are_rejected_until_caller_excludes_them():
    labels = np.asarray([0, 0, 1, 1, 0])
    bins = np.asarray([0, 1, 0, 1, -1])
    with pytest.raises(ValueError, match="bin_ids outside"):
        compute_fg_ccdb_weights(labels, bins, num_classes=2, num_bins=2)

    eligible = bins >= 0
    result = compute_fg_ccdb_weights(
        labels[eligible], bins[eligible], num_classes=2, num_bins=2
    )
    assert result.diagnostics["n_training_nodes"] == 4


def test_empty_active_cell_is_an_infeasible_target():
    labels = np.asarray([0, 0, 1, 1])
    bins = np.asarray([0, 0, 1, 1])
    with pytest.raises(ValueError, match="target is infeasible"):
        compute_fg_ccdb_weights(labels, bins, num_classes=2, num_bins=2)


def test_wholly_absent_declared_level_has_zero_target_marginal():
    labels = np.asarray([0, 1, 0, 1])
    bins = np.asarray([0, 0, 1, 1])
    result = compute_fg_ccdb_weights(
        labels, bins, num_classes=3, num_bins=3
    )
    assert result.diagnostics["class_counts"] == [2, 2, 0]
    assert result.diagnostics["bin_counts"] == [2, 2, 0]
    assert np.all(result.weights > 0)
