"""Focused tests for deterministic DFR refitting."""

import numpy as np
import pytest

from hfd.dfr import (
    DFRConfig,
    DFRSupportError,
    PAPER_C_GRID,
    fit_dfr,
    joint_stratified_split,
)


def _three_class_pool(n_per_cell: int = 6):
    rng = np.random.default_rng(123)
    labels = np.repeat(np.arange(3), 2 * n_per_cell)
    bins = np.tile(np.repeat(np.arange(2), n_per_cell), 3)
    features = rng.normal(scale=0.35, size=(labels.size, 5))
    features[:, :3] += np.eye(3)[labels] * 2.0
    features[:, 3] += bins
    return features, labels, bins


def _small_config(seed: int = 17):
    return DFRConfig(
        c_grid=(0.3, 0.1),
        n_final_fits=3,
        random_seed=seed,
        max_iter=2_000,
    )


def test_default_grid_matches_dfr_paper():
    assert DFRConfig().c_grid == PAPER_C_GRID
    assert PAPER_C_GRID == (1.0, 0.7, 0.3, 0.1, 0.07, 0.03, 0.01)


def test_fit_is_deterministic_balanced_and_multiclass():
    features, labels, bins = _three_class_pool()
    kwargs = {
        "config": _small_config(),
        "expected_classes": np.arange(3),
        "expected_bins": np.arange(2),
    }
    first = fit_dfr(features, labels, bins, **kwargs)
    second = fit_dfr(features, labels, bins, **kwargs)

    np.testing.assert_array_equal(first.classes_, np.arange(3))
    np.testing.assert_allclose(first.coef_, second.coef_, atol=0, rtol=0)
    np.testing.assert_allclose(first.intercept_, second.intercept_, atol=0, rtol=0)
    assert first.diagnostics["split"] == second.diagnostics["split"]
    assert first.diagnostics["solver"] == "saga"
    assert first.diagnostics["single_cpu_thread"] is True

    fit_support = first.diagnostics["support"]["tuning_fit_partition"]
    held_out_support = first.diagnostics["support"]["held_out_partition"]
    assert set(fit_support.values()) == {3}
    assert set(held_out_support.values()) == {3}
    for record in first.diagnostics["final"]["fits"]:
        assert record["per_joint_cell"] == 6
        assert record["sample_size"] == 36

    probabilities = first.predict_proba(features)
    assert probabilities.shape == (36, 3)
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)
    assert first.predict(features).shape == labels.shape


def test_tuning_scaler_has_no_held_out_leakage_and_prediction_is_read_only():
    features, labels, bins = _three_class_pool()
    result = fit_dfr(
        features,
        labels,
        bins,
        config=_small_config(),
        expected_classes=np.arange(3),
        expected_bins=np.arange(2),
    )
    np.testing.assert_allclose(result.scaler_mean_, features.mean(axis=0))
    np.testing.assert_allclose(result.scaler_scale_, features.std(axis=0))
    tuning_scaler = result.diagnostics["scaler"]["tuning"]
    assert tuning_scaler["fit_scope"] == "tuning_fit_partition"
    assert tuning_scaler["n_samples_seen"] == features.shape[0] // 2
    assert tuning_scaler["held_out_data_used"] is False
    assert tuning_scaler["test_data_used"] is False
    cells = tuple((class_value, bin_value) for class_value in range(3) for bin_value in range(2))
    fit_idx, _ = joint_stratified_split(
        labels,
        bins,
        cells=cells,
        tuning_fraction=0.5,
        random_seed=_small_config().random_seed,
    )
    np.testing.assert_allclose(tuning_scaler["mean"], features[fit_idx].mean(axis=0))
    np.testing.assert_allclose(tuning_scaler["scale"], features[fit_idx].std(axis=0))
    assert result.diagnostics["scaler"]["final"] == {
        "fit_scope": "complete_refit_pool",
        "n_samples_seen": features.shape[0],
        "test_data_used": False,
    }

    before = result.to_dict()
    result.predict(np.full((2, features.shape[1]), 1e6))
    assert result.to_dict() == before


def test_missing_joint_cell_fails_strict_support_audit():
    features, labels, bins = _three_class_pool()
    keep = ~((labels == 2) & (bins == 1))
    with pytest.raises(DFRSupportError, match="insufficient joint support"):
        fit_dfr(
            features[keep],
            labels[keep],
            bins[keep],
            config=_small_config(),
            expected_classes=np.arange(3),
            expected_bins=np.arange(2),
        )


def test_single_observation_joint_cell_cannot_be_split():
    features, labels, bins = _three_class_pool()
    target = np.flatnonzero((labels == 2) & (bins == 1))
    keep = np.ones(labels.size, dtype=bool)
    keep[target[1:]] = False
    with pytest.raises(DFRSupportError, match="required >= 2"):
        fit_dfr(
            features[keep],
            labels[keep],
            bins[keep],
            config=_small_config(),
            expected_classes=np.arange(3),
            expected_bins=np.arange(2),
        )
