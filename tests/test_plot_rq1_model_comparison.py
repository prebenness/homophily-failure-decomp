from pathlib import Path
import runpy
import sys
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pytest


pytest.importorskip("seaborn")

SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "plot_rq1_model_comparison.py"
)
sys.path.insert(0, str(SCRIPT.parent))
NAMESPACE = runpy.run_path(str(SCRIPT))


def make_data(
    accuracy_units: np.ndarray,
    support: np.ndarray,
) -> SimpleNamespace:
    n_bins = support.size
    edges = np.arange(n_bins + 1, dtype=np.float64) / n_bins
    return SimpleNamespace(
        edges=edges,
        centres=(edges[:-1] + edges[1:]) / 2.0,
        accuracy_units=accuracy_units,
        mean_run_support=support,
    )


def test_comparison_uses_ten_bins_by_default() -> None:
    assert NAMESPACE["DEFAULT_COMPARISON_N_BINS"] == 10


def test_complete_unit_rule_omits_only_sparse_bin() -> None:
    accuracy_units = np.ones((10, 10), dtype=np.float64)
    accuracy_units[3:, 8] = np.nan
    data = make_data(accuracy_units, np.arange(1, 11, dtype=np.float64))

    actual = NAMESPACE["complete_unit_display_mask"](data)

    expected = np.ones(10, dtype=np.bool_)
    expected[8] = False
    np.testing.assert_array_equal(actual, expected)


def test_support_panel_is_logarithmic_and_retains_partially_supported_bar() -> None:
    accuracy_units = np.ones((10, 10), dtype=np.float64)
    accuracy_units[3:, 8] = np.nan
    support = np.array(
        [5023.8, 21.1, 102.5, 261.5, 11.5, 188.2, 38.8, 2.8, 0.4, 15.4]
    )
    data = make_data(accuracy_units, support)
    support_display_mask = NAMESPACE["positive_support_display_mask"](data)
    figure, axis = plt.subplots()

    try:
        visible = NAMESPACE["plot_support_panel"](
            axis, data, support_display_mask
        )
        assert axis.get_yscale() == "log"
        assert len(axis.patches) == 10
        assert visible[8]
        np.testing.assert_allclose(
            axis.get_yticks(), [0.1, 1.0, 10.0, 100.0, 1_000.0]
        )
        assert any(
            patch.get_x() + patch.get_width() / 2.0
            == pytest.approx(data.centres[8])
            for patch in axis.patches
        )
    finally:
        plt.close(figure)


def test_support_panel_omits_zero_support_bar() -> None:
    accuracy_units = np.ones((10, 12), dtype=np.float64)
    support = np.arange(1, 13, dtype=np.float64)
    support[10] = 0.0
    data = make_data(accuracy_units, support)
    support_display_mask = NAMESPACE["positive_support_display_mask"](data)
    figure, axis = plt.subplots()

    try:
        visible = NAMESPACE["plot_support_panel"](
            axis, data, support_display_mask
        )
        assert len(axis.patches) == 11
        assert not visible[10]
        assert all(
            patch.get_x() + patch.get_width() / 2.0
            != pytest.approx(data.centres[10])
            for patch in axis.patches
        )
    finally:
        plt.close(figure)


def test_confidence_band_respects_accuracy_display_mask() -> None:
    data = make_data(
        np.ones((10, 5), dtype=np.float64),
        np.ones(5, dtype=np.float64),
    )
    data.accuracy_ci_low = np.array([0.2, 0.3, 0.4, 0.5, 0.6])
    data.accuracy_ci_high = np.array([0.4, 0.5, 0.6, 0.7, 0.8])
    display_mask = np.array([True, True, False, True, True])
    figure, axis = plt.subplots()

    try:
        NAMESPACE["plot_confidence_band"](
            axis, data, display_mask, (0.1, 0.2, 0.3), alpha=0.2
        )
        assert len(axis.collections) == 1
        assert axis.collections[0].get_alpha() == pytest.approx(0.2)
        paths = axis.collections[0].get_paths()
        assert len(paths) == 2
    finally:
        plt.close(figure)


def test_omission_record_retains_raw_support_and_unit_count() -> None:
    accuracy_units = np.ones((10, 10), dtype=np.float64)
    accuracy_units[3:, 8] = np.nan
    support = np.array(
        [5023.8, 21.1, 102.5, 261.5, 11.5, 188.2, 38.8, 2.8, 0.4, 15.4]
    )
    data = make_data(accuracy_units, support)
    display_mask = NAMESPACE["complete_unit_display_mask"](data)

    records = NAMESPACE["omitted_bin_records"](data, display_mask)

    assert records == [
        {
            "interval": "[0.8, 0.9)",
            "mean_test_nodes_per_run": pytest.approx(0.4),
            "independent_units_with_test_nodes": 3,
            "independent_unit_count": 10,
            "omits_accuracy_point": True,
            "omits_support_bar": False,
            "reason": NAMESPACE["DISPLAY_OMISSION_REASON"],
        }
    ]
