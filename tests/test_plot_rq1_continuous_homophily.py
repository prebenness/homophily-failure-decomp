from pathlib import Path
import runpy
import sys

import numpy as np
import pytest


pytest.importorskip("seaborn")

SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "plot_rq1_continuous_homophily.py"
)
NAMESPACE = runpy.run_path(str(SCRIPT))
SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_ROOT))
from hfd.metrics import macro_f1  # noqa: E402


def test_assign_bins_obeys_left_closed_intervals() -> None:
    edges = np.arange(21, dtype=np.float64) / 20
    values = np.array([0.0, 0.049, 0.05, 0.6, 1.0, np.nan])

    actual = NAMESPACE["assign_bins"](values, edges)

    np.testing.assert_array_equal(actual, [0, 0, 1, 12, 19, -1])


def test_true_class_margin_uses_best_wrong_probability() -> None:
    probabilities = np.array([[0.6, 0.3, 0.1], [0.7, 0.2, 0.1]])
    logits = np.log(probabilities)
    labels = np.array([0, 1])

    actual = NAMESPACE["true_class_margin"](logits, labels)

    np.testing.assert_allclose(actual, [0.3, -0.5], atol=1e-12)


def test_accuracy_units_average_seeds_within_split() -> None:
    run_accuracy = np.array([[0.2], [0.4], [0.8], [1.0]])
    record_type = NAMESPACE["RunRecord"]
    records = [
        record_type("a", "split-0", 1, Path("a"), ""),
        record_type("b", "split-0", 2, Path("b"), ""),
        record_type("c", "split-1", 1, Path("c"), ""),
        record_type("d", "split-1", 2, Path("d"), ""),
    ]

    units, definition = NAMESPACE["make_accuracy_units"](run_accuracy, records)

    np.testing.assert_allclose(units, [[0.3], [0.9]])
    assert definition == "2 split-level means, each averaging 2 seed runs"


def test_macro_f1_matches_authoritative_true_label_support_definition() -> None:
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 2, 2, 1])
    labels = np.unique(y_true)

    actual = NAMESPACE["macro_f1_true_label_support"](y_true, y_pred)
    expected = macro_f1(y_true, y_pred, labels)

    assert actual == pytest.approx(expected)
    assert actual == pytest.approx(2.0 / 3.0)


def test_macro_f1_units_average_seeds_within_split() -> None:
    run_macro_f1 = np.array([[0.1], [0.3], [0.7], [0.9]])
    record_type = NAMESPACE["RunRecord"]
    records = [
        record_type("a", "split-0", 1, Path("a"), ""),
        record_type("b", "split-0", 2, Path("b"), ""),
        record_type("c", "split-1", 1, Path("c"), ""),
        record_type("d", "split-1", 2, Path("d"), ""),
    ]

    units, definition = NAMESPACE["make_macro_f1_units"](
        run_macro_f1, records
    )

    np.testing.assert_allclose(units, [[0.2], [0.8]])
    assert definition == "2 split-level means, each averaging 2 seed runs"


def test_nanmean_preserves_an_empty_bin_without_warning() -> None:
    values = np.array([[0.2, np.nan], [0.4, np.nan]])

    actual = NAMESPACE["nanmean_preserving_empty"](values, axis=0)

    np.testing.assert_allclose(actual[0], 0.3)
    assert np.isnan(actual[1])
