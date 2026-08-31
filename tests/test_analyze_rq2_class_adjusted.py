from pathlib import Path
import runpy

import numpy as np
import pytest


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "analyze_rq2_within_class.py"
)
NAMESPACE = runpy.run_path(str(SCRIPT))


def test_adjustment_uses_whole_class_proportions() -> None:
    result = NAMESPACE["class_adjusted_metrics"](
        np.array([0, 1]),
        np.array([[8, 2, 0], [1, 1, 0]]),
        np.array([[1, 1, 0], [0, 2, 0]]),
    )

    assert result["reference_weights"] == pytest.approx([5 / 6, 1 / 6])
    assert result["whole_accuracy"] == pytest.approx(3 / 4)
    assert result["adjusted_tail_accuracy"] == pytest.approx(7 / 12)
    assert result["adjusted_accuracy_gap"] == pytest.approx(1 / 6)
    assert result["whole_macro_f1"] == pytest.approx(59 / 95)
    assert result["adjusted_tail_macro_f1"] == pytest.approx(5 / 9)
    assert result["adjusted_macro_f1_gap"] == pytest.approx(56 / 855)


def test_predictions_outside_tail_classes_are_false_negatives() -> None:
    result = NAMESPACE["class_adjusted_metrics"](
        np.array([0]),
        np.array([[8, 0, 2]]),
        np.array([[5, 0, 5]]),
    )

    assert result["whole_accuracy"] == pytest.approx(0.8)
    assert result["adjusted_tail_accuracy"] == pytest.approx(0.5)
    assert result["whole_macro_f1"] == pytest.approx(8 / 9)
    assert result["adjusted_tail_macro_f1"] == pytest.approx(2 / 3)


def test_adjustment_rejects_a_missing_tail_row() -> None:
    with pytest.raises(ValueError, match="whole and tail support"):
        NAMESPACE["class_adjusted_metrics"](
            np.array([0, 1]),
            np.array([[3, 0], [0, 4]]),
            np.array([[2, 0], [0, 0]]),
        )


def test_adjustment_accepts_a_class_absent_from_non_tail() -> None:
    result = NAMESPACE["class_adjusted_metrics"](
        np.array([0, 1]),
        np.array([[3, 0], [1, 3]]),
        np.array([[3, 0], [0, 1]]),
    )

    assert result["reference_weights"] == pytest.approx([3 / 7, 4 / 7])
    assert np.isfinite(result["adjusted_macro_f1_gap"])


def test_within_class_support_requires_nodes_on_both_sides() -> None:
    rows = [
        {"class": 2, "n_tail_test": 1, "n_non_tail_test": 5},
        {"class": 0, "n_tail_test": 5, "n_non_tail_test": 0},
        {"class": 1, "n_tail_test": 2, "n_non_tail_test": 3},
    ]

    retained = NAMESPACE["within_class_supported_rows"](rows)

    assert [row["class"] for row in retained] == [1, 2]


def test_within_class_accuracy_contrast_weights_classes_equally() -> None:
    rows = [
        {
            "n_tail_test": 7,
            "contrast_non_tail_minus_tail": 0.20,
        },
        {
            "n_tail_test": 100,
            "contrast_non_tail_minus_tail": -0.10,
        },
    ]

    assert NAMESPACE["mean_within_class_accuracy_contrast"](rows) == (
        pytest.approx(0.05)
    )


def test_within_class_bootstrap_preserves_online_seed() -> None:
    assert NAMESPACE["within_class_bootstrap_seed"](
        "pubmed", "mlp", "erm"
    ) == 9912105849082569820


def test_within_class_bootstrap_matches_online_interval() -> None:
    values = np.arange(30, dtype=np.float64) / 100

    assert NAMESPACE["within_class_bootstrap_interval"](
        values, "pubmed", "mlp", "erm"
    ) == pytest.approx([0.11564166666666667, 0.17333333333333331])


def test_within_class_latex_cell_uses_its_own_interval() -> None:
    row = {
        "mean_class_macro_accuracy_contrast": -0.009,
        "bootstrap_ci_lower_class_macro_accuracy": -0.024,
        "bootstrap_ci_upper_class_macro_accuracy": 0.006,
        "class_macro_accuracy_ci_excludes_zero": False,
    }

    assert NAMESPACE["latex_cell"](row, "within_accuracy") == (
        r"\uncertaingaincell{-0.009}{[-0.024,0.006]}"
    )


def test_latex_cell_uses_adjusted_interval_for_shading() -> None:
    row = {
        "mean_adjusted_accuracy_gap": 0.1,
        "bootstrap_ci_lower_adjusted_accuracy_gap": -0.01,
        "bootstrap_ci_upper_adjusted_accuracy_gap": 0.2,
        "adjusted_accuracy_gap_ci_excludes_zero": False,
        "erm_accuracy_gap_ci_excludes_zero": True,
        "erm_tail_gap_ci_excludes_zero": True,
    }

    assert NAMESPACE["latex_cell"](row, "accuracy") == (
        r"\uncertaingaincell{+0.100}{[-0.010,0.200]}"
    )
