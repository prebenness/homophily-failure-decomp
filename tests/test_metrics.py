"""Unit tests for the gap-audit metric block."""

import numpy as np
import pytest

from hfd.metrics import gap_audit_metrics, macro_f1


def test_macro_f1_excludes_predicted_only_classes():
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 2, 1, 2])
    true_labels = np.unique(y_true)

    # Classes 0 and 1 each have F1 = 2/3. Predicted-only class 2 is excluded.
    assert macro_f1(y_true, y_pred, true_labels) == pytest.approx(2 / 3)


def test_gap_metrics_use_each_true_label_subset():
    y_true = np.array([0, 0, 1, 1, 2])
    y_pred = y_true.copy()
    test_idx = np.arange(4)  # Dataset class 2 is absent from this subset.
    merged_bin = np.array([0, 0, 0, 0, -1])
    tail_mask = np.array([True, True, True, True, False])

    m = gap_audit_metrics(
        y_true, y_pred, test_idx, merged_bin, tail_mask, ["[0,1]"]
    )

    # A perfect predictor scores 1 even though dataset class 2 is absent.
    assert m["whole_test_macro_f1"] == pytest.approx(1.0)
    assert m["tail_macro_f1"] == pytest.approx(1.0)
    assert m["per_bin"][0]["macro_f1"] == pytest.approx(1.0)
    assert m["whole_test_true_labels"] == [0, 1]
    assert m["tail_true_labels"] == [0, 1]
    assert m["per_bin"][0]["true_labels"] == [0, 1]
    assert "predictions are evaluated" in m["macro_f1_definition"]["label_set"]


def _setup():
    # 40 test nodes, 2 classes, 2 bins; bin 1 is the tail.
    n = 40
    y_true = np.array([0] * 20 + [1] * 20)
    test_idx = np.arange(n)
    merged_bin = np.array(([0] * 8 + [1] * 12) + ([0] * 8 + [1] * 12))
    tail_mask = merged_bin == 1
    return y_true, test_idx, merged_bin, tail_mask


def test_class_composition_per_bin():
    y_true, test_idx, merged_bin, tail_mask = _setup()
    m = gap_audit_metrics(
        y_true, y_true.copy(), test_idx, merged_bin, tail_mask,
        ["[0,0.5)", "[0.5,1]"],
    )
    assert m["per_bin"][0]["class_composition"] == {0: 8, 1: 8}
    assert m["per_bin"][1]["class_composition"] == {0: 12, 1: 12}


def test_within_class_contrast_perfect_predictions():
    y_true, test_idx, merged_bin, tail_mask = _setup()
    m = gap_audit_metrics(
        y_true, y_true.copy(), test_idx, merged_bin, tail_mask,
        ["[0,0.5)", "[0.5,1]"],
    )
    rows = m["within_class_tail_contrast"]
    assert len(rows) == 2
    for row in rows:
        assert row["tail_correct_fraction"] == 1.0
        assert row["non_tail_correct_fraction"] == 1.0
        assert row["contrast_non_tail_minus_tail"] == 0.0


def test_within_class_contrast_tail_errors_only():
    y_true, test_idx, merged_bin, tail_mask = _setup()
    y_pred = y_true.copy()
    # Break half of class 0's tail nodes (indices 8..19 are class-0 tail).
    tail_c0 = test_idx[(y_true == 0) & tail_mask]
    y_pred[tail_c0[:6]] = 1
    m = gap_audit_metrics(
        y_true, y_pred, test_idx, merged_bin, tail_mask, ["[0,0.5)", "[0.5,1]"]
    )
    row0 = next(r for r in m["within_class_tail_contrast"] if r["class"] == 0)
    assert row0["tail_correct_fraction"] == 0.5
    assert row0["non_tail_correct_fraction"] == 1.0
    assert abs(row0["contrast_non_tail_minus_tail"] - 0.5) < 1e-12


def test_small_tail_class_included_when_present_on_both_sides():
    y_true, test_idx, merged_bin, tail_mask = _setup()
    # Rewrite: class 1 keeps only 5 tail nodes but remains present on both sides.
    merged_bin = merged_bin.copy()
    c1_tail = test_idx[(y_true == 1) & (merged_bin == 1)]
    merged_bin[c1_tail[5:]] = 0
    tail_mask = merged_bin == 1
    m = gap_audit_metrics(
        y_true, y_true.copy(), test_idx, merged_bin, tail_mask,
        ["[0,0.5)", "[0.5,1]"],
    )
    classes = [r["class"] for r in m["within_class_tail_contrast"]]
    assert classes == [0, 1]


def test_class_without_non_tail_support_is_excluded():
    y_true = np.array([0, 0, 1, 1])
    y_pred = y_true.copy()
    test_idx = np.arange(y_true.size)
    merged_bin = np.array([1, 1, 0, 1])
    tail_mask = merged_bin == 1

    metrics = gap_audit_metrics(
        y_true,
        y_pred,
        test_idx,
        merged_bin,
        tail_mask,
        ["[0,0.5)", "[0.5,1]"],
    )

    assert [row["class"] for row in metrics["within_class_tail_contrast"]] == [1]
