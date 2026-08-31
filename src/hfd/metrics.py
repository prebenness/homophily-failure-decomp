"""Evaluation metrics for the gap audit.

Whole-test performance uses the dataset's standard metric. Local-homophily
diagnostics use macro-F1 (whole-test, per merged bin, tail), following the
proposal's Experiment 1 protocol. The tail gap is whole-test macro-F1 minus
tail macro-F1.

To separate local-pattern effects from class effects, the summary also
reports the class composition of every bin and a within-class contrast. For
each class represented among both tail and non-tail eligible test nodes, it
reports the fraction predicted correctly on either side.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import f1_score


MACRO_F1_DEFINITION = {
    "label_set": (
        "classes present in the evaluated subset's true labels, fixed before "
        "predictions are evaluated"
    ),
    "zero_division": 0,
}


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float((y_true == y_pred).mean())


def macro_f1(
    y_true: np.ndarray, y_pred: np.ndarray, labels: np.ndarray
) -> float:
    """Macro-F1 over an explicit class set fixed before predictions are read."""
    return float(
        f1_score(
            y_true,
            y_pred,
            labels=labels,
            average="macro",
            zero_division=0,
        )
    )


def gap_audit_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    test_idx: np.ndarray,
    merged_bin: np.ndarray,
    tail_mask: np.ndarray,
    bin_intervals: list[str],
    standard_metric: str = "accuracy",
) -> dict:
    """Experiment 1 metric block for one run's saved predictions.

    All node-level arrays are full-length (N,); metrics are computed on
    test_idx and its intersections with the fixed bins.
    """
    yt, yp = y_true[test_idx], y_pred[test_idx]
    standard = {"accuracy": accuracy}[standard_metric]
    whole_labels = np.unique(yt)
    whole_macro_f1 = macro_f1(yt, yp, whole_labels)
    tail_test = test_idx[tail_mask[test_idx]]
    tail_labels = np.unique(y_true[tail_test])
    tail_f1 = (
        macro_f1(
            y_true[tail_test],
            y_pred[tail_test],
            tail_labels,
        )
        if tail_test.size
        else float("nan")
    )

    per_bin = []
    for b, interval in enumerate(bin_intervals):
        bin_test = test_idx[merged_bin[test_idx] == b]
        bin_labels = np.unique(y_true[bin_test])
        class_counts = {
            int(c): int(n)
            for c, n in zip(*np.unique(y_true[bin_test], return_counts=True))
        }
        per_bin.append(
            {
                "bin": interval,
                "n_test": int(bin_test.size),
                "true_labels": [int(label) for label in bin_labels],
                "macro_f1": macro_f1(
                    y_true[bin_test],
                    y_pred[bin_test],
                    bin_labels,
                )
                if bin_test.size
                else float("nan"),
                "accuracy": accuracy(y_true[bin_test], y_pred[bin_test])
                if bin_test.size
                else float("nan"),
                "class_composition": class_counts,
            }
        )
    scored = [p for p in per_bin if p["n_test"] > 0]
    worst = min(scored, key=lambda p: p["macro_f1"])

    # Within-class tail contrast: per-class correct fraction, tail vs
    # non-tail eligible test nodes, for classes represented on both sides.
    correct = y_true == y_pred
    non_tail_eligible_test = test_idx[
        (merged_bin[test_idx] >= 0) & ~tail_mask[test_idx]
    ]
    within_class = []
    tail_classes, tail_counts = np.unique(y_true[tail_test], return_counts=True)
    for c, n_tail in zip(tail_classes, tail_counts):
        tail_c = tail_test[y_true[tail_test] == c]
        non_tail_c = non_tail_eligible_test[y_true[non_tail_eligible_test] == c]
        if non_tail_c.size == 0:
            continue
        tail_frac = float(correct[tail_c].mean())
        non_tail_frac = float(correct[non_tail_c].mean())
        within_class.append(
            {
                "class": int(c),
                "n_tail_test": int(n_tail),
                "n_non_tail_test": int(non_tail_c.size),
                "tail_correct_fraction": tail_frac,
                "non_tail_correct_fraction": non_tail_frac,
                "contrast_non_tail_minus_tail": non_tail_frac - tail_frac,
            }
        )

    return {
        "macro_f1_definition": dict(MACRO_F1_DEFINITION),
        "standard_metric": standard_metric,
        "whole_test_standard": standard(yt, yp),
        "whole_test_macro_f1": whole_macro_f1,
        "whole_test_true_labels": [int(label) for label in whole_labels],
        "tail_macro_f1": tail_f1,
        "tail_true_labels": [int(label) for label in tail_labels],
        "tail_gap": whole_macro_f1 - tail_f1,
        "tail_n_test": int(tail_test.size),
        "per_bin": per_bin,
        "worst_bin": worst["bin"],
        "worst_bin_macro_f1": worst["macro_f1"],
        "within_class_tail_contrast": within_class,
    }
